from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import numpy as np
from pathlib import Path
import importlib.util
import sys
import types
from typing import Any

import torch


MAX_RESOLUTION = 16384


def _load_wdv3_module():
    try:
        from . import wdv3_tagger_node as module

        return module
    except ImportError:
        module_name = "anzhc_wdv3_tagger_node_fallback"
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached

        module_path = Path(__file__).with_name("wdv3_tagger_node.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


wdv3_tagger_node = _load_wdv3_module()


def _find_loaded_module_by_file(module_path: Path):
    module_path = module_path.resolve()
    for module in list(sys.modules.values()):
        loaded_path = getattr(module, "__file__", None)
        if loaded_path is None:
            continue
        try:
            if Path(loaded_path).resolve() == module_path:
                return module
        except OSError:
            continue
    return None


def _load_tiled_diffusion_module():
    tiled_diffusion_dir = Path(__file__).resolve().parents[1] / "ComfyUI-TiledDiffusion"
    tiled_diffusion_path = tiled_diffusion_dir / "tiled_diffusion.py"
    if not tiled_diffusion_path.exists():
        raise RuntimeError(
            "Autotagged Tile Conditioning (Anzhc) requires custom_nodes/ComfyUI-TiledDiffusion "
            "to provide the MODEL patching behavior."
        )

    loaded = _find_loaded_module_by_file(tiled_diffusion_path)
    if loaded is not None:
        return loaded

    package_name = "anzhc_tiled_diffusion_bridge"
    module_name = f"{package_name}.tiled_diffusion"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(tiled_diffusion_dir)]
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(module_name, tiled_diffusion_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AutotaggedDiffusionMixin:
    def configure_autotagging(self, base_model, tile_conditionings):
        self.anzhc_base_model = base_model
        self.anzhc_tile_conditionings = tile_conditionings
        self.anzhc_tile_model_cond_cache = {}
        self.anzhc_warned_tile_count_mismatch = False

    def _process_tile_conditioning(self, conditioning, x_in: torch.Tensor):
        if not conditioning:
            return {}

        raw = conditioning[0]
        params = raw[1].copy()
        if raw[0] is not None:
            params["cross_attn"] = raw[0]

        params["device"] = x_in.device
        params["noise"] = x_in
        if len(x_in.shape) >= 4:
            params["width"] = params.get("width", x_in.shape[-1] * 8)
            params["height"] = params.get("height", x_in.shape[-2] * 8)
        params["prompt_type"] = params.get("prompt_type", "positive")

        model_conds = self.anzhc_base_model.extra_conds(**params)
        out = {}
        for key, cond_obj in model_conds.items():
            processed = cond_obj.process_cond(batch_size=1, area=None)
            out[key] = processed.cond if hasattr(processed, "cond") else processed
        return out

    def _get_tile_model_conds(self, x_in: torch.Tensor):
        cache_key = (str(x_in.device), tuple(x_in.shape[2:]))
        cached = self.anzhc_tile_model_cond_cache.get(cache_key)
        if cached is not None:
            return cached

        tile_model_conds = [
            self._process_tile_conditioning(conditioning, x_in)
            for conditioning in self.anzhc_tile_conditionings
        ]
        self.anzhc_tile_model_cond_cache[cache_key] = tile_model_conds
        return tile_model_conds

    @staticmethod
    def _repeat_dim1_to_length(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        if tensor.shape[1] == target_len:
            return tensor
        repeat_count = math.ceil(target_len / tensor.shape[1])
        repeated = tensor.repeat([1, repeat_count] + [1] * (tensor.ndim - 2))
        return repeated.narrow(1, 0, target_len)

    def _tile_start_index(self, batch_id: int) -> int:
        return sum(len(batch) for batch in self.batched_bboxes[:batch_id])

    def _inject_tile_conditioning(
        self,
        c_tile: dict[str, Any],
        x_tile: torch.Tensor,
        x_in: torch.Tensor,
        cond_or_uncond: list[int],
        batch_id: int,
    ) -> dict[str, Any]:
        rows_per_tile = len(cond_or_uncond) if len(cond_or_uncond) > 0 else x_in.shape[0]
        if rows_per_tile <= 0:
            return c_tile

        positive_offsets = [idx for idx, cond_type in enumerate(cond_or_uncond) if cond_type == 0]
        if not positive_offsets and rows_per_tile == 1:
            positive_offsets = [0]
        if not positive_offsets:
            return c_tile

        tile_model_conds = self._get_tile_model_conds(x_in)
        if not tile_model_conds:
            return c_tile

        batch_tile_count = max(1, x_tile.shape[0] // rows_per_tile)
        tile_start = self._tile_start_index(batch_id)
        tile_indexes = [min(tile_start + idx, len(tile_model_conds) - 1) for idx in range(batch_tile_count)]
        if tile_start + batch_tile_count > len(tile_model_conds) and not self.anzhc_warned_tile_count_mismatch:
            logging.warning(
                "[Anzhc Autotag Tiles] Warning: diffusion tile count is larger than prepared tag count; "
                "reusing the last tile conditioning for extra tiles."
            )
            self.anzhc_warned_tile_count_mismatch = True

        out = c_tile.copy()
        for key, value in c_tile.items():
            if not isinstance(value, torch.Tensor):
                continue

            replacements = [tile_model_conds[tile_index].get(key) for tile_index in tile_indexes]
            if not any(isinstance(replacement, torch.Tensor) for replacement in replacements):
                continue

            patched = value
            tensor_replacements = []
            for replacement in replacements:
                if not isinstance(replacement, torch.Tensor):
                    tensor_replacements.append(None)
                    continue
                replacement = replacement.to(device=value.device)
                if value.is_floating_point() and replacement.is_floating_point():
                    replacement = replacement.to(dtype=value.dtype)
                tensor_replacements.append(replacement)

            if value.ndim >= 3:
                lengths = [value.shape[1]]
                for replacement in tensor_replacements:
                    if replacement is not None and replacement.ndim == value.ndim and replacement.shape[2:] == value.shape[2:]:
                        lengths.append(replacement.shape[1])
                target_len = math.lcm(*lengths)
                patched = self._repeat_dim1_to_length(value, target_len).clone()
            elif value.ndim == 2 and "mask" in key.lower():
                lengths = [value.shape[1]]
                for replacement in tensor_replacements:
                    if replacement is not None and replacement.ndim == value.ndim:
                        lengths.append(replacement.shape[1])
                target_len = math.lcm(*lengths)
                patched = self._repeat_dim1_to_length(value.unsqueeze(-1), target_len).squeeze(-1).clone()
            else:
                patched = value.clone()

            changed = False
            for local_tile_index, replacement in enumerate(tensor_replacements):
                if replacement is None or replacement.ndim != patched.ndim:
                    continue
                if patched.ndim >= 3:
                    if replacement.shape[2:] != patched.shape[2:]:
                        continue
                    replacement = self._repeat_dim1_to_length(replacement, patched.shape[1])
                elif patched.ndim == 2 and "mask" in key.lower():
                    replacement = self._repeat_dim1_to_length(replacement.unsqueeze(-1), patched.shape[1]).squeeze(-1)
                elif replacement.shape[1:] != patched.shape[1:]:
                    continue

                for positive_offset in positive_offsets:
                    row = local_tile_index * rows_per_tile + positive_offset
                    if row < patched.shape[0]:
                        patched[row : row + 1] = replacement[:1]
                        changed = True

            if changed:
                out[key] = patched

        return out

    def __call__(self, model_function, args: dict):
        self.anzhc_tile_batch_id = 0

        def wrapped_model_function(x_tile, t_tile, **c_tile):
            batch_id = self.anzhc_tile_batch_id
            self.anzhc_tile_batch_id += 1
            injected = self._inject_tile_conditioning(
                c_tile=c_tile,
                x_tile=x_tile,
                x_in=args["input"],
                cond_or_uncond=args.get("cond_or_uncond", []),
                batch_id=batch_id,
            )
            return model_function(x_tile, t_tile, **injected)

        return super().__call__(wrapped_model_function, args)


def _make_autotagged_diffusion_impl(tiled_diffusion, method: str, base_model, tile_conditionings):
    if method == "Mixture of Diffusers":
        base_cls = tiled_diffusion.MixtureOfDiffusers
        class_name = "AnzhcAutotaggedMixtureOfDiffusers"
    elif method == "MultiDiffusion":
        base_cls = tiled_diffusion.MultiDiffusion
        class_name = "AnzhcAutotaggedMultiDiffusion"
    else:
        raise ValueError("Autotagged Tile Conditioning does not support SpotDiffusion.")

    impl_cls = type(class_name, (AutotaggedDiffusionMixin, base_cls), {})
    impl = impl_cls()
    impl.configure_autotagging(base_model=base_model, tile_conditionings=tile_conditionings)
    return impl


def apply_tiled_diffusion_model(
    model,
    method: str,
    tile_width: int,
    tile_height: int,
    tile_overlap: int,
    tile_batch_size: int,
    tile_conditionings,
):
    tiled_diffusion = _load_tiled_diffusion_module()
    impl = _make_autotagged_diffusion_impl(
        tiled_diffusion=tiled_diffusion,
        method=method,
        base_model=model.model,
        tile_conditionings=tile_conditionings,
    )

    compression = get_model_compression(model)
    impl.tile_width = tile_width // compression
    impl.tile_height = tile_height // compression
    impl.tile_overlap = tile_overlap // compression
    impl.tile_batch_size = tile_batch_size
    impl.compression = compression
    impl.width = tile_width
    impl.height = tile_height
    impl.overlap = tile_overlap

    model = model.clone()
    model.set_model_unet_function_wrapper(impl)
    model.model_options["tiled_diffusion"] = True
    return model


def get_model_compression(model) -> int:
    try:
        if "CASCADE" in str(model.model.model_type):
            return 4
    except AttributeError:
        pass
    return 8


@dataclass(frozen=True)
class TileBBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


def ceildiv(big: int, small: int) -> int:
    return -(big // -small)


def clamp_tile_overlap(tile_width: int, tile_height: int, overlap: int) -> int:
    max_overlap = max(0, min(tile_width, tile_height) - 1)
    return max(0, min(int(overlap), max_overlap))


def split_tile_bboxes(
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
) -> list[TileBBox]:
    image_width = max(1, int(image_width))
    image_height = max(1, int(image_height))
    tile_width = min(max(1, int(tile_width)), image_width)
    tile_height = min(max(1, int(tile_height)), image_height)
    overlap = clamp_tile_overlap(tile_width, tile_height, overlap)

    cols = ceildiv(image_width - overlap, tile_width - overlap)
    rows = ceildiv(image_height - overlap, tile_height - overlap)
    dx = (image_width - tile_width) / (cols - 1) if cols > 1 else 0
    dy = (image_height - tile_height) / (rows - 1) if rows > 1 else 0

    bboxes: list[TileBBox] = []
    for row in range(rows):
        y = min(int(row * dy), image_height - tile_height)
        for col in range(cols):
            x = min(int(col * dx), image_width - tile_width)
            bboxes.append(TileBBox(x=x, y=y, w=tile_width, h=tile_height))
    return bboxes


def compose_tile_prompt(prompt: str, tags: str) -> str:
    prompt = str(prompt).strip()
    tags = str(tags).strip()
    if prompt and tags:
        return f"{prompt}, {tags}"
    return prompt or tags


def limit_tags(tags: list[str], max_tags: int) -> list[str]:
    max_tags = int(max_tags)
    if max_tags <= 0:
        return tags
    return tags[:max_tags]


def _throw_if_interrupted() -> None:
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        return


def _crop_tile_image(batch_image: torch.Tensor, bbox: TileBBox) -> torch.Tensor:
    tile = batch_image[bbox.y : bbox.y2, bbox.x : bbox.x2, ...]
    if tile.shape[-1] > 3:
        tile = tile[..., :3]
    return tile


def _progress_iter(iterable, total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _scores_to_tag_string(
    scores: np.ndarray,
    bundle,
    general_threshold: float,
    character_threshold: float,
    hide_rating_tags: bool,
    remove_separator: bool,
    character_tags_first: bool,
    max_tags_per_tile: int,
) -> str:
    tags = wdv3_tagger_node._scores_to_tags(
        scores=scores,
        tag_data=bundle.tag_data,
        general_thresh=float(general_threshold),
        character_thresh=float(character_threshold),
        hide_rating_tags=bool(hide_rating_tags),
        character_tags_first=bool(character_tags_first),
    )
    tags = limit_tags(tags, int(max_tags_per_tile))
    return wdv3_tagger_node._format_tag_string(tags, bool(remove_separator))


def _run_wdv3_batch(bundle, image_arrays: list[np.ndarray], tag_batch_size: int) -> list[np.ndarray]:
    scores: list[np.ndarray] = []
    tag_batch_size = max(1, int(tag_batch_size))
    ranges = range(0, len(image_arrays), tag_batch_size)

    for start in _progress_iter(ranges, math.ceil(len(image_arrays) / tag_batch_size), "[Anzhc Autotag Tiles]"):
        batch_arrays = image_arrays[start : start + tag_batch_size]
        try:
            batch_input = np.concatenate(batch_arrays, axis=0)
            batch_scores = wdv3_tagger_node._run_model(bundle, batch_input)
            if batch_scores.shape[0] != len(batch_arrays):
                raise RuntimeError("WDv3 batch output size did not match input batch size.")
            scores.extend(batch_scores[index : index + 1] for index in range(batch_scores.shape[0]))
        except Exception as exc:
            logging.warning(
                "[Anzhc Autotag Tiles] Batch inference failed; falling back to single-tile tagging. Reason: %s",
                exc,
            )
            for image_array in batch_arrays:
                scores.append(wdv3_tagger_node._run_model(bundle, image_array))

    return scores


def tag_tile_images(
    image: torch.Tensor,
    tile_width: int,
    tile_height: int,
    tile_overlap: int,
    image_index: int,
    general_threshold: float,
    character_threshold: float,
    hide_rating_tags: bool,
    remove_separator: bool,
    character_tags_first: bool,
    max_tags_per_tile: int,
    tag_batch_size: int,
) -> list[str]:
    if image.ndim != 4:
        raise ValueError("Autotagged Tile Conditioning expects IMAGE input with shape [B, H, W, C].")

    batch_size, image_height, image_width, _channels = image.shape
    image_index = max(0, min(int(image_index), batch_size - 1))
    batch_image = image[image_index]
    bboxes = split_tile_bboxes(image_width, image_height, tile_width, tile_height, tile_overlap)
    bundle = wdv3_tagger_node._ensure_model()

    image_arrays: list[np.ndarray] = []
    for bbox in bboxes:
        _throw_if_interrupted()
        tile_image = _crop_tile_image(batch_image, bbox)
        pil_image = wdv3_tagger_node._tensor_image_to_pil(tile_image)
        image_arrays.append(wdv3_tagger_node._preprocess_pil_image(pil_image, bundle.input_size))

    score_rows = _run_wdv3_batch(bundle, image_arrays, tag_batch_size)
    return [
        _scores_to_tag_string(
            scores=scores,
            bundle=bundle,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            hide_rating_tags=hide_rating_tags,
            remove_separator=remove_separator,
            character_tags_first=character_tags_first,
            max_tags_per_tile=max_tags_per_tile,
        )
        for scores in score_rows
    ]


def build_prompt_conditionings(
    image: torch.Tensor,
    clip,
    prompt: str,
    tile_width: int,
    tile_height: int,
    tile_overlap: int,
    image_index: int,
    general_threshold: float,
    character_threshold: float,
    hide_rating_tags: bool,
    remove_separator: bool,
    character_tags_first: bool,
    max_tags_per_tile: int,
    tag_batch_size: int,
):
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None")

    tags_by_tile = tag_tile_images(
        image=image,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_overlap=tile_overlap,
        image_index=image_index,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        hide_rating_tags=hide_rating_tags,
        remove_separator=remove_separator,
        character_tags_first=character_tags_first,
        max_tags_per_tile=max_tags_per_tile,
        tag_batch_size=tag_batch_size,
    )

    base_conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    tile_conditionings = []
    tile_prompts: list[str] = []
    for tags in tags_by_tile:
        tile_prompt = compose_tile_prompt(prompt, tags)
        tile_prompts.append(tile_prompt)
        tile_conditionings.append(clip.encode_from_tokens_scheduled(clip.tokenize(tile_prompt)))

    prompt_lines = [f"{idx}: {tile_prompt}" for idx, tile_prompt in enumerate(tile_prompts)]
    return base_conditioning, tile_conditionings, "\n".join(prompt_lines)


class AnzhcAutotaggedTileConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to patch with Tiled Diffusion sampling behavior."}),
                "image": ("IMAGE", {"tooltip": "Source image used to tag each tile."}),
                "clip": ("CLIP", {"tooltip": "CLIP model used to encode tile prompts."}),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Base positive prompt. Tile tags are appended to this text.",
                    },
                ),
                "method": (
                    ["MultiDiffusion", "Mixture of Diffusers"],
                    {
                        "default": "Mixture of Diffusers",
                        "tooltip": "Tile conditioning strategy. SpotDiffusion is omitted because its shifted windows do not align with static tags.",
                    },
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 768,
                        "min": 8,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Tile width in source image pixels.",
                    },
                ),
                "tile_height": (
                    "INT",
                    {
                        "default": 768,
                        "min": 8,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Tile height in source image pixels.",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Overlap between neighboring tiles in source image pixels.",
                    },
                ),
                "tile_batch_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "tooltip": "Number of latent tiles processed together by the Tiled Diffusion model wrapper.",
                    },
                ),
                "tag_batch_size": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Number of image tiles to run through WDv3 in each ONNX batch.",
                    },
                ),
                "image_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Batch image index to process.",
                    },
                ),
                "general_threshold": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": "WDv3 threshold for general tags.",
                    },
                ),
                "character_threshold": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": "WDv3 threshold for character tags.",
                    },
                ),
                "hide_rating_tags": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Hide WDv3 rating tags from tile prompts.",
                    },
                ),
                "remove_separator": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Replace underscores in WDv3 tag names with spaces.",
                    },
                ),
                "character_tags_first": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Place character tags before general tags.",
                    },
                ),
                "max_tags_per_tile": (
                    "INT",
                    {
                        "default": 24,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Maximum WDv3 tags appended to each tile prompt. Set 0 for no limit.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "tile_prompts")
    FUNCTION = "encode"
    CATEGORY = "anzhc/sampling"
    DESCRIPTION = "Patches a model for Tiled Diffusion and returns a global positive conditioning with WDv3 autotags injected per tile."

    def encode(
        self,
        model,
        image,
        clip,
        prompt: str,
        method: str,
        tile_width: int,
        tile_height: int,
        tile_overlap: int,
        tile_batch_size: int,
        tag_batch_size: int,
        image_index: int,
        general_threshold: float,
        character_threshold: float,
        hide_rating_tags: bool,
        remove_separator: bool,
        character_tags_first: bool,
        max_tags_per_tile: int,
    ):
        conditioning, tile_conditionings, tile_prompts = build_prompt_conditionings(
            image=image,
            clip=clip,
            prompt=prompt,
            tile_width=tile_width,
            tile_height=tile_height,
            tile_overlap=tile_overlap,
            image_index=image_index,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            hide_rating_tags=hide_rating_tags,
            remove_separator=remove_separator,
            character_tags_first=character_tags_first,
            max_tags_per_tile=max_tags_per_tile,
            tag_batch_size=tag_batch_size,
        )
        patched_model = apply_tiled_diffusion_model(
            model=model,
            method=method,
            tile_width=tile_width,
            tile_height=tile_height,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            tile_conditionings=tile_conditionings,
        )
        return patched_model, conditioning, tile_prompts


NODE_CLASS_MAPPINGS = {
    "Anzhc Autotagged Tile Conditioning": AnzhcAutotaggedTileConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc Autotagged Tile Conditioning": "Autotagged Tile Conditioning (Anzhc)",
}
