from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODEL_REPO = "SmilingWolf/wd-vit-tagger-v3"
LABEL_FILE = "selected_tags.csv"
MODEL_FILE = "model.onnx"

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {
    "bundle": None,
}


class _TagData:
    def __init__(
        self,
        names: list[str],
        rating_idx: list[int],
        general_idx: list[int],
        character_idx: list[int],
    ):
        self.names = names
        self.rating_idx = rating_idx
        self.general_idx = general_idx
        self.character_idx = character_idx


class _ModelBundle:
    def __init__(self, session: Any, tag_data: _TagData, input_size: int):
        self.session = session
        self.tag_data = tag_data
        self.input_size = input_size


def _require_dependencies():
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("WDv3 Tagger (Anzhc) requires onnxruntime.") from exc

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("WDv3 Tagger (Anzhc) requires huggingface_hub.") from exc

    return ort, hf_hub_download


def _load_tag_data(csv_path: str | Path) -> _TagData:
    names: list[str] = []
    rating_idx: list[int] = []
    general_idx: list[int] = []
    character_idx: list[int] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            name = str(row.get("name", "")).strip()
            names.append(name)

            try:
                category = int(row.get("category", "-1"))
            except (TypeError, ValueError):
                category = -1

            if category == 9:
                rating_idx.append(idx)
            elif category == 0:
                general_idx.append(idx)
            elif category == 4:
                character_idx.append(idx)

    return _TagData(
        names=names,
        rating_idx=rating_idx,
        general_idx=general_idx,
        character_idx=character_idx,
    )


def _preferred_providers(ort) -> list[Any]:
    available = set(ort.get_available_providers())
    providers: list[Any] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def clear_model_cache() -> None:
    with _MODEL_LOCK:
        _MODEL_CACHE["bundle"] = None


def _ensure_model() -> _ModelBundle:
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get("bundle")
        if cached is not None:
            return cached

        ort, hf_hub_download = _require_dependencies()
        csv_path = hf_hub_download(repo_id=MODEL_REPO, filename=LABEL_FILE)
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

        tag_data = _load_tag_data(csv_path)
        session = ort.InferenceSession(model_path, providers=_preferred_providers(ort))
        input_shape = session.get_inputs()[0].shape
        input_size = int(input_shape[2])

        bundle = _ModelBundle(session=session, tag_data=tag_data, input_size=input_size)
        _MODEL_CACHE["bundle"] = bundle
        return bundle


def _preprocess_pil_image(image: Image.Image, target_size: int) -> np.ndarray:
    rgb = image.convert("RGB")
    max_dim = max(rgb.size)
    pad_left = (max_dim - rgb.size[0]) // 2
    pad_top = (max_dim - rgb.size[1]) // 2

    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(rgb, (pad_left, pad_top))
    resized = padded.resize((target_size, target_size), Image.BICUBIC)

    arr = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
    return np.expand_dims(arr, axis=0)


def _tensor_image_to_pil(image) -> Image.Image:
    image_np = image.detach().cpu().numpy()
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_np, mode="RGB")


def _run_model(bundle: _ModelBundle, image_array: np.ndarray) -> np.ndarray:
    input_name = bundle.session.get_inputs()[0].name
    return bundle.session.run(None, {input_name: image_array})[0]


def _scores_to_tags(
    scores: np.ndarray,
    tag_data: _TagData,
    general_thresh: float,
    character_thresh: float,
    hide_rating_tags: bool,
    character_tags_first: bool,
) -> list[str]:
    flat = scores.flatten()
    general = [tag_data.names[i] for i in tag_data.general_idx if flat[i] >= general_thresh]
    character = [tag_data.names[i] for i in tag_data.character_idx if flat[i] >= character_thresh]
    rating = [] if hide_rating_tags else [tag_data.names[i] for i in tag_data.rating_idx]
    if character_tags_first:
        return character + general + rating
    return general + character + rating


def _format_tag_string(tags: list[str], remove_separator: bool) -> str:
    tag_string = ", ".join(tags)
    if remove_separator:
        tag_string = tag_string.replace("_", " ")
    return tag_string


def _join_batch_tags(tag_strings: list[str]) -> str:
    return "\n".join(tag_strings)


def _throw_if_processing_interrupted() -> None:
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        return


class AnzhcWDv3Tagger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image or batch of images to tag."}),
                "general_threshold": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": "Threshold for general tags.",
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
                        "tooltip": "Threshold for character tags.",
                    },
                ),
                "hide_rating_tags": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Hide WDv3 rating tags from the output.",
                    },
                ),
                "remove_separator": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Replace underscores in tag names with spaces.",
                    },
                ),
                "character_tags_first": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Place character tags before general tags.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "tag"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = (
        "Tags IMAGE inputs with SmilingWolf WD v3 Vit-L. "
        "For batched images, outputs one comma-separated tag line per image."
    )

    def tag(
        self,
        image,
        general_threshold: float,
        character_threshold: float,
        hide_rating_tags: bool,
        remove_separator: bool,
        character_tags_first: bool,
    ):
        bundle = _ensure_model()
        tag_strings: list[str] = []

        for batch_image in image:
            _throw_if_processing_interrupted()

            pil_image = _tensor_image_to_pil(batch_image)
            image_array = _preprocess_pil_image(pil_image, bundle.input_size)
            scores = _run_model(bundle, image_array)
            tags = _scores_to_tags(
                scores=scores,
                tag_data=bundle.tag_data,
                general_thresh=float(general_threshold),
                character_thresh=float(character_threshold),
                hide_rating_tags=bool(hide_rating_tags),
                character_tags_first=bool(character_tags_first),
            )
            tag_strings.append(_format_tag_string(tags, bool(remove_separator)))

        return (_join_batch_tags(tag_strings),)


NODE_CLASS_MAPPINGS = {
    "Anzhc WDv3 Tagger": AnzhcWDv3Tagger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc WDv3 Tagger": "WDv3 Tagger (Anzhc)",
}
