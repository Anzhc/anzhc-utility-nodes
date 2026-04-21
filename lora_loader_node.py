from __future__ import annotations

import logging
import os
import re

from typing import Union

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths


class _AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class _FlexibleOptionalInputType(dict):
    def __init__(self, input_type, data: dict | None = None):
        self.input_type = input_type
        self.data = data
        if self.data is not None:
            for key, value in self.data.items():
                self[key] = value

    def __getitem__(self, key):
        if self.data is not None and key in self.data:
            return self.data[key]
        return (self.input_type,)

    def __contains__(self, _key):
        return True


ANY_TYPE = _AnyType("*")
NODE_NAME = "Anzhc Lora Loader"
_BACKBONE_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")
_BLOCK_WEIGHT_KEYS = (
    "early_blocks",
    "mid_blocks",
    "late_blocks",
    "text",
    "others",
)
_TEXT_KEY_PREFIXES = (
    "clip_",
    "t5xxl.",
    "hydit_clip.",
    "text_encoders.",
    "conditioner.",
    "cond_stage_model.",
)
_TEXT_KEY_FRAGMENTS = (
    "text_model",
    "text_projection",
    "token_embedding",
    "position_embedding",
    "embedder",
    "embedders",
    "adapter",
    "adapters",
    "caption_projection",
    "context_embedder",
    "txt_in",
    ".txt_",
    ".txt.",
)


def _get_lora_by_filename(file_path: str, lora_paths: list[str] | None = None) -> str | None:
    lora_paths = lora_paths if lora_paths is not None else folder_paths.get_filename_list("loras")

    if file_path in lora_paths:
        return file_path

    lora_paths_no_ext = [os.path.splitext(path)[0] for path in lora_paths]
    if file_path in lora_paths_no_ext:
        return lora_paths[lora_paths_no_ext.index(file_path)]

    file_path_force_no_ext = os.path.splitext(file_path)[0]
    if file_path_force_no_ext in lora_paths_no_ext:
        return lora_paths[lora_paths_no_ext.index(file_path_force_no_ext)]

    lora_filenames_only = [os.path.basename(path) for path in lora_paths]
    if file_path in lora_filenames_only:
        return lora_paths[lora_filenames_only.index(file_path)]

    file_path_force_filename = os.path.basename(file_path)
    if file_path_force_filename in lora_filenames_only:
        return lora_paths[lora_filenames_only.index(file_path_force_filename)]

    lora_filenames_and_no_ext = [os.path.splitext(os.path.basename(path))[0] for path in lora_paths]
    if file_path in lora_filenames_and_no_ext:
        return lora_paths[lora_filenames_and_no_ext.index(file_path)]

    file_path_force_filename_and_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    if file_path_force_filename_and_no_ext in lora_filenames_and_no_ext:
        return lora_paths[lora_filenames_and_no_ext.index(file_path_force_filename_and_no_ext)]

    for index, lora_path in enumerate(lora_paths):
        if file_path in lora_path:
            return lora_paths[index]

    return None


def _get_patch_target_name(patch_key) -> str | None:
    if isinstance(patch_key, str):
        return patch_key
    if isinstance(patch_key, tuple) and patch_key:
        candidate = patch_key[0]
        if isinstance(candidate, str):
            return candidate
    return None


def _normalize_backbone_name(name: str) -> str:
    if name.startswith("diffusion_model."):
        return name[len("diffusion_model."):]
    return name


def _resolve_backbone_block_stage(name: str, total_blocks: int) -> str | None:
    if total_blocks < 1:
        return None

    match = _BACKBONE_BLOCK_RE.match(name)
    if match is None:
        return None

    block_index = int(match.group(1))
    if block_index < total_blocks // 3:
        return "early_blocks"
    if block_index < (2 * total_blocks) // 3:
        return "mid_blocks"
    return "late_blocks"


def _count_backbone_blocks_from_keys(keys) -> int:
    block_indices: set[int] = set()
    for key in keys:
        name = _get_patch_target_name(key)
        if not name:
            continue
        normalized = _normalize_backbone_name(name)
        match = _BACKBONE_BLOCK_RE.match(normalized)
        if match is not None:
            block_indices.add(int(match.group(1)))
    return (max(block_indices) + 1) if block_indices else 0


def _get_total_backbone_blocks(model, loaded_patches: dict | None = None) -> int:
    if model is not None:
        total_blocks = _count_backbone_blocks_from_keys(model.model.state_dict().keys())
        if total_blocks > 0:
            return total_blocks
    if loaded_patches:
        return _count_backbone_blocks_from_keys(loaded_patches.keys())
    return 0


def _is_text_like_patch(name: str) -> bool:
    name = name.lower()
    if name.startswith(_TEXT_KEY_PREFIXES):
        return True
    return any(fragment in name for fragment in _TEXT_KEY_FRAGMENTS)


def _classify_patch_group(patch_key, total_blocks: int) -> str:
    name = _get_patch_target_name(patch_key)
    if not name:
        return "others"

    stage = _resolve_backbone_block_stage(_normalize_backbone_name(name), total_blocks)
    if stage is not None:
        return stage

    if _is_text_like_patch(name):
        return "text"

    return "others"


def _get_block_weight(value: dict, key: str) -> float:
    try:
        return float(value.get(key, 1.0))
    except (TypeError, ValueError):
        return 1.0


class AnzhcLoraLoader:
    NAME = NODE_NAME
    CATEGORY = "anzhc/utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": _FlexibleOptionalInputType(
                input_type=ANY_TYPE,
                data={
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                },
            ),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "NAME_STRING")
    FUNCTION = "load_loras"
    DESCRIPTION = (
        "Exact rgthree-style Power LoRA Loader behavior with an additional NAME_STRING output "
        "containing enabled LoRA names joined by commas."
    )

    def __init__(self):
        self.loaded_lora = None

    def load_loras(self, model=None, clip=None, **kwargs):
        active_names: list[str] = []

        for key, value in kwargs.items():
            key = key.upper()
            if key.startswith("LORA_") and "on" in value and "lora" in value and "strength" in value:
                strength_model = value["strength"]
                strength_clip = value["strengthTwo"] if "strengthTwo" in value else None
                block_weights_enabled = value.get("block_weights_enabled", True)

                if clip is None:
                    strength_clip = 0
                else:
                    strength_clip = strength_clip if strength_clip is not None else strength_model

                if value["on"] and (strength_model != 0 or strength_clip != 0):
                    lora = _get_lora_by_filename(value["lora"])
                    if (model is not None or clip is not None) and lora is not None:
                        lora_path = folder_paths.get_full_path_or_raise("loras", lora)

                        loaded_lora = None
                        if self.loaded_lora is not None:
                            if self.loaded_lora[0] == lora_path:
                                loaded_lora = self.loaded_lora[1]
                            else:
                                self.loaded_lora = None

                        if loaded_lora is None:
                            loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                            self.loaded_lora = (lora_path, loaded_lora)

                        key_map = {}
                        if model is not None:
                            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
                        if clip is not None:
                            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

                        converted_lora = comfy.lora_convert.convert_lora(loaded_lora)
                        loaded_patches = comfy.lora.load_lora(converted_lora, key_map)
                        total_blocks = _get_total_backbone_blocks(model, loaded_patches)
                        grouped_patches = {group: {} for group in _BLOCK_WEIGHT_KEYS}

                        for patch_key, patch_value in loaded_patches.items():
                            group = _classify_patch_group(patch_key, total_blocks)
                            grouped_patches[group][patch_key] = patch_value

                        applied_model_keys = set()
                        applied_clip_keys = set()
                        new_model = model.clone() if model is not None else None
                        new_clip = clip.clone() if clip is not None else None

                        for group_name in _BLOCK_WEIGHT_KEYS:
                            patches = grouped_patches[group_name]
                            if not patches:
                                continue

                            group_multiplier = _get_block_weight(value, group_name) if block_weights_enabled else 1.0
                            model_multiplier = strength_model * group_multiplier
                            clip_multiplier = strength_clip * group_multiplier

                            if new_model is not None and model_multiplier != 0:
                                applied_model_keys.update(new_model.add_patches(patches, model_multiplier))
                            if new_clip is not None and clip_multiplier != 0:
                                applied_clip_keys.update(new_clip.add_patches(patches, clip_multiplier))

                        for patch_key in loaded_patches:
                            if patch_key not in applied_model_keys and patch_key not in applied_clip_keys:
                                logging.warning("NOT LOADED %s", _get_patch_target_name(patch_key) or patch_key)

                        model, clip = new_model, new_clip
                        active_names.append(os.path.splitext(os.path.basename(lora))[0])

        return (model, clip, ", ".join(active_names))

    @classmethod
    def get_enabled_loras_from_prompt_node(
        cls,
        prompt_node: dict,
    ) -> list[dict[str, Union[str, float]]]:
        result = []
        for name, lora in prompt_node["inputs"].items():
            if name.startswith("lora_") and lora["on"]:
                lora_file = _get_lora_by_filename(lora["lora"])
                if lora_file is not None:
                    lora_dict = {
                        "name": lora["lora"],
                        "strength": lora["strength"],
                        "path": folder_paths.get_full_path("loras", lora_file),
                    }
                    if "strengthTwo" in lora:
                        lora_dict["strength_clip"] = lora["strengthTwo"]
                    for block_weight_key in _BLOCK_WEIGHT_KEYS:
                        if block_weight_key in lora:
                            lora_dict[block_weight_key] = lora[block_weight_key]
                    result.append(lora_dict)
        return result


NODE_CLASS_MAPPINGS = {
    NODE_NAME: AnzhcLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: "Lora Loader (Anzhc)",
}
