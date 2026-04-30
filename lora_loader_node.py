from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re

from typing import Union

import comfy.lora
import comfy.lora_convert
import comfy.utils
import comfy.weight_adapter
import folder_paths
import torch

from .svc_utils import merge_svc_delta_tensors


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
_MERGE_METHOD_NONE = "none"
_MERGE_METHOD_SVC = "svc"
_MERGE_METHODS = (_MERGE_METHOD_NONE, _MERGE_METHOD_SVC)


@dataclass
class _LoadedLora:
    lora: str
    name: str
    value: dict
    loaded_patches: dict
    grouped_patches: dict[str, dict]
    strength_model: float
    strength_clip: float
    block_weights_enabled: bool
    merge: bool


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


def _normalize_merge_method(value) -> str:
    value = str(value or _MERGE_METHOD_NONE).lower()
    if value in _MERGE_METHODS:
        return value
    return _MERGE_METHOD_NONE


def _get_entry_patch_group(entry: _LoadedLora, patch_key) -> str | None:
    for group_name in _BLOCK_WEIGHT_KEYS:
        if patch_key in entry.grouped_patches[group_name]:
            return group_name
    return None


def _get_entry_patch_strength(entry: _LoadedLora, patch_key, strength: float) -> float:
    group_name = _get_entry_patch_group(entry, patch_key)
    if group_name is None:
        return 0.0
    group_multiplier = _get_block_weight(entry.value, group_name) if entry.block_weights_enabled else 1.0
    return strength * group_multiplier


def _tensor_for_svc_merge(tensor: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
    tensor = tensor.detach().to(device="cpu", copy=True)
    if multiplier != 1.0:
        tensor = tensor * multiplier
    return tensor


def _adapter_strength_slots(adapter) -> tuple[int, ...] | None:
    name = getattr(adapter, "name", "").lower()
    weights = getattr(adapter, "weights", None)

    if name == "lora":
        return (0,)
    if name == "loha":
        return (0,)
    if name == "lokr" and weights is not None:
        if weights[0] is not None:
            return (0,)
        if weights[3] is not None:
            return (3,)
        if weights[1] is not None:
            return (1,)
        if weights[5] is not None:
            return (5,)
        return None
    if name == "glora":
        return (0, 2)

    return None


def _merge_svc_tensor_values(
    patch_key,
    tensor_values: list[torch.Tensor | None],
    multipliers: list[float],
    ref_tensor: torch.Tensor,
) -> torch.Tensor:
    ref_cpu = ref_tensor.detach().to(device="cpu", copy=True)
    task_tensors: list[torch.Tensor] = []

    for value, multiplier in zip(tensor_values, multipliers):
        if value is None:
            task_tensors.append(torch.zeros_like(ref_cpu))
            continue
        if value.shape != ref_tensor.shape:
            raise ValueError(
                f"SVC LoRA merge shape mismatch for {patch_key!r}: "
                f"expected {tuple(ref_tensor.shape)} vs input {tuple(value.shape)}"
            )
        task_tensors.append(_tensor_for_svc_merge(value, multiplier))

    return merge_svc_delta_tensors(task_tensors, alpha=0.0).to(dtype=ref_tensor.dtype)


def _merge_svc_adapter_patch(
    patch_key,
    patch_values: list,
    multipliers: list[float],
):
    ref_adapter = next(
        (value for value, multiplier in zip(patch_values, multipliers) if value is not None and multiplier != 0.0),
        None,
    )
    if ref_adapter is None:
        ref_adapter = next((value for value in patch_values if value is not None), None)
    if ref_adapter is None:
        return None

    strength_slots = _adapter_strength_slots(ref_adapter)
    if strength_slots is None:
        raise ValueError(
            f"SVC LoRA merge does not support adapter type {getattr(ref_adapter, 'name', type(ref_adapter).__name__)!r}"
        )

    ref_weights = ref_adapter.weights
    merged_weights = []

    for index, ref_weight in enumerate(ref_weights):
        if isinstance(ref_weight, torch.Tensor):
            slot_values: list[torch.Tensor | None] = []
            for patch_value, multiplier in zip(patch_values, multipliers):
                if patch_value is None or multiplier == 0.0:
                    slot_values.append(None)
                    continue
                if type(patch_value) is not type(ref_adapter):
                    raise ValueError(
                        f"SVC LoRA merge adapter mismatch for {patch_key!r}: "
                        f"{type(ref_adapter).__name__} vs {type(patch_value).__name__}"
                    )
                value = patch_value.weights[index]
                if value is not None and not isinstance(value, torch.Tensor):
                    raise ValueError(f"SVC LoRA merge expected tensor adapter weight for {patch_key!r}")
                slot_values.append(value)

            slot_multipliers = multipliers if index in strength_slots else [1.0] * len(multipliers)
            merged_weights.append(_merge_svc_tensor_values(patch_key, slot_values, slot_multipliers, ref_weight))
        else:
            merged_weights.append(ref_weight)

    return type(ref_adapter)(set(), tuple(merged_weights))


def _normalize_patch_tuple(patch_value) -> tuple[str, tuple]:
    if not isinstance(patch_value, tuple):
        raise ValueError("patch value is not a tuple")
    if len(patch_value) == 1:
        return "diff", patch_value
    if len(patch_value) == 2:
        return patch_value[0], patch_value[1]
    raise ValueError("unsupported patch tuple")


def _merge_svc_diff_patch(
    patch_key,
    patch_values: list,
    multipliers: list[float],
):
    ref_patch = next(
        (value for value, multiplier in zip(patch_values, multipliers) if value is not None and multiplier != 0.0),
        None,
    )
    if ref_patch is None:
        ref_patch = next((value for value in patch_values if value is not None), None)
    if ref_patch is None:
        return None

    patch_type, ref_data = _normalize_patch_tuple(ref_patch)
    if patch_type != "diff":
        raise ValueError(f"SVC LoRA merge only supports diff tuple patches, got {patch_type!r} for {patch_key!r}")
    if not ref_data or not isinstance(ref_data[0], torch.Tensor):
        raise ValueError(f"SVC LoRA merge expected tensor diff patch for {patch_key!r}")

    tensor_values: list[torch.Tensor | None] = []
    for patch_value, multiplier in zip(patch_values, multipliers):
        if patch_value is None or multiplier == 0.0:
            tensor_values.append(None)
            continue
        current_type, current_data = _normalize_patch_tuple(patch_value)
        if current_type != patch_type:
            raise ValueError(
                f"SVC LoRA merge patch type mismatch for {patch_key!r}: {patch_type!r} vs {current_type!r}"
            )
        if not current_data or not isinstance(current_data[0], torch.Tensor):
            raise ValueError(f"SVC LoRA merge expected tensor diff patch for {patch_key!r}")
        tensor_values.append(current_data[0])

    merged_tensor = _merge_svc_tensor_values(patch_key, tensor_values, multipliers, ref_data[0])
    return ("diff", (merged_tensor, *ref_data[1:]))


def _merge_svc_patch_values(
    patch_key,
    patch_values: list,
    multipliers: list[float],
):
    ref_patch = next(
        (value for value, multiplier in zip(patch_values, multipliers) if value is not None and multiplier != 0.0),
        None,
    )
    if ref_patch is None:
        ref_patch = next((value for value in patch_values if value is not None), None)
    if ref_patch is None:
        return None

    if isinstance(ref_patch, comfy.weight_adapter.WeightAdapterBase):
        return _merge_svc_adapter_patch(patch_key, patch_values, multipliers)
    if isinstance(ref_patch, tuple):
        return _merge_svc_diff_patch(patch_key, patch_values, multipliers)

    raise ValueError(f"SVC LoRA merge does not support patch value for {patch_key!r}: {type(ref_patch).__name__}")


def _merge_svc_lora_patches_for_state_dict(
    entries: list[_LoadedLora],
    state_dict: dict,
    strength_attr: str,
) -> dict:
    patch_keys: set = set()
    for entry in entries:
        patch_keys.update(entry.loaded_patches.keys())

    merged_patches = {}
    with torch.no_grad():
        for patch_key in patch_keys:
            target_name = _get_patch_target_name(patch_key)
            if target_name not in state_dict:
                continue

            patch_values = [entry.loaded_patches.get(patch_key) for entry in entries]
            multipliers = [
                _get_entry_patch_strength(entry, patch_key, getattr(entry, strength_attr))
                if patch_values[index] is not None
                else 0.0
                for index, entry in enumerate(entries)
            ]
            if all(multiplier == 0.0 for multiplier in multipliers):
                continue

            merged_patch = _merge_svc_patch_values(patch_key, patch_values, multipliers)
            if merged_patch is None:
                continue
            merged_patches[patch_key] = merged_patch

    return merged_patches


def _apply_patch_dicts(
    model,
    clip,
    model_patches: dict,
    clip_patches: dict,
):
    new_model = model.clone() if model is not None else None
    new_clip = clip.clone() if clip is not None else None
    applied_model_keys = set()
    applied_clip_keys = set()

    if new_model is not None and model_patches:
        applied_model_keys = set(new_model.add_patches(model_patches, 1.0))
        for patch_key in model_patches:
            if patch_key not in applied_model_keys:
                logging.warning("NOT LOADED %s", _get_patch_target_name(patch_key) or patch_key)

    if new_clip is not None and clip_patches:
        applied_clip_keys = set(new_clip.add_patches(clip_patches, 1.0))
        for patch_key in clip_patches:
            if patch_key not in applied_clip_keys:
                logging.warning("NOT LOADED %s", _get_patch_target_name(patch_key) or patch_key)

    return new_model, new_clip, applied_model_keys, applied_clip_keys


def _warn_unapplied_lora_patches(entries: list[_LoadedLora], applied_model_keys: set, applied_clip_keys: set) -> None:
    for entry in entries:
        for patch_key in entry.loaded_patches:
            if patch_key not in applied_model_keys and patch_key not in applied_clip_keys:
                logging.warning("NOT LOADED %s", _get_patch_target_name(patch_key) or patch_key)


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

    def _get_loaded_lora_file(self, lora_path: str):
        loaded_lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                loaded_lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if loaded_lora is None:
            loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, loaded_lora)

        return loaded_lora

    def _collect_loras(self, model, clip, kwargs) -> tuple[list[_LoadedLora], list[str]]:
        active_loras: list[_LoadedLora] = []
        active_names: list[str] = []

        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        for key, value in kwargs.items():
            key = key.upper()
            if (
                isinstance(value, dict)
                and key.startswith("LORA_")
                and "on" in value
                and "lora" in value
                and "strength" in value
            ):
                try:
                    strength_model = float(value["strength"])
                except (TypeError, ValueError):
                    strength_model = 0.0
                strength_clip = value["strengthTwo"] if "strengthTwo" in value else None
                block_weights_enabled = value.get("block_weights_enabled", True)

                if clip is None:
                    strength_clip = 0.0
                else:
                    strength_clip = strength_clip if strength_clip is not None else strength_model
                    try:
                        strength_clip = float(strength_clip)
                    except (TypeError, ValueError):
                        strength_clip = 0.0

                if value["on"] and (strength_model != 0 or strength_clip != 0):
                    lora = _get_lora_by_filename(value["lora"])
                    if (model is not None or clip is not None) and lora is not None:
                        lora_path = folder_paths.get_full_path_or_raise("loras", lora)
                        loaded_lora = self._get_loaded_lora_file(lora_path)
                        converted_lora = comfy.lora_convert.convert_lora(loaded_lora)
                        loaded_patches = comfy.lora.load_lora(converted_lora, key_map)
                        total_blocks = _get_total_backbone_blocks(model, loaded_patches)
                        grouped_patches = {group: {} for group in _BLOCK_WEIGHT_KEYS}

                        for patch_key, patch_value in loaded_patches.items():
                            group = _classify_patch_group(patch_key, total_blocks)
                            grouped_patches[group][patch_key] = patch_value

                        active_loras.append(
                            _LoadedLora(
                                lora=lora,
                                name=os.path.splitext(os.path.basename(lora))[0],
                                value=value,
                                loaded_patches=loaded_patches,
                                grouped_patches=grouped_patches,
                                strength_model=strength_model,
                                strength_clip=strength_clip,
                                block_weights_enabled=block_weights_enabled,
                                merge=bool(value.get("merge", False)),
                            )
                        )
                        active_names.append(os.path.splitext(os.path.basename(lora))[0])

        return active_loras, active_names

    def _apply_lora_entry_normal(self, model, clip, entry: _LoadedLora):
        applied_model_keys = set()
        applied_clip_keys = set()
        new_model = model.clone() if model is not None else None
        new_clip = clip.clone() if clip is not None else None

        for group_name in _BLOCK_WEIGHT_KEYS:
            patches = entry.grouped_patches[group_name]
            if not patches:
                continue

            group_multiplier = _get_block_weight(entry.value, group_name) if entry.block_weights_enabled else 1.0
            model_multiplier = entry.strength_model * group_multiplier
            clip_multiplier = entry.strength_clip * group_multiplier

            if new_model is not None and model_multiplier != 0:
                applied_model_keys.update(new_model.add_patches(patches, model_multiplier))
            if new_clip is not None and clip_multiplier != 0:
                applied_clip_keys.update(new_clip.add_patches(patches, clip_multiplier))

        for patch_key in entry.loaded_patches:
            if patch_key not in applied_model_keys and patch_key not in applied_clip_keys:
                logging.warning("NOT LOADED %s", _get_patch_target_name(patch_key) or patch_key)

        return new_model, new_clip

    def _apply_svc_merged_loras(self, model, clip, entries: list[_LoadedLora]):
        merged_model_patches = {}
        merged_clip_patches = {}

        if model is not None:
            merged_model_patches = _merge_svc_lora_patches_for_state_dict(
                entries,
                model.model.state_dict(),
                "strength_model",
            )

        if clip is not None:
            merged_clip_patches = _merge_svc_lora_patches_for_state_dict(
                entries,
                clip.cond_stage_model.state_dict(),
                "strength_clip",
            )

        model, clip, applied_model_keys, applied_clip_keys = _apply_patch_dicts(
            model,
            clip,
            merged_model_patches,
            merged_clip_patches,
        )
        _warn_unapplied_lora_patches(entries, applied_model_keys, applied_clip_keys)
        return model, clip

    def load_loras(self, model=None, clip=None, **kwargs):
        merge_method = _normalize_merge_method(kwargs.get("merge_method", _MERGE_METHOD_NONE))
        active_loras, active_names = self._collect_loras(model, clip, kwargs)

        marked_loras = [entry for entry in active_loras if merge_method == _MERGE_METHOD_SVC and entry.merge]
        should_merge = len(marked_loras) >= 2
        merged_loras_applied = False

        for entry in active_loras:
            if should_merge and entry.merge:
                if not merged_loras_applied:
                    model, clip = self._apply_svc_merged_loras(model, clip, marked_loras)
                    merged_loras_applied = True
                continue
            model, clip = self._apply_lora_entry_normal(model, clip, entry)

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
                    if "merge" in lora:
                        lora_dict["merge"] = bool(lora["merge"])
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
