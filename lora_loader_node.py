from __future__ import annotations

import os

from typing import Union

import folder_paths
from nodes import LoraLoader


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

    def load_loras(self, model=None, clip=None, **kwargs):
        active_names: list[str] = []

        for key, value in kwargs.items():
            key = key.upper()
            if key.startswith("LORA_") and "on" in value and "lora" in value and "strength" in value:
                strength_model = value["strength"]
                strength_clip = value["strengthTwo"] if "strengthTwo" in value else None

                if clip is None:
                    strength_clip = 0
                else:
                    strength_clip = strength_clip if strength_clip is not None else strength_model

                if value["on"] and (strength_model != 0 or strength_clip != 0):
                    lora = _get_lora_by_filename(value["lora"])
                    if model is not None and lora is not None:
                        model, clip = LoraLoader().load_lora(
                            model,
                            clip,
                            lora,
                            strength_model,
                            strength_clip,
                        )
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
                    result.append(lora_dict)
        return result


NODE_CLASS_MAPPINGS = {
    NODE_NAME: AnzhcLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: "Lora Loader (Anzhc)",
}
