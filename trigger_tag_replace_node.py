from __future__ import annotations

import re


TRIGGER_TAG_PATTERN = re.compile(r"(?<![A-Za-z0-9_])TRIGGER(?![A-Za-z0-9_])")
PAREN_CONTENT_PATTERN = re.compile(r"\(([^()]*)\)")
MULTISPACE_PATTERN = re.compile(r"\s{2,}")


def _extract_parenthetical_values(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    for raw_value in PAREN_CONTENT_PATTERN.findall(text):
        value = raw_value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)

    return values


def _cleanup_prompt_text(text: str) -> str:
    segments: list[str] = []

    for raw_segment in text.split(","):
        segment = MULTISPACE_PATTERN.sub(" ", raw_segment).strip()
        if segment:
            segments.append(segment)

    return ", ".join(segments)


def replace_trigger_tag(template_text: str, source_text: str) -> str:
    if not TRIGGER_TAG_PATTERN.search(template_text):
        return template_text

    replacements = _extract_parenthetical_values(source_text)
    if not replacements:
        return _cleanup_prompt_text(TRIGGER_TAG_PATTERN.sub("", template_text))

    return TRIGGER_TAG_PATTERN.sub(", ".join(replacements), template_text)


class AnzhcTriggerTagReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template_text": ("STRING", {"multiline": True, "default": ""}),
                "source_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = (
        "Replaces exact TRIGGER tags in the first string with the unique (...) values from "
        "the second string, joined by commas. If no trigger values are found, TRIGGER is removed."
    )

    @classmethod
    def IS_CHANGED(cls, template_text: str, source_text: str):
        # Pure string transform: only upstream input changes should invalidate it.
        return False

    def process(self, template_text: str, source_text: str):
        return (replace_trigger_tag(template_text, source_text),)


NODE_CLASS_MAPPINGS = {
    "Anzhc Trigger Tag Replace": AnzhcTriggerTagReplace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc Trigger Tag Replace": "Trigger Tag Replace (Anzhc)",
}
