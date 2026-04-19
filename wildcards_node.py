import random
import re
from pathlib import Path


TRIGGER_PATTERN = re.compile(r"__([A-Za-z0-9_\-./]+?)__")
WILDCARDS_DIR = Path(__file__).resolve().parent / "wildcards"
MAX_RESOLUTION_PASSES = 10


def _read_text_lines(file_path: Path):
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            content = file_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return []

    entries = []
    for line in content.splitlines():
        value = line.strip()
        if value:
            entries.append(value)
    return entries


def _load_wildcards():
    wildcard_map = {}
    if not WILDCARDS_DIR.exists():
        return wildcard_map

    for txt_file in WILDCARDS_DIR.rglob("*.txt"):
        entries = _read_text_lines(txt_file)
        if not entries:
            continue

        relative_key = txt_file.relative_to(WILDCARDS_DIR).with_suffix("").as_posix().lower()
        basename_key = txt_file.stem.lower()

        wildcard_map.setdefault(relative_key, []).extend(entries)
        if basename_key != relative_key:
            wildcard_map.setdefault(basename_key, []).extend(entries)
    return wildcard_map


def _resolve_wildcards(text: str, rng: random.Random):
    wildcards = _load_wildcards()
    if not wildcards:
        return text

    def replace_match(match):
        key = match.group(1).strip().lower()
        choices = wildcards.get(key)
        if not choices:
            return match.group(0)
        return rng.choice(choices)

    resolved = text
    for _ in range(MAX_RESOLUTION_PASSES):
        updated = TRIGGER_PATTERN.sub(replace_match, resolved)
        if updated == resolved:
            break
        resolved = updated
    return resolved


class AnzhcWildcards:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "anzhc/utility"

    def process(self, text, seed):
        rng = random.Random(int(seed))
        return (_resolve_wildcards(text, rng),)


NODE_CLASS_MAPPINGS = {
    "Anzhc Wildcards": AnzhcWildcards,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc Wildcards": "Wildcards (Anzhc)",
}
