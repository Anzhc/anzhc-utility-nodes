from __future__ import annotations


def resolve_dimensions(width: int, height: int) -> tuple[int, int]:
    return (int(width), int(height))


class AnzhcResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Output width.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Output height.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = "Compact width/height utility node with matching INT outputs."

    def get_resolution(self, width: int, height: int):
        return resolve_dimensions(width, height)


NODE_CLASS_MAPPINGS = {
    "Anzhc Resolution": AnzhcResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc Resolution": "Resolution (Anzhc)",
}
