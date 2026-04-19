from __future__ import annotations

from aiohttp import web

from server import PromptServer

from .mcp_runtime import (
    DEFAULT_MCP_SERVER_OPTION,
    DEFAULT_MCP_SKILL_OPTION,
    build_ephemeral_mcp_config,
    get_mcp_server_options,
    get_mcp_skill_options,
    get_mcp_skills_payload,
)


@PromptServer.instance.routes.get("/anzhc/mcp/skills")
async def get_anzhc_mcp_skills(_request):
    try:
        payload = get_mcp_skills_payload(force_refresh=True)
    except RuntimeError as exc:
        return web.json_response(
            {
                "servers": [],
                "error": str(exc),
            },
            status=503,
        )

    return web.json_response(payload)


class AnzhcMCPSkills:
    @classmethod
    def INPUT_TYPES(cls):
        server_options = get_mcp_server_options()
        default_server = server_options[0]
        skill_options = get_mcp_skill_options(None if default_server == DEFAULT_MCP_SERVER_OPTION else default_server)

        return {
            "required": {
                "mcp_server": (
                    server_options,
                    {
                        "default": default_server,
                        "tooltip": "MCP skill pack to expose to LM Studio as an ephemeral MCP server.",
                    },
                ),
                "skill": (
                    skill_options,
                    {
                        "default": skill_options[0],
                        "tooltip": "Tool from the selected MCP skill pack that LM Studio is allowed to call.",
                    },
                ),
                "danbooru_login": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional Danbooru login for authenticated official API requests.",
                    },
                ),
                "danbooru_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional Danbooru API key used together with the Danbooru login.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("ANZHC_MCP",)
    RETURN_NAMES = ("mcp",)
    FUNCTION = "build"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = (
        "Builds an ephemeral MCP integration config for LM Studio using locally hosted Anzhc MCP skill packs."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def build(
        self,
        mcp_server: str,
        skill: str,
        danbooru_login: str = "",
        danbooru_api_key: str = "",
    ):
        if mcp_server == DEFAULT_MCP_SERVER_OPTION:
            raise RuntimeError("No MCP skill packs were found in custom_nodes/anzhc-utility-nodes/MCP.")
        if skill == DEFAULT_MCP_SKILL_OPTION:
            raise RuntimeError(f"No MCP skills are available for server '{mcp_server}'.")

        return (
            build_ephemeral_mcp_config(
                mcp_server,
                skill,
                danbooru_login=danbooru_login,
                danbooru_api_key=danbooru_api_key,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "Anzhc MCP Skills": AnzhcMCPSkills,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc MCP Skills": "MCP Skills (Anzhc)",
}
