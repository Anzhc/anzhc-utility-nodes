from __future__ import annotations

import contextvars
import importlib
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MCP_SERVER_HOST = "127.0.0.1"
MCP_SERVER_ENDPOINT_PATH = "/mcp"
MCP_SERVER_LABEL_PREFIX = "anzhc"
DEFAULT_MCP_SERVER_OPTION = "<no MCP skill packs>"
DEFAULT_MCP_SKILL_OPTION = "<no MCP skills>"
DANBOORU_LOGIN_HEADER = "X-Anzhc-Danbooru-Login"
DANBOORU_API_KEY_HEADER = "X-Anzhc-Danbooru-Api-Key"

_MCP_DIR = Path(__file__).resolve().parent / "MCP"
_PACK_DISCOVERY_LOCK = threading.Lock()
_PACK_DISCOVERY_CACHE: list["McpPackMetadata"] | None = None
_CURRENT_HTTP_HEADERS = contextvars.ContextVar("anzhc_mcp_http_headers", default={})
_MCP_SERVER_LOCK = threading.Lock()
_MCP_SERVER_STATE: dict[str, Any] = {
    "thread": None,
    "server": None,
    "socket": None,
    "url": None,
    "error": None,
}


@dataclass(frozen=True)
class McpSkillMetadata:
    name: str
    label: str
    description: str


@dataclass(frozen=True)
class McpPackMetadata:
    pack_id: str
    label: str
    module_stem: str
    skills: tuple[McpSkillMetadata, ...]


class _HeaderContextASGIWrapper:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        token = _CURRENT_HTTP_HEADERS.set(headers)
        try:
            await self.app(scope, receive, send)
        finally:
            _CURRENT_HTTP_HEADERS.reset(token)


def get_current_request_headers() -> dict[str, str]:
    current_headers = _CURRENT_HTTP_HEADERS.get({})
    return dict(current_headers) if isinstance(current_headers, dict) else {}


def get_current_request_header(name: str) -> str | None:
    if not name:
        return None
    return get_current_request_headers().get(name.lower())


def _humanize_module_stem(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().title() or stem


def _import_pack_module(module_stem: str):
    return importlib.import_module(f".MCP.{module_stem}", package=__package__)


def _discover_mcp_packs_uncached() -> list[McpPackMetadata]:
    packs: list[McpPackMetadata] = []
    if not _MCP_DIR.exists():
        return packs

    for module_path in sorted(_MCP_DIR.glob("*.py")):
        if module_path.name == "__init__.py" or module_path.stem.startswith("_"):
            continue

        try:
            module = _import_pack_module(module_path.stem)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load MCP pack '{module_path.stem}': {exc}") from exc

        pack_id = str(getattr(module, "PACK_ID", module_path.stem)).strip()
        pack_label = str(getattr(module, "PACK_LABEL", _humanize_module_stem(module_path.stem))).strip()
        raw_skills = getattr(module, "SKILLS", [])
        if not pack_id:
            raise RuntimeError(f"MCP pack '{module_path.stem}' is missing PACK_ID.")
        if not isinstance(raw_skills, list):
            raise RuntimeError(f"MCP pack '{pack_id}' must define SKILLS as a list.")

        skills: list[McpSkillMetadata] = []
        for raw_skill in raw_skills:
            if not isinstance(raw_skill, dict):
                raise RuntimeError(f"MCP pack '{pack_id}' contains an invalid skill definition.")

            skill_name = str(raw_skill.get("name", "")).strip()
            skill_label = str(raw_skill.get("label", skill_name)).strip()
            skill_description = str(raw_skill.get("description", "")).strip()
            if not skill_name:
                raise RuntimeError(f"MCP pack '{pack_id}' contains a skill without a name.")

            skills.append(
                McpSkillMetadata(
                    name=skill_name,
                    label=skill_label or skill_name,
                    description=skill_description,
                )
            )

        if not skills:
            raise RuntimeError(f"MCP pack '{pack_id}' does not define any skills.")
        if not callable(getattr(module, "register_tools", None)):
            raise RuntimeError(f"MCP pack '{pack_id}' is missing register_tools(mcp).")

        packs.append(
            McpPackMetadata(
                pack_id=pack_id,
                label=pack_label or pack_id,
                module_stem=module_path.stem,
                skills=tuple(skills),
            )
        )

    packs.sort(key=lambda pack: (pack.label.lower(), pack.pack_id.lower()))
    return packs


def discover_mcp_packs(*, force_refresh: bool = False) -> list[McpPackMetadata]:
    global _PACK_DISCOVERY_CACHE

    with _PACK_DISCOVERY_LOCK:
        if _PACK_DISCOVERY_CACHE is not None and not force_refresh:
            return list(_PACK_DISCOVERY_CACHE)

        discovered = _discover_mcp_packs_uncached()
        _PACK_DISCOVERY_CACHE = list(discovered)
        return list(discovered)


def _get_pack_metadata(pack_id: str) -> McpPackMetadata:
    normalized_pack_id = str(pack_id or "").strip()
    for pack in discover_mcp_packs():
        if pack.pack_id == normalized_pack_id:
            return pack
    raise RuntimeError(f"Unknown MCP skill pack: {normalized_pack_id or '<empty>'}")


def _get_skill_metadata(pack: McpPackMetadata, skill_name: str) -> McpSkillMetadata:
    normalized_skill_name = str(skill_name or "").strip()
    for skill in pack.skills:
        if skill.name == normalized_skill_name:
            return skill
    raise RuntimeError(
        f"Unknown MCP skill '{normalized_skill_name or '<empty>'}' for pack '{pack.pack_id}'."
    )


def get_mcp_server_options() -> list[str]:
    try:
        packs = discover_mcp_packs()
    except RuntimeError:
        packs = []
    return [pack.pack_id for pack in packs] or [DEFAULT_MCP_SERVER_OPTION]


def get_mcp_skill_options(pack_id: str | None = None) -> list[str]:
    try:
        packs = discover_mcp_packs()
    except RuntimeError:
        packs = []

    if not packs:
        return [DEFAULT_MCP_SKILL_OPTION]

    normalized_pack_id = str(pack_id or "").strip()
    selected_pack = next((pack for pack in packs if pack.pack_id == normalized_pack_id), packs[0])
    return [skill.name for skill in selected_pack.skills] or [DEFAULT_MCP_SKILL_OPTION]


def get_mcp_skills_payload(*, force_refresh: bool = False) -> dict[str, Any]:
    packs = discover_mcp_packs(force_refresh=force_refresh)
    return {
        "servers": [
            {
                "id": pack.pack_id,
                "label": pack.label,
                "skills": [
                    {
                        "name": skill.name,
                        "label": skill.label,
                        "description": skill.description,
                    }
                    for skill in pack.skills
                ],
            }
            for pack in packs
        ],
        "endpoint": MCP_SERVER_ENDPOINT_PATH,
    }


def _require_mcp_dependencies():
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Anzhc MCP support requires the 'mcp' Python package. "
            "Install custom_nodes/anzhc-utility-nodes/requirements.txt."
        ) from exc

    try:
        import uvicorn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Anzhc MCP support requires uvicorn.") from exc

    return FastMCP, uvicorn


def _build_fastmcp_app():
    FastMCP, _ = _require_mcp_dependencies()

    mcp = FastMCP(
        "anzhc-local-mcp",
        instructions=(
            "Local MCP tools exposed by Anzhc Utility Nodes for LM Studio. "
            "Use only the tools the client has enabled for the request."
        ),
    )

    for pack in discover_mcp_packs():
        module = _import_pack_module(pack.module_stem)
        module.register_tools(mcp)

    return _HeaderContextASGIWrapper(mcp.streamable_http_app())


def _clear_mcp_server_state_locked() -> None:
    _MCP_SERVER_STATE["thread"] = None
    _MCP_SERVER_STATE["server"] = None
    _MCP_SERVER_STATE["socket"] = None
    _MCP_SERVER_STATE["url"] = None
    _MCP_SERVER_STATE["error"] = None


def ensure_local_mcp_server_url(timeout: float = 10.0) -> str:
    _, uvicorn = _require_mcp_dependencies()

    with _MCP_SERVER_LOCK:
        thread = _MCP_SERVER_STATE.get("thread")
        server = _MCP_SERVER_STATE.get("server")
        url = _MCP_SERVER_STATE.get("url")
        error = _MCP_SERVER_STATE.get("error")

        if error is not None:
            raise RuntimeError(f"Local MCP server failed to start previously: {error}") from error

        if thread is not None and server is not None and thread.is_alive():
            existing_url = str(url or "").strip()
            if existing_url:
                pass
        else:
            _clear_mcp_server_state_locked()

            app = _build_fastmcp_app()
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((MCP_SERVER_HOST, 0))
            server_socket.listen(128)
            server_socket.setblocking(False)
            port = server_socket.getsockname()[1]
            server_url = f"http://{MCP_SERVER_HOST}:{port}{MCP_SERVER_ENDPOINT_PATH}"

            config = uvicorn.Config(
                app,
                host=MCP_SERVER_HOST,
                port=port,
                log_level="warning",
                access_log=False,
                lifespan="on",
            )
            uvicorn_server = uvicorn.Server(config)

            def run_server() -> None:
                try:
                    uvicorn_server.run(sockets=[server_socket])
                except Exception as exc:  # noqa: BLE001
                    with _MCP_SERVER_LOCK:
                        _MCP_SERVER_STATE["error"] = exc
                finally:
                    try:
                        server_socket.close()
                    except Exception:
                        pass

            thread = threading.Thread(target=run_server, name="anzhc-local-mcp", daemon=True)
            _MCP_SERVER_STATE["thread"] = thread
            _MCP_SERVER_STATE["server"] = uvicorn_server
            _MCP_SERVER_STATE["socket"] = server_socket
            _MCP_SERVER_STATE["url"] = server_url
            thread.start()

    deadline = time.monotonic() + max(timeout, 1.0)
    while time.monotonic() < deadline:
        with _MCP_SERVER_LOCK:
            server = _MCP_SERVER_STATE.get("server")
            thread = _MCP_SERVER_STATE.get("thread")
            url = _MCP_SERVER_STATE.get("url")
            error = _MCP_SERVER_STATE.get("error")

        if error is not None:
            raise RuntimeError(f"Local MCP server failed to start: {error}") from error
        if server is not None and getattr(server, "started", False):
            resolved_url = str(url or "").strip()
            if resolved_url:
                return resolved_url
        if thread is not None and not thread.is_alive():
            break
        time.sleep(0.05)

    raise RuntimeError("Local MCP server failed to become ready in time.")


def build_ephemeral_mcp_config(
    mcp_server: str,
    skill: str,
    *,
    danbooru_login: str = "",
    danbooru_api_key: str = "",
) -> dict[str, Any]:
    pack = _get_pack_metadata(mcp_server)
    skill_meta = _get_skill_metadata(pack, skill)
    headers: dict[str, str] = {}

    danbooru_login = str(danbooru_login or "").strip()
    danbooru_api_key = str(danbooru_api_key or "").strip()
    if pack.pack_id == "danbooru" and (danbooru_login or danbooru_api_key):
        if not danbooru_login or not danbooru_api_key:
            raise RuntimeError("Danbooru auth requires both login and api key.")
        headers[DANBOORU_LOGIN_HEADER] = danbooru_login
        headers[DANBOORU_API_KEY_HEADER] = danbooru_api_key

    config: dict[str, Any] = {
        "type": "ephemeral_mcp",
        "server_label": f"{MCP_SERVER_LABEL_PREFIX}-{pack.pack_id}",
        "server_url": ensure_local_mcp_server_url(),
        "allowed_tools": [skill_meta.name],
    }
    if headers:
        config["headers"] = headers
    return config


def coerce_mcp_integration_config(mcp_config: Any) -> dict[str, Any]:
    if not isinstance(mcp_config, dict):
        raise RuntimeError("Invalid ANZHC_MCP input: expected a config object.")

    config_type = str(mcp_config.get("type", "")).strip()
    server_label = str(mcp_config.get("server_label", "")).strip()
    server_url = str(mcp_config.get("server_url", "")).strip()
    allowed_tools = mcp_config.get("allowed_tools", [])
    headers = mcp_config.get("headers", {})

    if config_type != "ephemeral_mcp":
        raise RuntimeError(f"Unsupported MCP config type: {config_type or '<empty>'}")
    if not server_label or not server_url:
        raise RuntimeError("Invalid ANZHC_MCP input: missing server_label or server_url.")
    if not isinstance(allowed_tools, list) or not all(isinstance(item, str) and item.strip() for item in allowed_tools):
        raise RuntimeError("Invalid ANZHC_MCP input: allowed_tools must be a non-empty string list.")
    if headers is None:
        headers = {}
    if not isinstance(headers, dict):
        raise RuntimeError("Invalid ANZHC_MCP input: headers must be an object.")

    normalized_headers = {
        str(key): str(value)
        for key, value in headers.items()
        if str(key).strip() and value is not None
    }

    normalized_allowed_tools = [
        item.strip() for item in allowed_tools if isinstance(item, str) and item.strip()
    ]
    if not normalized_allowed_tools:
        raise RuntimeError("Invalid ANZHC_MCP input: allowed_tools must contain at least one tool name.")

    normalized: dict[str, Any] = {
        "type": "ephemeral_mcp",
        "server_label": server_label,
        "server_url": server_url,
        "allowed_tools": normalized_allowed_tools,
    }
    if normalized_headers:
        normalized["headers"] = normalized_headers
    return normalized
