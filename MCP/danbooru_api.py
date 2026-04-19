from __future__ import annotations

import base64
import json
import re
from typing import Any
from urllib import error, parse, request

from ..mcp_runtime import (
    DANBOORU_API_KEY_HEADER,
    DANBOORU_LOGIN_HEADER,
    get_current_request_header,
)


try:
    from mcp.server.fastmcp import Context, FastMCP
except Exception:  # noqa: BLE001
    Context = Any  # type: ignore[assignment]
    FastMCP = Any  # type: ignore[assignment]


PACK_ID = "danbooru"
PACK_LABEL = "Danbooru API"
SKILLS = [
    {
        "name": "related_tag_search",
        "label": "Related Tag Search",
        "description": "Return the top 20 Danbooru related tags by cosine similarity for a tag.",
    }
]

DANBOORU_BASE_URL = "https://danbooru.donmai.us"
DANBOORU_RELATED_TAG_URL = f"{DANBOORU_BASE_URL}/related_tag.json"
DANBOORU_USER_AGENT = "ComfyUI-Anzhc-MCP-Danbooru/1.0"


def _normalize_tag(tag: str) -> str:
    normalized = re.sub(r"\s+", "_", str(tag or "").strip().lower())
    if not normalized:
        raise RuntimeError("Danbooru related tag search requires a non-empty tag.")
    return normalized


def _build_authorization_header() -> str | None:
    login = (get_current_request_header(DANBOORU_LOGIN_HEADER) or "").strip()
    api_key = (get_current_request_header(DANBOORU_API_KEY_HEADER) or "").strip()

    if not login and not api_key:
        return None
    if not login or not api_key:
        raise RuntimeError("Danbooru authentication requires both login and api key.")

    token = base64.b64encode(f"{login}:{api_key}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _extract_related_tag_names(payload: dict[str, Any]) -> list[str]:
    related_tags = payload.get("related_tags")
    if not isinstance(related_tags, list):
        raise RuntimeError(f"Unexpected Danbooru response: {payload!r}")

    tag_names: list[str] = []
    seen: set[str] = set()
    for item in related_tags:
        if not isinstance(item, dict):
            continue

        name = None
        nested_tag = item.get("tag")
        if isinstance(nested_tag, dict):
            nested_name = nested_tag.get("name")
            if isinstance(nested_name, str):
                name = nested_name

        if name is None:
            direct_name = item.get("name")
            if isinstance(direct_name, str):
                name = direct_name

        if not isinstance(name, str):
            continue

        normalized_name = name.strip()
        if not normalized_name or normalized_name in seen:
            continue

        seen.add(normalized_name)
        tag_names.append(normalized_name.replace("_", " "))
        if len(tag_names) >= 20:
            break

    if not tag_names:
        raise RuntimeError("Danbooru returned no related tags for this query.")
    return tag_names


def _fetch_related_tags(tag: str) -> str:
    auth_header = _build_authorization_header()
    query_params = parse.urlencode(
        {
            "query": tag,
            "order": "cosine",
            "limit": 20,
        }
    )
    req = request.Request(f"{DANBOORU_RELATED_TAG_URL}?{query_params}", method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", DANBOORU_USER_AGENT)
    if auth_header is not None:
        req.add_header("Authorization", auth_header)

    try:
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403 and auth_header is None:
            raise RuntimeError(
                "Danbooru blocked anonymous API access from this machine or network. "
                "Provide both Danbooru login and api key in MCP Skills to use the official endpoint."
            ) from exc
        raise RuntimeError(f"Danbooru request failed ({exc.code}): {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Danbooru connection failed: {exc.reason}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Danbooru returned invalid JSON: {body}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Danbooru response: {payload!r}")

    return ", ".join(_extract_related_tag_names(payload))


def register_tools(mcp: FastMCP) -> None:
    from mcp.server.fastmcp import Context

    @mcp.tool(
        name="related_tag_search",
        description="Return the top 20 Danbooru related tags by cosine similarity for a tag.",
    )
    async def related_tag_search(tag: str, ctx: Context) -> str:
        normalized_tag = _normalize_tag(tag)
        await ctx.info(f"Danbooru related tag search for {normalized_tag}")
        return _fetch_related_tags(normalized_tag)
