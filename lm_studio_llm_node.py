from __future__ import annotations

import json
import threading
import time
from typing import Any
from urllib import error, request

from aiohttp import web

from server import PromptServer

from .mcp_runtime import coerce_mcp_integration_config


LM_STUDIO_BASE_URL = "http://127.0.0.1:1234"
LM_STUDIO_MODELS_URL = f"{LM_STUDIO_BASE_URL}/v1/models"
LM_STUDIO_CHAT_URL = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"
LM_STUDIO_NATIVE_CHAT_URL = f"{LM_STUDIO_BASE_URL}/api/v1/chat"
LM_STUDIO_USER_AGENT = "ComfyUI-Anzhc-LM-Studio/1.0"
TEXT_INPUT_POINTER = "<text_input>"
DEFAULT_MODEL_OPTION = "<no active LM Studio models>"
MODEL_CACHE_TTL_SECONDS = 5.0

_MODEL_CACHE = {
    "models": None,
    "expires_at": 0.0,
}


def _request_json(url: str, method: str = "GET", payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", LM_STUDIO_USER_AGENT)
    if payload is not None:
        req.add_header("Content-Type", "application/json")

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio request failed ({exc.code}): {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"LM Studio connection failed: {exc.reason}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LM Studio returned invalid JSON: {body}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected LM Studio response: {parsed!r}")
    return parsed


def _extract_model_ids(payload: dict) -> list[str]:
    items = payload.get("data", [])
    if not isinstance(items, list):
        return []

    model_ids = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        model_ids.append(model_id)
    return model_ids


def fetch_lm_studio_models(timeout: float = 5.0, *, force_refresh: bool = False) -> list[str]:
    now = time.monotonic()
    cached_models = _MODEL_CACHE["models"]
    if (
        not force_refresh
        and isinstance(cached_models, list)
        and now < float(_MODEL_CACHE["expires_at"])
    ):
        return list(cached_models)

    payload = _request_json(LM_STUDIO_MODELS_URL, timeout=timeout)
    model_ids = _extract_model_ids(payload)

    _MODEL_CACHE["models"] = list(model_ids)
    _MODEL_CACHE["expires_at"] = now + MODEL_CACHE_TTL_SECONDS
    return model_ids


def get_lm_studio_model_options() -> list[str]:
    try:
        model_ids = fetch_lm_studio_models()
    except RuntimeError:
        model_ids = []
    return model_ids or [DEFAULT_MODEL_OPTION]


def inject_text_input(session_instruction: str, text_input: str | None) -> str:
    session_instruction = str(session_instruction or "")
    if TEXT_INPUT_POINTER not in session_instruction:
        return session_instruction
    return session_instruction.replace(TEXT_INPUT_POINTER, text_input or "")


def _extract_response_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _extract_stream_chunk_delta(payload: dict) -> dict:
    try:
        delta = payload["choices"][0]["delta"]
    except (KeyError, IndexError, TypeError):
        return {}
    if isinstance(delta, dict):
        return delta
    return {}


def _extract_stream_chunk_text(payload: dict) -> str:
    delta = _extract_stream_chunk_delta(payload)
    if not delta:
        return ""

    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _extract_response_text(content)
    return ""


def _extract_stream_chunk_reasoning_text(payload: dict) -> str:
    delta = _extract_stream_chunk_delta(payload)
    if not delta:
        return ""

    reasoning_content = delta.get("reasoning_content", "")
    if isinstance(reasoning_content, str):
        return reasoning_content
    return ""


def _iter_utf8_sse_lines(response):
    # LM Studio omits an SSE charset, so requests defaults to ISO-8859-1.
    # Decode after byte-line splitting to avoid UTF-8 bytes like E2 9C 85
    # becoming control characters that split JSON frames.
    for raw_line in response.iter_lines(decode_unicode=False):
        if raw_line is None:
            continue
        if isinstance(raw_line, bytes):
            yield raw_line.decode("utf-8", errors="replace")
        else:
            yield str(raw_line)


class _StreamingRequestState:
    def __init__(self):
        self._lock = threading.Lock()
        self._session = None
        self._response = None
        self._cancelled = False

    def bind(self, session, response) -> None:
        with self._lock:
            self._session = session
            self._response = response
            cancelled = self._cancelled

        if cancelled:
            self.cancel()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            response = self._response
            session = self._session

        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


def _call_lm_studio_openai_chat(
    model: str,
    system_prompt: str,
    session_instruction: str,
    temperature: float,
) -> str:
    import requests
    import comfy.model_management as model_management

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": session_instruction})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": True,
    }

    state = _StreamingRequestState()
    result: dict[str, object] = {
        "text_parts": [],
        "reasoning_chars": 0,
        "error": None,
    }

    def run_request() -> None:
        session = requests.Session()
        response = None
        try:
            response = session.post(
                LM_STUDIO_CHAT_URL,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                    "User-Agent": LM_STUDIO_USER_AGENT,
                },
                json=payload,
                stream=True,
                timeout=(5, 120),
            )
            response.raise_for_status()
            state.bind(session, response)

            for raw_line in _iter_utf8_sse_lines(response):
                if state.is_cancelled():
                    return

                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].lstrip()
                if data == "[DONE]":
                    break

                try:
                    chunk_payload = json.loads(data)
                except json.JSONDecodeError as exc:
                    result["error"] = RuntimeError(
                        f"LM Studio returned invalid streaming JSON: {data}"
                    )
                    return

                text = _extract_stream_chunk_text(chunk_payload)
                if text:
                    result["text_parts"].append(text)
                reasoning_text = _extract_stream_chunk_reasoning_text(chunk_payload)
                if reasoning_text:
                    result["reasoning_chars"] = int(result["reasoning_chars"]) + len(
                        reasoning_text
                    )
        except requests.RequestException as exc:
            if state.is_cancelled():
                return

            detail = ""
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                try:
                    detail = response_obj.text
                except Exception:
                    detail = ""

            if detail:
                result["error"] = RuntimeError(
                    f"LM Studio request failed ({response_obj.status_code}): {detail}"
                )
            else:
                result["error"] = RuntimeError(f"LM Studio connection failed: {exc}")
        except Exception as exc:
            if state.is_cancelled():
                return
            result["error"] = exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            try:
                session.close()
            except Exception:
                pass

    worker = threading.Thread(target=run_request, name="anzhc-lmstudio-stream", daemon=True)
    worker.start()

    while worker.is_alive():
        if model_management.processing_interrupted():
            state.cancel()
            worker.join(timeout=1.0)
            model_management.throw_exception_if_processing_interrupted()
        worker.join(timeout=0.05)

    if result["error"] is not None:
        raise result["error"]

    response_text = "".join(result["text_parts"]).strip()
    if not response_text:
        if int(result.get("reasoning_chars") or 0) > 0:
            raise RuntimeError(
                "LM Studio returned reasoning_content chunks but no final assistant content. "
                "This thinking model may need more generation tokens or thinking disabled in LM Studio."
            )
        raise RuntimeError("LM Studio returned an empty response.")
    return response_text


def _set_result_error(result: dict[str, object], exc: Exception, key: str = "error") -> None:
    if result.get(key) is None:
        result[key] = exc


def _handle_native_chat_sse_event(
    event_name: str | None,
    data_lines: list[str],
    result: dict[str, object],
) -> None:
    if not data_lines:
        return

    raw_data = "\n".join(data_lines).strip()
    if not raw_data:
        return

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        _set_result_error(
            result,
            RuntimeError(
                f"LM Studio returned invalid native streaming JSON for event {event_name or '<unknown>'}: {raw_data}"
            ),
        )
        return

    if not isinstance(payload, dict):
        _set_result_error(
            result,
            RuntimeError(f"Unexpected LM Studio native stream event: {payload!r}"),
        )
        return

    event_type = (event_name or str(payload.get("type", "")).strip() or "").strip()
    if event_type == "tool_call.failure":
        reason = payload.get("reason")
        metadata = payload.get("metadata")
        detail = reason or metadata or payload
        _set_result_error(result, RuntimeError(f"LM Studio MCP tool call failed: {detail}"), key="tool_error")
        return

    if event_type == "error":
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            detail = error_payload.get("message") or error_payload.get("type") or error_payload
        else:
            detail = error_payload or payload
        _set_result_error(result, RuntimeError(f"LM Studio native chat error: {detail}"))
        return

    if event_type == "chat.end":
        chat_result = payload.get("result")
        if not isinstance(chat_result, dict):
            _set_result_error(result, RuntimeError(f"LM Studio returned invalid chat.end payload: {payload!r}"))
            return
        result["chat_result"] = chat_result


def _extract_message_from_native_chat_result(chat_result: dict[str, Any]) -> str:
    output = chat_result.get("output")
    if not isinstance(output, list):
        raise RuntimeError(f"Unexpected LM Studio native chat result: {chat_result!r}")

    last_message = ""
    for item in output:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).strip()
        if item_type == "invalid_tool_call":
            raise RuntimeError(f"LM Studio returned invalid_tool_call output: {item!r}")
        if item_type == "error":
            raise RuntimeError(f"LM Studio returned error output: {item!r}")
        if item_type != "message":
            continue

        message_text = _extract_response_text(item.get("content", "")).strip()
        if message_text:
            last_message = message_text

    if not last_message:
        raise RuntimeError("LM Studio native chat returned no final message output.")
    return last_message


def _call_lm_studio_native_chat_with_mcp(
    model: str,
    system_prompt: str,
    session_instruction: str,
    temperature: float,
    mcp_config: dict[str, Any],
) -> str:
    import requests
    import comfy.model_management as model_management

    payload: dict[str, Any] = {
        "model": model,
        "input": session_instruction,
        "temperature": float(temperature),
        "stream": True,
        "store": False,
        "integrations": [mcp_config],
    }
    if system_prompt.strip():
        payload["system_prompt"] = system_prompt

    state = _StreamingRequestState()
    result: dict[str, object] = {
        "chat_result": None,
        "tool_error": None,
        "error": None,
    }

    def run_request() -> None:
        session = requests.Session()
        response = None
        try:
            response = session.post(
                LM_STUDIO_NATIVE_CHAT_URL,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                    "User-Agent": LM_STUDIO_USER_AGENT,
                },
                json=payload,
                stream=True,
                timeout=(5, 120),
            )
            response.raise_for_status()
            state.bind(session, response)

            current_event: str | None = None
            current_data_lines: list[str] = []

            for raw_line in _iter_utf8_sse_lines(response):
                if state.is_cancelled():
                    return

                line = raw_line.rstrip("\r")
                if not line:
                    _handle_native_chat_sse_event(current_event, current_data_lines, result)
                    current_event = None
                    current_data_lines = []
                    if result["error"] is not None or result["tool_error"] is not None:
                        return
                    continue

                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    current_data_lines.append(line[5:].lstrip())

            _handle_native_chat_sse_event(current_event, current_data_lines, result)
        except requests.RequestException as exc:
            if state.is_cancelled():
                return

            detail = ""
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                try:
                    detail = response_obj.text
                except Exception:
                    detail = ""

            if detail:
                _set_result_error(
                    result,
                    RuntimeError(f"LM Studio request failed ({response_obj.status_code}): {detail}"),
                )
            else:
                _set_result_error(result, RuntimeError(f"LM Studio connection failed: {exc}"))
        except Exception as exc:
            if state.is_cancelled():
                return
            _set_result_error(result, exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            try:
                session.close()
            except Exception:
                pass

    worker = threading.Thread(target=run_request, name="anzhc-lmstudio-native-stream", daemon=True)
    worker.start()

    while worker.is_alive():
        if model_management.processing_interrupted():
            state.cancel()
            worker.join(timeout=1.0)
            model_management.throw_exception_if_processing_interrupted()
        worker.join(timeout=0.05)

    if result["tool_error"] is not None:
        raise result["tool_error"]
    if result["error"] is not None:
        raise result["error"]

    chat_result = result["chat_result"]
    if not isinstance(chat_result, dict):
        raise RuntimeError("LM Studio native chat stream ended without a chat.end result.")

    return _extract_message_from_native_chat_result(chat_result)


def call_lm_studio_chat(
    model: str,
    system_prompt: str,
    session_instruction: str,
    temperature: float,
    mcp_config: dict[str, Any] | None = None,
) -> str:
    if mcp_config is None:
        return _call_lm_studio_openai_chat(
            model=model,
            system_prompt=system_prompt,
            session_instruction=session_instruction,
            temperature=temperature,
        )

    return _call_lm_studio_native_chat_with_mcp(
        model=model,
        system_prompt=system_prompt,
        session_instruction=session_instruction,
        temperature=temperature,
        mcp_config=mcp_config,
    )


@PromptServer.instance.routes.get("/anzhc/lm-studio/models")
async def get_lm_studio_models(_request):
    try:
        models = fetch_lm_studio_models(force_refresh=True)
    except RuntimeError as exc:
        return web.json_response(
            {
                "models": [],
                "endpoint": LM_STUDIO_MODELS_URL,
                "error": str(exc),
            },
            status=503,
        )

    return web.json_response(
        {
            "models": models,
            "endpoint": LM_STUDIO_MODELS_URL,
        }
    )


class AnzhcLMStudioLLM:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_lm_studio_model_options()
        return {
            "required": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "System prompt sent as the system message to LM Studio.",
                    },
                ),
                "session_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "User instruction for this request. Use <text_input> to inject the optional text input.",
                    },
                ),
                "model": (
                    models,
                    {
                        "default": models[0],
                        "tooltip": f"Active LM Studio model from {LM_STUDIO_MODELS_URL}.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": "Sampling temperature sent to LM Studio. Some loaded models may ignore it.",
                    },
                ),
            },
            "optional": {
                "text_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional upstream text inserted where <text_input> appears in Session instruction.",
                    },
                ),
                "mcp": (
                    "ANZHC_MCP",
                    {
                        "tooltip": (
                            "Optional MCP integration config from MCP Skills (Anzhc). "
                            "When connected, LM Studio uses /api/v1/chat with the selected ephemeral MCP server."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = (
        "Calls LM Studio's local OpenAI-compatible chat endpoint. "
        "If Session instruction contains <text_input>, the optional upstream text is inserted there. "
        "When MCP is connected, uses LM Studio's native /api/v1/chat endpoint with the selected ephemeral MCP server."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # External API node: always re-execute when queued.
        return float("nan")

    def generate(
        self,
        system_prompt: str,
        session_instruction: str,
        model: str,
        temperature: float,
        text_input: str = "",
        mcp: dict[str, Any] | None = None,
    ):
        if model == DEFAULT_MODEL_OPTION:
            raise RuntimeError(
                f"No active LM Studio models were found at {LM_STUDIO_MODELS_URL}. "
                "Load a model in LM Studio, then refresh the node."
            )

        resolved_instruction = inject_text_input(session_instruction, text_input).strip()
        if not resolved_instruction:
            raise RuntimeError(
                "Session instruction is empty after applying the optional <text_input> replacement."
            )

        resolved_mcp = coerce_mcp_integration_config(mcp) if mcp is not None else None
        response_text = call_lm_studio_chat(
            model=model,
            system_prompt=system_prompt or "",
            session_instruction=resolved_instruction,
            temperature=temperature,
            mcp_config=resolved_mcp,
        )
        return (response_text,)


NODE_CLASS_MAPPINGS = {
    "Anzhc LM Studio LLM": AnzhcLMStudioLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc LM Studio LLM": "LM Studio LLM (Anzhc)",
}
