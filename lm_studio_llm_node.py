from __future__ import annotations

import base64
from io import BytesIO
import json
from math import gcd
import random
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
NODE_INFO_POINTER = "<node_info>"
DEFAULT_MODEL_OPTION = "<no active LM Studio models>"
DEFAULT_FAKE_CHAT_TURNS = "[]"
MIN_UNICODE_NOISE_LENGTH = 100
MAX_UNICODE_NOISE_LENGTH = 300
MODEL_CACHE_TTL_SECONDS = 5.0

_UNICODE_NOISE_RANDOM = random.SystemRandom()

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


def build_node_info(width: int, height: int) -> str:
    width = int(width)
    height = int(height)
    divisor = gcd(width, height)
    aspect_width = width // divisor
    aspect_height = height // divisor
    return (
        "Generation info:\n"
        f"- Resolution: {width} x {height}\n"
        f"- Aspect ratio: {aspect_width} x {aspect_height}"
    )


def inject_node_info(prompt: str, width: int, height: int) -> str:
    prompt = str(prompt or "")
    if NODE_INFO_POINTER not in prompt:
        return prompt
    return prompt.replace(NODE_INFO_POINTER, build_node_info(width, height))


def append_request_text(session_instruction: str, request_append: str | None) -> str:
    request_append = str(request_append or "").strip()
    if not request_append:
        return str(session_instruction or "")

    session_instruction = str(session_instruction or "")
    if not session_instruction.strip():
        return request_append

    return f"{session_instruction.rstrip()}\n\n{request_append}"


def generate_unicode_noise() -> str:
    length = _UNICODE_NOISE_RANDOM.randint(
        MIN_UNICODE_NOISE_LENGTH,
        MAX_UNICODE_NOISE_LENGTH,
    )
    characters: list[str] = []
    while len(characters) < length:
        character = chr(_UNICODE_NOISE_RANDOM.randrange(0x110000))
        if character.isprintable():
            characters.append(character)
    return "".join(characters)


def prepend_unicode_noise(system_prompt: str | None) -> str:
    return f"{generate_unicode_noise()}\n{system_prompt or ''}"


def parse_fake_chat_turns(fake_chat_turns: str | None) -> list[dict[str, str]]:
    raw_turns = str(fake_chat_turns or "").strip()
    if not raw_turns:
        return []

    try:
        parsed = json.loads(raw_turns)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Fake chat turns must be valid JSON.") from exc

    if not isinstance(parsed, list):
        raise RuntimeError("Fake chat turns must be a JSON array.")

    turns: list[dict[str, str]] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Fake chat turn {index} must be an object.")

        user_text = str(item.get("user", "") or "").strip()
        assistant_text = str(item.get("assistant", "") or "").strip()
        if not user_text and not assistant_text:
            continue
        if not user_text or not assistant_text:
            raise RuntimeError(
                f"Fake chat turn {index} must include both user and assistant text."
            )

        turns.append({"user": user_text, "assistant": assistant_text})

    return turns


def _image_tensor_to_data_url(image) -> str:
    try:
        from PIL import Image
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "LM Studio image input requires Pillow and torch to encode ComfyUI IMAGE tensors."
        ) from exc

    if not torch.is_tensor(image):
        raise RuntimeError(f"Expected ComfyUI IMAGE tensor, got {type(image)!r}.")
    if image.ndim != 3:
        raise RuntimeError(f"Expected single image tensor shaped (H, W, C), got {tuple(image.shape)}.")
    if image.shape[-1] < 1:
        raise RuntimeError(f"Expected image tensor with at least one channel, got {tuple(image.shape)}.")

    image_tensor = image.detach().cpu().float().clamp(0.0, 1.0)
    if image_tensor.shape[-1] == 1:
        image_tensor = image_tensor.expand(*image_tensor.shape[:-1], 3)
    elif image_tensor.shape[-1] >= 3:
        image_tensor = image_tensor[..., :3]
    else:
        raise RuntimeError(f"Expected image tensor with 1, 3, or 4 channels, got {tuple(image.shape)}.")

    image_np = (image_tensor * 255.0).round().to(torch.uint8).numpy()
    pil_image = Image.fromarray(image_np)
    image_buffer = BytesIO()
    pil_image.save(image_buffer, format="PNG")
    encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_input_to_data_urls(image) -> list[str]:
    if image is None:
        return []

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("LM Studio image input requires torch to inspect ComfyUI IMAGE tensors.") from exc

    if not torch.is_tensor(image):
        raise RuntimeError(f"Expected ComfyUI IMAGE tensor, got {type(image)!r}.")

    if image.ndim == 3:
        return [_image_tensor_to_data_url(image)]
    if image.ndim != 4:
        raise RuntimeError(
            f"Expected image tensor shaped (B, H, W, C) or (H, W, C), got {tuple(image.shape)}."
        )

    return [_image_tensor_to_data_url(image[index]) for index in range(int(image.shape[0]))]


def _build_openai_user_content(session_instruction: str, image_data_urls: list[str] | None):
    if not image_data_urls:
        return session_instruction

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": session_instruction,
        }
    ]
    for data_url in image_data_urls:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                },
            }
        )
    return content


def _build_openai_chat_messages(
    system_prompt: str,
    session_instruction: str,
    fake_chat_turns: list[dict[str, str]] | None,
    image_data_urls: list[str] | None = None,
    text_input: str | None = "",
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    for index, turn in enumerate(fake_chat_turns or []):
        user_text = turn["user"]
        if index == 0:
            user_text = inject_text_input(user_text, text_input)
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    final_user_text = text_input if fake_chat_turns else session_instruction
    messages.append(
        {
            "role": "user",
            "content": _build_openai_user_content(final_user_text or "", image_data_urls),
        }
    )
    return messages


def _build_native_chat_input(session_instruction: str, image_data_urls: list[str] | None):
    if not image_data_urls:
        return session_instruction

    input_items: list[dict[str, str]] = [
        {
            "type": "message",
            "content": session_instruction,
        }
    ]
    for data_url in image_data_urls:
        input_items.append(
            {
                "type": "image",
                "data_url": data_url,
            }
        )
    return input_items


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
    image_data_urls: list[str] | None = None,
    fake_chat_turns: list[dict[str, str]] | None = None,
    text_input: str | None = "",
    seed: int | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repeat_penalty: float | None = None,
) -> str:
    import requests
    import comfy.model_management as model_management

    messages = _build_openai_chat_messages(
        system_prompt=system_prompt,
        session_instruction=session_instruction,
        fake_chat_turns=fake_chat_turns,
        image_data_urls=image_data_urls,
        text_input=text_input,
    )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": True,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if min_p is not None:
        payload["min_p"] = float(min_p)
    if repeat_penalty is not None:
        payload["repeat_penalty"] = float(repeat_penalty)

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
                if reasoning_text.strip():
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


def _call_lm_studio_native_chat(
    model: str,
    system_prompt: str,
    session_instruction: str,
    temperature: float,
    mcp_config: dict[str, Any] | None = None,
    image_data_urls: list[str] | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repeat_penalty: float | None = None,
    max_output_tokens: int | None = None,
    reasoning_budget: int | None = None,
) -> str:
    import requests
    import comfy.model_management as model_management

    payload: dict[str, Any] = {
        "model": model,
        "input": _build_native_chat_input(session_instruction, image_data_urls),
        "temperature": float(temperature),
        "stream": True,
        "store": False,
    }
    if mcp_config is not None:
        payload["integrations"] = [mcp_config]
    if system_prompt.strip():
        payload["system_prompt"] = system_prompt
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if top_k is not None and top_k > 0:
        payload["top_k"] = int(top_k)
    if min_p is not None:
        payload["min_p"] = float(min_p)
    if repeat_penalty is not None:
        payload["repeat_penalty"] = float(repeat_penalty)
    if max_output_tokens is not None:
        payload["max_output_tokens"] = int(max_output_tokens)
    if reasoning_budget is not None:
        payload["reasoning_budget"] = int(reasoning_budget)

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
    image_data_urls: list[str] | None = None,
    fake_chat_turns: list[dict[str, str]] | None = None,
    text_input: str | None = "",
    seed: int | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repeat_penalty: float | None = None,
    reasoning_budget: int | None = None,
) -> str:
    if mcp_config is None and reasoning_budget is None:
        return _call_lm_studio_openai_chat(
            model=model,
            system_prompt=system_prompt,
            session_instruction=session_instruction,
            temperature=temperature,
            image_data_urls=image_data_urls,
            fake_chat_turns=fake_chat_turns,
            text_input=text_input,
            seed=seed,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
        )

    if fake_chat_turns:
        raise RuntimeError(
            "Fake chat turns require LM Studio's OpenAI-compatible /v1/chat/completions "
            "endpoint and cannot be used with MCP or a reasoning budget."
        )

    max_output_tokens = max_tokens
    if max_output_tokens is not None and reasoning_budget is not None:
        max_output_tokens += reasoning_budget

    return _call_lm_studio_native_chat(
        model=model,
        system_prompt=system_prompt,
        session_instruction=session_instruction,
        temperature=temperature,
        mcp_config=mcp_config,
        image_data_urls=image_data_urls,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        max_output_tokens=max_output_tokens,
        reasoning_budget=reasoning_budget,
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
                "request_append": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional text appended to the bottom of the user request.",
                    },
                ),
                "fake_chat_turns": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_FAKE_CHAT_TURNS,
                        "tooltip": (
                            "Serialized fake user/assistant chat turns inserted before the final request. "
                            "Managed by the Add Chat Turn controls."
                        ),
                    },
                ),
                "inject_noise": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "enabled",
                        "label_off": "disabled",
                        "tooltip": (
                            "Experimental: prepend 100-300 random printable Unicode characters "
                            "and a newline to the system prompt on every run."
                        ),
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": "Nucleus sampling threshold. 1 disables top-p filtering.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Limits sampling to the K most likely tokens. 0 disables top-k filtering.",
                    },
                ),
                "min_p": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": "Minimum token probability relative to the most likely token. 0 disables min-p filtering.",
                    },
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": "Penalty applied to repeated tokens. 1 applies no penalty.",
                    },
                ),
                "output_token_limit": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 1048576,
                        "step": 1,
                        "tooltip": "Maximum number of final-answer tokens. A reasoning budget is added on top.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Generation width inserted where <node_info> appears.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 8,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Generation height inserted where <node_info> appears.",
                    },
                ),
                "reasoning_budget": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1048576,
                        "step": 1,
                        "tooltip": (
                            "Maximum reasoning tokens. Added to the output token limit; "
                            "0 leaves LM Studio's reasoning budget unchanged."
                        ),
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
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional reference image or image batch sent to LM Studio VLMs as PNG data URLs.",
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
        "If the system prompt or final user request contains <node_info>, the generation resolution "
        "and reduced aspect ratio are inserted there. "
        "Request append is added to the bottom of the user request when non-empty. "
        "Fake chat turns are sent as real user/assistant messages before the final user request. "
        "Inject noise experimentally prefixes the system prompt with 100-300 random printable Unicode characters. "
        "Optional images are sent as VLM reference inputs. "
        "Reasoning budget is added to the final-answer token limit and uses LM Studio's native /api/v1/chat endpoint. "
        "When MCP is connected, the native endpoint also receives the selected ephemeral MCP server."
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
        request_append: str = "",
        fake_chat_turns: str = DEFAULT_FAKE_CHAT_TURNS,
        inject_noise: bool = False,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        output_token_limit: int = 512,
        text_input: str = "",
        image=None,
        mcp: dict[str, Any] | None = None,
        width: int = 1024,
        height: int = 1024,
        reasoning_budget: int = 0,
    ):
        if model == DEFAULT_MODEL_OPTION:
            raise RuntimeError(
                f"No active LM Studio models were found at {LM_STUDIO_MODELS_URL}. "
                "Load a model in LM Studio, then refresh the node."
            )

        resolved_instruction = inject_node_info(
            append_request_text(
                inject_text_input(session_instruction, text_input),
                request_append,
            ),
            width,
            height,
        ).strip()
        if not resolved_instruction:
            raise RuntimeError(
                "Session instruction is empty after applying the optional <text_input> replacement."
            )

        resolved_fake_chat_turns = parse_fake_chat_turns(fake_chat_turns)
        resolved_mcp = coerce_mcp_integration_config(mcp) if mcp is not None else None
        if resolved_mcp is not None and resolved_fake_chat_turns:
            raise RuntimeError(
                "Fake chat turns require LM Studio's OpenAI-compatible /v1/chat/completions "
                "endpoint and cannot be used while MCP is connected."
            )

        image_data_urls = image_input_to_data_urls(image)
        resolved_system_prompt = inject_node_info(system_prompt, width, height)
        if inject_noise:
            resolved_system_prompt = prepend_unicode_noise(resolved_system_prompt)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "system_prompt": resolved_system_prompt,
            "session_instruction": resolved_instruction,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "max_tokens": output_token_limit,
            "mcp_config": resolved_mcp,
            "image_data_urls": image_data_urls,
        }
        if reasoning_budget > 0:
            call_kwargs["reasoning_budget"] = reasoning_budget
        if resolved_fake_chat_turns:
            call_kwargs["fake_chat_turns"] = resolved_fake_chat_turns
            call_kwargs["text_input"] = text_input

        response_text = call_lm_studio_chat(**call_kwargs)
        return (response_text,)


NODE_CLASS_MAPPINGS = {
    "Anzhc LM Studio LLM": AnzhcLMStudioLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc LM Studio LLM": "LM Studio LLM (Anzhc)",
}
