from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

from prompt_optimizer_agent.json_utils import render_tool_calls_as_function_wrappers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLMPARTY_ROOT = PROJECT_ROOT.parent / "llmparty"
DEFAULT_AUDIT_PATH = Path(
    os.getenv("COMPANY_LLM_REQUEST_LOG", str(PROJECT_ROOT / "logs" / "company_api_requests.jsonl"))
)
DEFAULT_SYSTEM_PROMPT_LOG_PATH = Path(
    os.getenv(
        "COMPANY_LLM_SYSTEM_PROMPT_LOG",
        str(PROJECT_ROOT / "logs" / "company_api_system_prompts.log"),
    )
)
DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH = Path(
    os.getenv(
        "COMPANY_LLM_PREFLIGHT_SYSTEM_PROMPT_LOG",
        str(PROJECT_ROOT / "logs" / "company_api_preflight_system_prompt.log"),
    )
)
DEFAULT_LOGPROBS_LOG_PATH = Path(
    os.getenv(
        "COMPANY_LLM_LOGPROBS_LOG",
        str(PROJECT_ROOT / "logs" / "company_api_logprobs.json"),
    )
)
REQUEST_AUDIT_LOG: list[dict[str, Any]] = []
LAST_RESPONSE_DIAGNOSTICS: dict[str, Any] = {}


def generate_with_company_demo(
    messages: list[dict[str, str]],
    model: str,
    provider: str,
    url: str,
    tools: dict[str, Any] | None = None,
    tool_choice: dict[str, Any] | str | None = None,
    max_completion_tokens: int = 4096,
    temperature: float = 0.3,
    purpose: str = "chat",
    expected_system_prompt_hash: str | None = None,
    expected_prompt_version: str | None = None,
) -> str:
    """Call the locally copied generate_from_chat_demo_server.py when its deps are present."""
    entry = _record_request_audit(
        messages=messages,
        model=model,
        provider=provider,
        url=url,
        tools=tools,
        tool_choice=tool_choice,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        purpose=purpose,
        expected_system_prompt_hash=expected_system_prompt_hash,
        expected_prompt_version=expected_prompt_version,
    )
    request_logprobs = _should_request_logprobs(purpose)
    response_diagnostics: dict[str, Any] = {}
    try:
        if request_logprobs:
            response, response_diagnostics = _call_http_api(
                messages=messages,
                model=model,
                provider=provider,
                url=url,
                tools=tools,
                tool_choice=tool_choice,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                request_logprobs=True,
            )
        else:
            response = _call_copied_generate(
                messages=messages,
                model=model,
                provider=provider,
                url=url,
                tools=tools,
                tool_choice=tool_choice,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
    except (ImportError, ModuleNotFoundError):
        response, response_diagnostics = _call_http_api(
            messages=messages,
            model=model,
            provider=provider,
            url=url,
            tools=tools,
            tool_choice=tool_choice,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            request_logprobs=request_logprobs,
        )
    except RuntimeError as direct_http_error:
        if not request_logprobs:
            raise
        try:
            response = _call_copied_generate(
                messages=messages,
                model=model,
                provider=provider,
                url=url,
                tools=tools,
                tool_choice=tool_choice,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
        except (ImportError, ModuleNotFoundError):
            raise direct_http_error
        response_diagnostics = _response_diagnostics_without_logprobs(
            request_logprobs=True,
            reason="Direct HTTP request with logprobs failed; fell back to copied generate client without logprobs.",
        )
    _record_response_diagnostics(
        entry=entry,
        response=response,
        response_diagnostics=response_diagnostics,
    )
    return _extract_response_content(response)


def get_company_request_audit_log(limit: int = 20) -> list[dict[str, Any]]:
    return REQUEST_AUDIT_LOG[-limit:][::-1]


def company_request_audit_path() -> Path:
    return DEFAULT_AUDIT_PATH


def company_system_prompt_log_path() -> Path:
    return DEFAULT_SYSTEM_PROMPT_LOG_PATH


def company_preflight_system_prompt_log_path() -> Path:
    return DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH


def get_last_company_response_diagnostics() -> dict[str, Any]:
    return dict(LAST_RESPONSE_DIAGNOSTICS)


def preflight_company_request(
    messages: list[dict[str, str]],
    model: str,
    provider: str,
    url: str,
    tools: dict[str, Any] | None = None,
    tool_choice: dict[str, Any] | str | None = None,
    max_completion_tokens: int = 4096,
    temperature: float = 0.3,
    purpose: str = "chat",
    expected_system_prompt_hash: str | None = None,
    expected_prompt_version: str | None = None,
) -> None:
    preflight_purpose = purpose if purpose.endswith("_preflight") else f"{purpose}_preflight"
    entry, system_prompt, payload_prompt_values = _build_request_audit_entry(
        messages=messages,
        model=model,
        provider=provider,
        url=url,
        tools=tools,
        tool_choice=tool_choice,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        purpose=preflight_purpose,
        expected_system_prompt_hash=expected_system_prompt_hash,
        expected_prompt_version=expected_prompt_version,
    )
    try:
        if _log_full_prompts_enabled():
            _write_system_prompt_log(
                entry,
                system_prompt,
                payload_prompt_values,
                DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH,
            )
        print(
            "[company-api-preflight] "
            f"request_id={entry['request_id']} purpose={preflight_purpose} "
            f"model={provider}:{model} system_hash={entry['system_prompt_hash']} "
            f"expected={expected_system_prompt_hash or '-'} match={entry['expected_prompt_match']} "
            f"log={DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH}",
            file=sys.stderr,
        )
    except OSError:
        pass
    _raise_if_expected_prompt_mismatch(entry, DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH)


def list_company_models(url: str) -> list[str]:
    response = requests.get(url.rstrip("/") + "/v1/models", timeout=15)
    if response.status_code != 200:
        raise RuntimeError(response.text)
    data = response.json()
    if isinstance(data, list):
        return sorted(str(item) for item in data if str(item).strip())
    if isinstance(data, dict):
        for key in ("data", "models"):
            value = data.get(key)
            if isinstance(value, list):
                return sorted(_model_item_to_name(item) for item in value if _model_item_to_name(item))
    raise RuntimeError(f"Could not read model list from response: {data}")


def _model_item_to_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if isinstance(item.get("id"), str):
            return item["id"]
        if isinstance(item.get("model"), str):
            provider = item.get("provider")
            return f"{provider}:{item['model']}" if isinstance(provider, str) else item["model"]
    return ""


def _call_copied_generate(
    messages: list[dict[str, str]],
    model: str,
    provider: str,
    url: str,
    tools: dict[str, Any] | None,
    tool_choice: dict[str, Any] | str | None = None,
    max_completion_tokens: int = 4096,
    temperature: float = 0.3,
) -> Any:
    llmparty_path = str(LLMPARTY_ROOT)
    if LLMPARTY_ROOT.exists() and llmparty_path not in sys.path:
        sys.path.insert(0, llmparty_path)

    from tools.generate_from_chat_demo_server import generate

    responses = generate(
        messages=messages,
        model=model,
        provider=provider,
        url=url,
        tools=tools,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        n_samples=1,
    )
    if isinstance(responses, list) and responses:
        return responses[0]
    return responses


def _call_http_api(
    messages: list[dict[str, str]],
    model: str,
    provider: str,
    url: str,
    tools: dict[str, Any] | None,
    tool_choice: dict[str, Any] | str | None = None,
    max_completion_tokens: int = 4096,
    temperature: float = 0.3,
    request_logprobs: bool = False,
) -> tuple[Any, dict[str, Any]]:
    logprob_options = _logprob_request_options() if request_logprobs else {}
    payload = {
        "messages": messages,
        "model": model,
        "provider": provider,
        "temperature": temperature,
        "top_k": -1,
        "top_p": 0.95,
        "min_p": 0.0,
        "top_n_sigma": 0.0,
        "max_completion_tokens": max_completion_tokens,
        "tools": list(tools.values()) if tools else None,
    }
    if tool_choice:
        payload["tool_choice"] = tool_choice
    payload.update(logprob_options)
    response = requests.post(
        url.rstrip("/") + "/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    retry_without_logprobs = False
    unsupported_error = None
    if response.status_code != 200 and request_logprobs and _looks_like_logprobs_unsupported(response.text):
        unsupported_error = response.text
        retry_without_logprobs = True
        for key in ("logprobs", "top_logprobs"):
            payload.pop(key, None)
        response = requests.post(
            url.rstrip("/") + "/v1/chat/completions",
            json=payload,
            timeout=120,
        )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    data = response.json()
    if isinstance(data, dict) and ("error" in data or "detail" in data):
        error_text = str(data)
        if request_logprobs and not retry_without_logprobs and _looks_like_logprobs_unsupported(error_text):
            payload.pop("logprobs", None)
            payload.pop("top_logprobs", None)
            retry_without_logprobs = True
            unsupported_error = error_text
            response = requests.post(
                url.rstrip("/") + "/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            if response.status_code != 200:
                raise RuntimeError(response.text)
            data = response.json()
            if isinstance(data, dict) and ("error" in data or "detail" in data):
                raise RuntimeError(str(data))
        else:
            raise RuntimeError(error_text)
    diagnostics = _response_logprob_diagnostics(
        data,
        requested=request_logprobs,
        top_logprobs_requested=logprob_options.get("top_logprobs"),
        retry_without_logprobs=retry_without_logprobs,
        unsupported_error=unsupported_error,
    )
    return data, diagnostics


def _record_request_audit(
    messages: list[dict[str, str]],
    model: str,
    provider: str,
    url: str,
    tools: dict[str, Any] | None,
    tool_choice: dict[str, Any] | str | None = None,
    max_completion_tokens: int = 4096,
    temperature: float = 0.3,
    purpose: str = "chat",
    expected_system_prompt_hash: str | None = None,
    expected_prompt_version: str | None = None,
) -> dict[str, Any]:
    entry, system_prompt, payload_prompt_values = _build_request_audit_entry(
        messages=messages,
        model=model,
        provider=provider,
        url=url,
        tools=tools,
        tool_choice=tool_choice,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        purpose=purpose,
        expected_system_prompt_hash=expected_system_prompt_hash,
        expected_prompt_version=expected_prompt_version,
    )
    REQUEST_AUDIT_LOG.append(entry)
    del REQUEST_AUDIT_LOG[:-50]
    try:
        DEFAULT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEFAULT_AUDIT_PATH.open("w", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if _log_full_prompts_enabled():
            _write_system_prompt_log(entry, system_prompt, payload_prompt_values, DEFAULT_SYSTEM_PROMPT_LOG_PATH)
        print(
            "[company-api-request] "
            f"request_id={entry['request_id']} purpose={purpose} "
            f"model={provider}:{model} system_hash={entry['system_prompt_hash']} "
            f"expected={expected_system_prompt_hash or '-'} match={entry['expected_prompt_match']} "
            f"log={DEFAULT_SYSTEM_PROMPT_LOG_PATH}",
            file=sys.stderr,
        )
    except OSError:
        pass
    _raise_if_expected_prompt_mismatch(entry, DEFAULT_SYSTEM_PROMPT_LOG_PATH)
    return entry


def _build_request_audit_entry(
    messages: list[dict[str, str]],
    model: str,
    provider: str,
    url: str,
    tools: dict[str, Any] | None,
    tool_choice: dict[str, Any] | str | None = None,
    max_completion_tokens: int = 4096,
    temperature: float = 0.3,
    purpose: str = "chat",
    expected_system_prompt_hash: str | None = None,
    expected_prompt_version: str | None = None,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    system_prompt = _first_system_prompt(messages)
    payload_prompt_values = _payload_prompt_values(messages)
    actual_system_prompt_hash = _prompt_hash(system_prompt)
    expected_prompt_match = (
        expected_system_prompt_hash is None
        or actual_system_prompt_hash == expected_system_prompt_hash
    )
    entry = {
        "request_id": uuid4().hex[:10],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "expected_prompt_version": expected_prompt_version,
        "expected_system_prompt_hash": expected_system_prompt_hash,
        "expected_prompt_match": expected_prompt_match,
        "endpoint": url.rstrip("/") + "/v1/chat/completions",
        "provider": provider,
        "model": model,
        "message_count": len(messages),
        "message_roles": [str(message.get("role", "")) for message in messages],
        "system_prompt_hash": actual_system_prompt_hash,
        "system_prompt_len": len(system_prompt),
        "system_prompt_preview": _preview(system_prompt),
        "payload_prompt_refs": _payload_prompt_refs(messages),
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "tools_count": len(tools or {}),
        "tool_choice": tool_choice,
    }
    if _log_full_prompts_enabled():
        entry["system_prompt"] = system_prompt
        entry["payload_prompts"] = payload_prompt_values
    return entry, system_prompt, payload_prompt_values


def _should_request_logprobs(purpose: str) -> bool:
    if os.getenv("COMPANY_LLM_REQUEST_LOGPROBS", "1") == "0":
        return False
    return purpose in {"targeted_rerun", "full_rerun"}


def _logprob_request_options() -> dict[str, Any]:
    try:
        top_logprobs = int(os.getenv("COMPANY_LLM_TOP_LOGPROBS", "5"))
    except ValueError:
        top_logprobs = 5
    top_logprobs = min(max(top_logprobs, 0), 20)
    options: dict[str, Any] = {"logprobs": True}
    if top_logprobs > 0:
        options["top_logprobs"] = top_logprobs
    return options


def _looks_like_logprobs_unsupported(text: str) -> bool:
    lowered = text.lower()
    return (
        "logprob" in lowered
        and any(marker in lowered for marker in ("unsupported", "unknown", "unrecognized", "invalid", "extra"))
    )


def _response_diagnostics_without_logprobs(request_logprobs: bool, reason: str) -> dict[str, Any]:
    return {
        "logprobs": {
            "requested": request_logprobs,
            "available": False,
            "reason": reason,
        }
    }


def _response_logprob_diagnostics(
    data: Any,
    requested: bool,
    top_logprobs_requested: Any = None,
    retry_without_logprobs: bool = False,
    unsupported_error: str | None = None,
) -> dict[str, Any]:
    logprobs = _extract_choice_logprobs(data)
    diagnostics: dict[str, Any] = {
        "logprobs": {
            "requested": requested,
            "top_logprobs_requested": top_logprobs_requested,
            "retry_without_logprobs": retry_without_logprobs,
            "available": bool(logprobs),
        }
    }
    if unsupported_error:
        diagnostics["logprobs"]["unsupported_error"] = _preview(unsupported_error, limit=500)
    if not logprobs:
        if requested and retry_without_logprobs:
            diagnostics["logprobs"]["reason"] = "Company API rejected logprobs/top_logprobs and the request was retried without them."
        elif requested:
            diagnostics["logprobs"]["reason"] = "Company API response did not include choices[0].logprobs."
        else:
            diagnostics["logprobs"]["reason"] = "Logprobs were not requested for this request purpose."
        return diagnostics

    content_logprobs = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content_logprobs, list):
        diagnostics["logprobs"]["available"] = False
        diagnostics["logprobs"]["reason"] = "choices[0].logprobs did not include a content token list."
        return diagnostics
    token_rows = [
        row
        for row in content_logprobs
        if isinstance(row, dict) and isinstance(row.get("logprob"), (int, float))
    ]
    logprob_values = [float(row["logprob"]) for row in token_rows]
    diagnostics["logprobs"].update(
        {
            "content_token_count": len(content_logprobs),
            "scored_token_count": len(logprob_values),
            "avg_logprob": sum(logprob_values) / len(logprob_values) if logprob_values else None,
            "min_logprob": min(logprob_values) if logprob_values else None,
            "low_confidence_tokens": _low_confidence_token_samples(token_rows),
            "sample_content_logprobs": content_logprobs[:80],
        }
    )
    return diagnostics


def _extract_choice_logprobs(data: Any) -> Any:
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    return first.get("logprobs")


def _low_confidence_token_samples(token_rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        token_rows,
        key=lambda row: float(row.get("logprob") if isinstance(row.get("logprob"), (int, float)) else 0.0),
    )
    samples = []
    for row in sorted_rows[:limit]:
        sample = {
            "token": row.get("token"),
            "logprob": row.get("logprob"),
        }
        top = row.get("top_logprobs")
        if isinstance(top, list):
            sample["top_logprobs"] = top[:5]
        samples.append(sample)
    return samples


def _record_response_diagnostics(
    entry: dict[str, Any],
    response: Any,
    response_diagnostics: dict[str, Any] | None,
) -> None:
    global LAST_RESPONSE_DIAGNOSTICS
    diagnostics = dict(response_diagnostics or _response_diagnostics_without_logprobs(False, "No response diagnostics were recorded."))
    diagnostics.update(
        {
            "request_id": entry.get("request_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": entry.get("purpose"),
            "provider": entry.get("provider"),
            "model": entry.get("model"),
            "endpoint": entry.get("endpoint"),
        }
    )
    LAST_RESPONSE_DIAGNOSTICS = diagnostics
    try:
        DEFAULT_LOGPROBS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEFAULT_LOGPROBS_LOG_PATH.open("w", encoding="utf-8") as file:
            json.dump(diagnostics, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except OSError:
        pass


def _first_system_prompt(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if str(message.get("role", "")).lower() == "system":
            return str(message.get("content", ""))
    return ""


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def _preview(text: str, limit: int = 260) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _payload_prompt_refs(messages: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for message in messages:
        if str(message.get("role", "")).lower() != "user":
            continue
        content = str(message.get("content", ""))
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("system_prompt", "current_system_prompt", "optimized_prompt", "before_prompt"):
            value = payload.get(key)
            if isinstance(value, str):
                refs[key] = {
                    "hash": _prompt_hash(value),
                    "len": len(value),
                    "preview": _preview(value),
                }
    return refs


def _payload_prompt_values(messages: list[dict[str, str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for message in messages:
        if str(message.get("role", "")).lower() != "user":
            continue
        content = str(message.get("content", ""))
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("system_prompt", "current_system_prompt", "optimized_prompt", "before_prompt"):
            value = payload.get(key)
            if isinstance(value, str):
                values[key] = value
    return values


def _log_full_prompts_enabled() -> bool:
    return os.getenv("COMPANY_LLM_LOG_FULL_PROMPTS", "1") != "0"


def _write_system_prompt_log(
    entry: dict[str, Any],
    system_prompt: str,
    payload_prompt_values: dict[str, str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("=" * 96 + "\n")
        file.write(
            f"timestamp={entry['timestamp']} request_id={entry['request_id']} "
            f"purpose={entry['purpose']} model={entry['provider']}:{entry['model']}\n"
        )
        file.write(f"endpoint={entry['endpoint']}\n")
        if entry.get("expected_system_prompt_hash"):
            file.write(
                f"expected_prompt_version={entry.get('expected_prompt_version') or '-'} "
                f"expected_hash={entry['expected_system_prompt_hash']} "
                f"match={entry['expected_prompt_match']}\n"
            )
        file.write(
            f"{_first_system_message_label(entry)} hash={entry['system_prompt_hash']} "
            f"len={entry['system_prompt_len']}\n"
        )
        file.write("-" * 96 + "\n")
        file.write(system_prompt)
        file.write("\n")
        for key, value in payload_prompt_values.items():
            file.write("-" * 96 + "\n")
            file.write(f"{_payload_prompt_label(entry, key)} hash={_prompt_hash(value)} len={len(value)}\n")
            file.write("-" * 96 + "\n")
            file.write(value)
            file.write("\n")
        file.write("=" * 96 + "\n")


def _first_system_message_label(entry: dict[str, Any]) -> str:
    purpose = str(entry.get("purpose") or "")
    if purpose.startswith("judge"):
        return "first_system_message (judge instruction, not the working prompt)"
    if purpose in {"targeted_rerun", "targeted_rerun_preflight", "full_rerun", "full_rerun_preflight"}:
        return "first_system_message (outgoing working system prompt)"
    return "first_system_message"


def _payload_prompt_label(entry: dict[str, Any], key: str) -> str:
    purpose = str(entry.get("purpose") or "")
    if purpose.startswith("judge") and key == "system_prompt":
        return "payload.system_prompt (working prompt evaluated by judge)"
    if key == "current_system_prompt":
        return "payload.current_system_prompt (prompt being edited)"
    if key == "optimized_prompt":
        return "payload.optimized_prompt"
    if key == "before_prompt":
        return "payload.before_prompt"
    return f"payload.{key}"


def _raise_if_expected_prompt_mismatch(entry: dict[str, Any], log_path: Path) -> None:
    expected_hash = entry.get("expected_system_prompt_hash")
    purpose = str(entry.get("purpose") or "")
    base_purpose = purpose.removesuffix("_preflight")
    if not expected_hash or base_purpose not in {"targeted_rerun", "full_rerun"}:
        return
    if entry.get("expected_prompt_match") is True:
        return
    raise RuntimeError(
        "Outgoing system prompt hash mismatch; Company API request was blocked before send. "
        f"purpose={purpose}, expected_version={entry.get('expected_prompt_version') or '-'}, "
        f"expected={expected_hash}, actual={entry.get('system_prompt_hash')}. "
        f"See {log_path}."
    )


def _extract_response_content(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("content"), str):
            return data["content"]
        direct_tool_calls = render_tool_calls_as_function_wrappers(data.get("tool_calls"))
        if direct_tool_calls:
            return direct_tool_calls
        if isinstance(data.get("message"), dict):
            message_content = _message_content_with_tool_calls(data["message"])
            if message_content is not None:
                return message_content
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if isinstance(first.get("message"), dict):
                    message_content = _message_content_with_tool_calls(first["message"])
                    if message_content is not None:
                        return message_content
                if isinstance(first.get("delta"), dict):
                    delta_content = _message_content_with_tool_calls(first["delta"])
                    if delta_content is not None:
                        return delta_content
                if isinstance(first.get("text"), str):
                    return first["text"]
    raise RuntimeError(f"Could not extract assistant content from response: {data}")


def _message_content_with_tool_calls(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    tool_call_content = render_tool_calls_as_function_wrappers(message.get("tool_calls"))
    pieces = [
        piece
        for piece in (content, tool_call_content)
        if isinstance(piece, str) and piece.strip()
    ]
    return "\n".join(pieces) if pieces else None
