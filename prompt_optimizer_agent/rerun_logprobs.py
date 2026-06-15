from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from prompt_optimizer_agent.json_utils import (
    ConversationData,
    Interaction,
    function_call_wrappers_to_tool_calls,
    render_tool_calls_as_function_wrappers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPANY_URL = os.getenv("COMPANY_LLM_URL", "http://192.168.101.15:9898")
DEFAULT_COMPANY_PROVIDER = os.getenv("COMPANY_LLM_PROVIDER", "openai_api_like")
DEFAULT_COMPANY_MODEL = os.getenv("COMPANY_LLM_MODEL", "voyager-1.6-gemma4-26b-a4b-it")
AUTO_RERUN_CONTEXT_WINDOW = -1
AUTO_RERUN_CONTEXT_CHAR_BUDGET = 12000
AUTO_RERUN_CONTEXT_MIN_MESSAGES = 6
AUTO_RERUN_CONTEXT_MAX_MESSAGES = 24


@dataclass(frozen=True)
class RerunLogprobsSettings:
    model: str = DEFAULT_COMPANY_MODEL
    provider: str = DEFAULT_COMPANY_PROVIDER
    url: str = DEFAULT_COMPANY_URL
    temperature: float = 0.2
    max_completion_tokens: int = 4096
    top_logprobs: int = 5
    request_logprobs: bool = True
    transport: str = "auto"
    timeout_seconds: int = 120
    context_window_turns: int | None = AUTO_RERUN_CONTEXT_WINDOW
    insert_tool_placeholders: bool = True


@dataclass(frozen=True)
class NormalizedChatResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    usage: Any = None
    logprobs: Any = None
    finish_reason: Any = None


@dataclass(frozen=True)
class RerunLogprobsTurn:
    user_turn_index: int
    assistant_turn_index: int
    user_message: str
    old_assistant_response: str
    new_assistant_response: str
    tool_calls: list[dict[str, Any]]
    usage: Any
    logprobs: Any
    request_meta: dict[str, Any]
    response_diagnostics: dict[str, Any]
    transport_used: str
    raw_response: Any = None
    request_payload: dict[str, Any] | None = None
    error: str | None = None

    def to_record(
        self,
        *,
        include_raw_response: bool = True,
        include_request_payload: bool = False,
    ) -> dict[str, Any]:
        record = {
            "user_turn_index": self.user_turn_index,
            "assistant_turn_index": self.assistant_turn_index,
            "user_message": self.user_message,
            "old_assistant_response": self.old_assistant_response,
            "new_assistant_response": self.new_assistant_response,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "logprobs": self.logprobs,
            "request_meta": self.request_meta,
            "response_diagnostics": self.response_diagnostics,
            "transport_used": self.transport_used,
            "error": self.error,
        }
        if include_raw_response:
            record["raw_response"] = self.raw_response
        if include_request_payload and self.request_payload is not None:
            record["request_payload"] = self.request_payload
        return _jsonable(record)


@dataclass(frozen=True)
class RerunLogprobsResult:
    updated_data: ConversationData
    turns: list[RerunLogprobsTurn]

    def diagnostics_payload(
        self,
        *,
        source_path: str | None = None,
        output_path: str | None = None,
        settings: RerunLogprobsSettings | None = None,
        include_raw_response: bool = True,
        include_request_payload: bool = False,
    ) -> dict[str, Any]:
        turn_records = [
            turn.to_record(
                include_raw_response=include_raw_response,
                include_request_payload=include_request_payload,
            )
            for turn in self.turns
        ]
        return {
            "created_at": utc_timestamp(),
            "source_path": source_path,
            "output_path": output_path,
            "target_turns": [turn.assistant_turn_index for turn in self.turns],
            "settings": _settings_record(settings) if settings else None,
            "aggregate": {
                "result_count": len(self.turns),
                "error_count": len([turn for turn in self.turns if turn.error]),
                "logprobs_available_count": len(
                    [
                        turn
                        for turn in self.turns
                        if turn.response_diagnostics.get("logprobs", {}).get("available") is True
                    ]
                ),
            },
            "results": turn_records,
        }


BackendCaller = Callable[[dict[str, Any], RerunLogprobsSettings], tuple[Any, str]]


def rerun_target_turns_with_logprobs(
    *,
    data: ConversationData,
    target_assistant_turn_indices: set[int],
    optimized_prompt: str | None = None,
    settings: RerunLogprobsSettings | None = None,
    required_tools_by_turn: dict[int, str] | None = None,
    backend_caller: BackendCaller | None = None,
) -> RerunLogprobsResult:
    if not target_assistant_turn_indices:
        raise ValueError("At least one target assistant turn index is required.")
    settings = settings or RerunLogprobsSettings()
    prompt = optimized_prompt if optimized_prompt is not None else data.system_prompt
    required_tools_by_turn = required_tools_by_turn or {}
    backend_caller = backend_caller or send_chat_completion

    results: list[RerunLogprobsTurn] = []
    replacements: dict[int, tuple[str, list[dict[str, Any]]]] = {}
    replay_messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
    pending_user: tuple[int, str] | None = None
    latest_user: tuple[int, str] | None = None

    for index, turn in enumerate(data.interactions):
        if turn.role == "user":
            pending_user = (index, turn.content)
            latest_user = pending_user
            replay_messages.append(interaction_replay_message(turn))
            continue

        if turn.role == "assistant" and (
            pending_user is not None or index in target_assistant_turn_indices
        ):
            user_index, user_message = pending_user or latest_user or (-1, "")
            if index not in target_assistant_turn_indices:
                replay_messages.append(interaction_replay_message(turn))
                pending_user = None
                continue

            request_messages = rerun_request_messages(
                replay_messages,
                settings.context_window_turns,
            )
            required_tool = required_tools_by_turn.get(index)
            request_messages = with_required_tool_instruction(request_messages, required_tool)
            tool_choice = tool_choice_for_required_tool(required_tool, data.tools)
            payload = build_chat_completion_payload(
                messages=request_messages,
                tools=data.tools,
                tool_choice=tool_choice,
                settings=settings,
            )
            request_meta = build_request_meta(
                payload=payload,
                target_assistant_turn_index=index,
                user_turn_index=user_index,
                tools=data.tools,
                settings=settings,
            )
            try:
                raw_response, transport_used = backend_caller(payload, settings)
                normalized = normalize_chat_response(raw_response)
                content = normalized.content
                tool_calls = normalized.tool_calls or function_call_wrappers_to_tool_calls(
                    content,
                    id_prefix=f"call_rerun_logprobs_{index}",
                )
                if tool_calls and not content.strip():
                    content = render_tool_calls_as_function_wrappers(tool_calls) or ""
                replay_messages.append({"role": "assistant", "content": content})
                replacements[index] = (content, tool_calls)
                diagnostics = response_logprob_diagnostics(
                    normalized.logprobs,
                    requested=settings.request_logprobs,
                    top_logprobs_requested=settings.top_logprobs if settings.request_logprobs else None,
                )
                results.append(
                    RerunLogprobsTurn(
                        user_turn_index=user_index,
                        assistant_turn_index=index,
                        user_message=user_message,
                        old_assistant_response=turn.content,
                        new_assistant_response=content,
                        tool_calls=tool_calls,
                        usage=_jsonable(normalized.usage),
                        logprobs=_jsonable(normalized.logprobs),
                        request_meta=request_meta,
                        response_diagnostics=diagnostics,
                        transport_used=transport_used,
                        raw_response=_jsonable(raw_response),
                        request_payload=_jsonable(payload),
                    )
                )
            except (requests.RequestException, RuntimeError, ValueError, ImportError, ModuleNotFoundError, Exception) as exc:
                replay_messages.append(interaction_replay_message(turn))
                results.append(
                    RerunLogprobsTurn(
                        user_turn_index=user_index,
                        assistant_turn_index=index,
                        user_message=user_message,
                        old_assistant_response=turn.content,
                        new_assistant_response="",
                        tool_calls=[],
                        usage=None,
                        logprobs=None,
                        request_meta=request_meta,
                        response_diagnostics=response_logprob_diagnostics(
                            None,
                            requested=settings.request_logprobs,
                            top_logprobs_requested=settings.top_logprobs if settings.request_logprobs else None,
                            error=str(exc),
                        ),
                        transport_used=settings.transport,
                        raw_response=None,
                        request_payload=_jsonable(payload),
                        error=str(exc),
                    )
                )
            pending_user = None
            continue

        replay_messages.append(interaction_replay_message(turn))

    return RerunLogprobsResult(
        updated_data=build_updated_conversation(
            data=data,
            optimized_prompt=prompt,
            replacements=replacements,
            insert_tool_placeholders=settings.insert_tool_placeholders,
        ),
        turns=results,
    )


def build_chat_completion_payload(
    *,
    messages: list[dict[str, Any]],
    tools: dict[str, Any] | None,
    tool_choice: dict[str, Any] | str | None,
    settings: RerunLogprobsSettings,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": copy_messages(messages),
        "model": settings.model,
        "provider": settings.provider,
        "temperature": settings.temperature,
        "top_k": -1,
        "top_p": 0.95,
        "min_p": 0.0,
        "top_n_sigma": 0.0,
        "max_completion_tokens": settings.max_completion_tokens,
        "tools": list(tools.values()) if tools else None,
    }
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if settings.request_logprobs:
        payload["logprobs"] = True
        if settings.top_logprobs > 0:
            payload["top_logprobs"] = min(max(settings.top_logprobs, 0), 20)
    return {key: value for key, value in payload.items() if value is not None}


def copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_jsonable(message) for message in messages if isinstance(message, dict)]


def send_chat_completion(
    payload: dict[str, Any],
    settings: RerunLogprobsSettings,
) -> tuple[Any, str]:
    if settings.transport not in {"auto", "llmparty", "direct"}:
        raise ValueError("transport must be one of: auto, llmparty, direct")
    if settings.transport in {"auto", "llmparty"}:
        try:
            return call_with_llmparty_api_client(payload, settings), "llmparty"
        except (ImportError, ModuleNotFoundError):
            if settings.transport == "llmparty":
                raise
        except Exception:
            if settings.transport == "llmparty":
                raise
            raise
    return call_direct_chat_completions(payload, settings), "direct"


def call_with_llmparty_api_client(
    payload: dict[str, Any],
    settings: RerunLogprobsSettings,
) -> Any:
    add_local_llmparty_to_path()
    from llmparty.demo.chat.api_client import APIClient

    client = APIClient(
        model_name=settings.model,
        provider=settings.provider,
        url=settings.url,
        temperature=settings.temperature,
        max_completion_tokens=settings.max_completion_tokens,
        top_k=int(payload.get("top_k", -1)),
        top_p=float(payload.get("top_p", 0.95)),
        min_p=float(payload.get("min_p", 0.0)),
        top_n_sigma=float(payload.get("top_n_sigma", 0.0)),
        logprobs=bool(payload.get("logprobs", False)),
        top_logprobs=payload.get("top_logprobs"),
        tool_choice=payload.get("tool_choice"),
    )
    return client.chat_completion(
        messages=payload["messages"],
        tools=payload.get("tools"),
    )


def call_direct_chat_completions(
    payload: dict[str, Any],
    settings: RerunLogprobsSettings,
) -> Any:
    response = requests.post(
        settings.url.rstrip("/") + "/v1/chat/completions",
        json=payload,
        timeout=settings.timeout_seconds,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    data = response.json()
    if isinstance(data, dict) and ("error" in data or "detail" in data):
        raise RuntimeError(str(data))
    return data


def normalize_chat_response(response: Any) -> NormalizedChatResponse:
    response = _first_response(response)
    if isinstance(response, str):
        return NormalizedChatResponse(content=response, tool_calls=[])
    if not isinstance(response, dict):
        raise ValueError(f"Unsupported response shape: {type(response).__name__}")

    if isinstance(response.get("content"), str):
        return NormalizedChatResponse(
            content=response["content"],
            tool_calls=normalize_tool_calls(response.get("tool_calls")),
            usage=response.get("usage"),
            logprobs=response.get("logprobs"),
            finish_reason=response.get("finish_reason"),
        )

    message = response.get("message")
    if isinstance(message, dict):
        return _normalize_message_response(
            message=message,
            usage=response.get("usage"),
            logprobs=response.get("logprobs"),
            finish_reason=response.get("finish_reason"),
        )

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                return _normalize_message_response(
                    message=message,
                    usage=response.get("usage"),
                    logprobs=first.get("logprobs"),
                    finish_reason=first.get("finish_reason"),
                )
            if isinstance(first.get("text"), str):
                return NormalizedChatResponse(
                    content=first["text"],
                    tool_calls=[],
                    usage=response.get("usage"),
                    logprobs=first.get("logprobs"),
                    finish_reason=first.get("finish_reason"),
                )
    raise ValueError(f"Could not extract assistant response from response: {response}")


def response_logprob_diagnostics(
    logprobs: Any,
    *,
    requested: bool,
    top_logprobs_requested: int | None,
    error: str | None = None,
) -> dict[str, Any]:
    rows = logprob_content_rows(logprobs)
    info: dict[str, Any] = {
        "requested": requested,
        "top_logprobs_requested": top_logprobs_requested,
        "available": bool(rows),
    }
    if error:
        info["reason"] = error
        return {"logprobs": info}
    if not rows:
        info["reason"] = (
            "Response did not include token logprobs."
            if requested
            else "Logprobs were not requested."
        )
        return {"logprobs": info}

    scored_rows = [
        row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("logprob"), (int, float))
    ]
    values = [float(row["logprob"]) for row in scored_rows]
    info.update(
        {
            "content_token_count": len(rows),
            "scored_token_count": len(values),
            "avg_logprob": sum(values) / len(values) if values else None,
            "min_logprob": min(values) if values else None,
            "low_confidence_tokens": low_confidence_token_samples(scored_rows),
            "sample_content_logprobs": _jsonable(rows[:80]),
        }
    )
    return {"logprobs": info}


def logprob_content_rows(logprobs: Any) -> list[Any]:
    if isinstance(logprobs, dict):
        content = logprobs.get("content")
        return content if isinstance(content, list) else []
    if isinstance(logprobs, list):
        return logprobs
    return []


def low_confidence_token_samples(token_rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
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
            sample["top_logprobs"] = _jsonable(top[:5])
        samples.append(sample)
    return samples


def build_updated_conversation(
    *,
    data: ConversationData,
    optimized_prompt: str,
    replacements: dict[int, tuple[str, list[dict[str, Any]]]],
    insert_tool_placeholders: bool,
) -> ConversationData:
    rebuilt: list[Interaction] = []
    for index, turn in enumerate(data.interactions):
        replacement = replacements.get(index)
        if replacement is None:
            rebuilt.append(turn)
            continue
        content, tool_calls = replacement
        rebuilt.append(
            turn.model_copy(
                update={
                    "content": content,
                    "tool_calls": tool_calls or None,
                    "tool_call_id": None,
                }
            )
        )
        if insert_tool_placeholders:
            for call in tool_calls:
                rebuilt.append(tool_placeholder_interaction(call))
    return data.model_copy(
        update={
            "system_prompt": optimized_prompt,
            "interactions": rebuilt,
        }
    )


def conversation_payload(data: ConversationData) -> dict[str, Any]:
    return {
        "system_prompt": data.system_prompt,
        "interactions": [turn.model_dump(exclude_none=True) for turn in data.interactions],
        "tools": data.tools,
        "source_meta": data.source_meta,
    }


def parse_turn_specs(values: list[str]) -> set[int]:
    turns: set[int] = set()
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                if end < start:
                    raise ValueError(f"Invalid descending turn range: {part}")
                turns.update(range(start, end + 1))
            else:
                turns.add(int(part))
    return turns


def interaction_replay_message(turn: Interaction) -> dict[str, Any]:
    message = turn.model_dump(exclude_none=True)
    if turn.role == "assistant" and turn.tool_calls and "<function-call" not in turn.content.lower():
        wrappers = render_tool_calls_as_function_wrappers(turn.tool_calls)
        if wrappers:
            message["content"] = "\n".join(
                piece for piece in (turn.content, wrappers) if piece.strip()
            )
    return message


def rerun_request_messages(
    replay_messages: list[dict[str, Any]],
    context_window_turns: int | None,
) -> list[dict[str, Any]]:
    if context_window_turns == AUTO_RERUN_CONTEXT_WINDOW:
        return auto_rerun_request_messages(replay_messages)
    if context_window_turns is None or context_window_turns <= 0 or len(replay_messages) <= 1:
        return replay_messages
    system_message = replay_messages[0]
    recent_messages = replay_messages[1:][-context_window_turns:]
    return [system_message, *recent_messages]


def auto_rerun_request_messages(replay_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(replay_messages) <= 1:
        return replay_messages
    system_message = replay_messages[0]
    context_messages = replay_messages[1:]
    if len(context_messages) <= AUTO_RERUN_CONTEXT_MIN_MESSAGES:
        return replay_messages

    selected_reversed: list[dict[str, Any]] = []
    selected_chars = 0
    for message in reversed(context_messages):
        message_chars = len(str(message.get("content", ""))) + len(str(message.get("role", ""))) + 16
        must_keep = len(selected_reversed) < AUTO_RERUN_CONTEXT_MIN_MESSAGES
        within_budget = selected_chars + message_chars <= AUTO_RERUN_CONTEXT_CHAR_BUDGET
        if not must_keep and not within_budget:
            break
        selected_reversed.append(message)
        selected_chars += message_chars
        if len(selected_reversed) >= AUTO_RERUN_CONTEXT_MAX_MESSAGES:
            break

    return [system_message, *reversed(selected_reversed)]


def with_required_tool_instruction(
    messages: list[dict[str, Any]],
    required_tool: str | None,
) -> list[dict[str, Any]]:
    if not required_tool:
        return messages
    instruction = (
        "For the next assistant turn, you MUST call the function/tool "
        f"`{required_tool}`. Do not answer in natural language before this tool call. "
        "Use the latest customer message and relevant immediate context to populate the arguments."
    )
    return [*messages, {"role": "system", "content": instruction}]


def tool_choice_for_required_tool(
    required_tool: str | None,
    tools: dict[str, Any] | None,
) -> dict[str, Any] | str | None:
    if not required_tool or not tools or required_tool not in tools:
        return None
    return {
        "type": "function",
        "function": {
            "name": required_tool,
        },
    }


def build_request_meta(
    *,
    payload: dict[str, Any],
    target_assistant_turn_index: int,
    user_turn_index: int,
    tools: dict[str, Any] | None,
    settings: RerunLogprobsSettings,
) -> dict[str, Any]:
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    system_prompt = first_system_prompt(messages)
    return {
        "timestamp": utc_timestamp(),
        "target_assistant_turn_index": target_assistant_turn_index,
        "user_turn_index": user_turn_index,
        "endpoint": settings.url.rstrip("/") + "/v1/chat/completions",
        "provider": settings.provider,
        "model": settings.model,
        "transport": settings.transport,
        "temperature": settings.temperature,
        "max_completion_tokens": settings.max_completion_tokens,
        "request_logprobs": bool(payload.get("logprobs")),
        "top_logprobs": payload.get("top_logprobs"),
        "message_count": len(messages),
        "message_roles": [str(message.get("role", "")) for message in messages if isinstance(message, dict)],
        "system_prompt_hash": prompt_hash(system_prompt),
        "system_prompt_len": len(system_prompt),
        "messages_hash": prompt_hash(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
        "tools_count": len(tools or {}),
        "tool_choice": payload.get("tool_choice"),
        "payload_keys": sorted(payload.keys()),
    }


def first_system_prompt(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "system":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def add_local_llmparty_to_path() -> None:
    candidates = [
        PROJECT_ROOT / "llmparty" / "llmparty",
        PROJECT_ROOT.parent / "llmparty",
    ]
    for candidate in candidates:
        if (candidate / "llmparty" / "demo" / "chat" / "api_client.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


def tool_placeholder_interaction(call: dict[str, Any]) -> Interaction:
    function = call.get("function") if isinstance(call, dict) else None
    tool_name = function.get("name") if isinstance(function, dict) else None
    payload = {
        "status": "not_executed",
        "tool_name": tool_name,
        "note": "Placeholder inserted by rerun_with_llmparty_logprobs; real tools are not executed locally.",
    }
    return Interaction(
        role="tool",
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=str(call.get("id") or "") if isinstance(call, dict) else "",
    )


def normalize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(raw_tool_calls, start=1):
        call_data = _jsonable(call)
        if not isinstance(call_data, dict):
            continue
        function = call_data.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = "" if arguments is None else json.dumps(arguments, ensure_ascii=False)
        normalized.append(
            {
                "id": str(call_data.get("id") or f"call_rerun_logprobs_{index}"),
                "type": str(call_data.get("type") or "function"),
                "function": {
                    "name": name.strip(),
                    "arguments": arguments,
                },
            }
        )
    return normalized


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_message_response(
    *,
    message: dict[str, Any],
    usage: Any,
    logprobs: Any,
    finish_reason: Any,
) -> NormalizedChatResponse:
    content = message.get("content")
    tool_calls = normalize_tool_calls(message.get("tool_calls"))
    if not isinstance(content, str):
        content = render_tool_calls_as_function_wrappers(tool_calls) or ""
    return NormalizedChatResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        logprobs=logprobs,
        finish_reason=finish_reason,
    )


def _first_response(response: Any) -> Any:
    if isinstance(response, list) and response:
        return response[0]
    return response


def _settings_record(settings: RerunLogprobsSettings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "provider": settings.provider,
        "url": settings.url,
        "temperature": settings.temperature,
        "max_completion_tokens": settings.max_completion_tokens,
        "top_logprobs": settings.top_logprobs,
        "request_logprobs": settings.request_logprobs,
        "transport": settings.transport,
        "timeout_seconds": settings.timeout_seconds,
        "context_window_turns": settings.context_window_turns,
        "insert_tool_placeholders": settings.insert_tool_placeholders,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            pass
    return repr(value)
