from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class Interaction(BaseModel):
    role: str = Field(..., description="One of system, user, assistant, or tool.")
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        value = value.strip().lower()
        allowed_roles = {"system", "user", "assistant", "tool"}
        if value not in allowed_roles:
            raise ValueError(f"role must be one of {sorted(allowed_roles)}")
        return value


class ConversationData(BaseModel):
    system_prompt: str
    interactions: list[Interaction]
    tools: dict[str, Any] | None = None
    source_meta: dict[str, Any] | None = None

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("system_prompt cannot be empty")
        return value

    @field_validator("interactions")
    @classmethod
    def validate_interactions(cls, value: list[Interaction]) -> list[Interaction]:
        if not value:
            raise ValueError("interactions must contain at least one turn")
        return value


@dataclass(frozen=True)
class ParseResult:
    data: ConversationData | None
    raw_data: dict[str, Any] | None
    repaired_text: str | None
    warnings: list[str]
    error: str | None = None


def parse_conversation_json(raw_text: str) -> ParseResult:
    """Parse strict JSON first, then try conservative repairs for common upload mistakes."""
    if not raw_text or not raw_text.strip():
        return ParseResult(None, None, None, [], "The uploaded file is empty.")

    warnings: list[str] = []
    try:
        raw_data = json.loads(raw_text)
        return _validate_raw_data(raw_data, None, warnings)
    except json.JSONDecodeError as initial_error:
        repaired_text, repair_warnings = repair_common_json_issues(raw_text)
        warnings.extend(repair_warnings)
        try:
            raw_data = json.loads(repaired_text)
            warnings.append("The JSON had syntax issues and was automatically repaired.")
            return _validate_raw_data(raw_data, repaired_text, warnings)
        except json.JSONDecodeError:
            pass

        try:
            raw_data = ast.literal_eval(repaired_text)
            warnings.append("The JSON looked like a Python literal and was converted.")
            repaired_text = json.dumps(raw_data, indent=2, ensure_ascii=False)
            return _validate_raw_data(raw_data, repaired_text, warnings)
        except (ValueError, SyntaxError):
            return ParseResult(
                None,
                None,
                repaired_text,
                warnings,
                _friendly_json_error(initial_error),
            )


def repair_common_json_issues(raw_text: str) -> tuple[str, list[str]]:
    """Apply safe-ish fixes: comments, smart quotes, trailing commas, single quotes, and missing commas."""
    text = raw_text.strip().lstrip("\ufeff")
    warnings: list[str] = []

    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for source, target in replacements.items():
        if source in text:
            text = text.replace(source, target)
            warnings.append("Smart quotes were normalized.")
            break

    without_comments = _strip_json_comments(text)
    if without_comments != text:
        text = without_comments
        warnings.append("JavaScript-style comments were removed.")

    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", text)
    if no_trailing_commas != text:
        text = no_trailing_commas
        warnings.append("Trailing commas were removed.")

    quoted_keys = re.sub(r"(?m)([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:", r'\1"\2":', text)
    if quoted_keys != text:
        text = quoted_keys
        warnings.append("Unquoted object keys were quoted.")

    inserted_commas = re.sub(r'([}\]"])\s*\n\s*("?[A-Za-z_][A-Za-z0-9_-]*"?\s*:)', r"\1,\n\2", text)
    inserted_commas = re.sub(r'(")\s*\n\s*([{\[])', r"\1,\n\2", inserted_commas)
    if inserted_commas != text:
        text = inserted_commas
        warnings.append("Some likely missing commas were inserted.")

    if "'" in text and '"' not in text[: max(1, len(text) // 8)]:
        warnings.append("Single quotes may need to be changed to double quotes.")

    return text, warnings


def conversation_to_openai_messages(data: ConversationData) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": data.system_prompt}]
    messages.extend(turn.model_dump(exclude_none=True) for turn in data.interactions)
    return messages


def _validate_raw_data(
    raw_data: Any, repaired_text: str | None, warnings: list[str]
) -> ParseResult:
    raw_data = _normalize_conversation_shape(raw_data, warnings)
    if not isinstance(raw_data, dict):
        return ParseResult(
            None,
            None,
            repaired_text,
            warnings,
            "Top-level JSON must be an object, or a list of role/content messages.",
        )
    try:
        data = ConversationData.model_validate(raw_data)
    except ValidationError as exc:
        return ParseResult(None, raw_data, repaired_text, warnings, _friendly_validation_error(exc, raw_data))
    return ParseResult(data, raw_data, repaired_text, warnings)


def _normalize_conversation_shape(raw_data: Any, warnings: list[str]) -> Any:
    """Accept common chat-export shapes and convert them to the app's standard schema."""
    if isinstance(raw_data, list):
        if raw_data and all(isinstance(item, dict) and _message_content(item) for item in raw_data):
            warnings.append("Detected a top-level message list and converted it to the standard schema.")
        return _messages_to_standard(
            raw_data,
            default_system_prompt="You are a helpful assistant.",
            source_meta={"format": "top_level_message_list"},
        )
        if raw_data and isinstance(raw_data[0], dict):
            warnings.append("Detected a list of records. Loaded the first record.")
            return _normalize_conversation_shape(raw_data[0], warnings)
        return raw_data

    if not isinstance(raw_data, dict):
        return raw_data

    if "system_prompt" in raw_data and "interactions" in raw_data:
        return _normalize_standard_tools(raw_data)

    system_prompt = _first_string(raw_data, ("system_prompt", "system", "prompt", "instruction", "instructions"))
    message_container = _first_present(
        raw_data,
        (
            "interactions",
            "messages",
            "dialog",
            "conversation",
            "conversations",
            "chat",
            "turns",
        ),
    )

    if isinstance(message_container, dict):
        nested = _normalize_conversation_shape(message_container, warnings)
        if isinstance(nested, dict) and "interactions" in nested:
            if system_prompt and not nested.get("system_prompt"):
                nested["system_prompt"] = system_prompt
            warnings.append("Detected a nested conversation object and converted it to the standard schema.")
            return nested

    if isinstance(message_container, list):
        warnings.append("Detected a non-standard conversation key and converted it to the standard schema.")
        return _messages_to_standard(
            message_container,
            default_system_prompt=system_prompt,
            tools=_normalize_tools(raw_data.get("tools")),
            source_meta=_extract_source_meta(raw_data),
        )

    for value in raw_data.values():
        if isinstance(value, dict):
            nested = _normalize_conversation_shape(value, warnings)
            if isinstance(nested, dict) and "system_prompt" in nested and "interactions" in nested:
                warnings.append("Detected a nested standard conversation object.")
                return nested

    return raw_data


def _messages_to_standard(
    messages: list[Any],
    default_system_prompt: str | None,
    tools: dict[str, Any] | None = None,
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system_prompt = default_system_prompt or "You are a helpful assistant."
    interactions: list[dict[str, str]] = []

    for item in messages:
        if not isinstance(item, dict):
            continue
        role = _message_role(item)
        content = _message_content(item)
        if not role or not content:
            continue
        if role == "system" and not interactions:
            system_prompt = content
            continue
        if role in {"user", "assistant", "tool"}:
            interaction = {"role": role, "content": content}
            tool_calls = _normalized_tool_calls(item.get("tool_calls"))
            if tool_calls:
                interaction["tool_calls"] = tool_calls
            tool_call_id = item.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                interaction["tool_call_id"] = tool_call_id.strip()
            interactions.append(interaction)

    return {
        "system_prompt": system_prompt,
        "interactions": interactions,
        "tools": tools,
        "source_meta": source_meta,
    }


def _normalize_standard_tools(raw_data: dict[str, Any]) -> dict[str, Any]:
    if "tools" not in raw_data:
        return raw_data
    normalized_tools = _normalize_tools(raw_data.get("tools"))
    if normalized_tools is raw_data.get("tools"):
        return raw_data
    return {**raw_data, "tools": normalized_tools}


def _normalize_tools(raw_tools: Any) -> dict[str, Any] | None:
    if raw_tools is None:
        return None
    if isinstance(raw_tools, list):
        normalized: dict[str, Any] = {}
        for index, item in enumerate(raw_tools):
            if not isinstance(item, dict):
                continue
            name = _tool_definition_name(item) or f"tool_{index + 1}"
            normalized[name] = item
        return normalized or None
    if isinstance(raw_tools, dict):
        single_name = _tool_definition_name(raw_tools)
        if single_name:
            return {single_name: raw_tools}
        normalized = {}
        for key, value in raw_tools.items():
            if not isinstance(value, dict):
                continue
            name = _tool_definition_name(value) or str(key)
            if name.strip():
                normalized[name] = value
        return normalized or None
    return None


def _tool_definition_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"].strip() or None
    if isinstance(tool.get("name"), str):
        return str(tool["name"]).strip() or None
    return None


def _extract_source_meta(raw_data: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if isinstance(raw_data.get("meta"), dict):
        meta["meta"] = raw_data["meta"]
    if isinstance(raw_data.get("type"), str):
        meta["type"] = raw_data["type"]
    model_info = _find_first_model_info(raw_data)
    if model_info:
        meta["model_info"] = model_info
    return meta


def _find_first_model_info(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if isinstance(value.get("meta"), dict) and isinstance(value["meta"].get("kwargs"), dict):
            kwargs = value["meta"]["kwargs"]
            return {
                "provider": _first_model_provider(value["meta"], kwargs),
                "model": _first_model_name(value["meta"], kwargs),
                "url": kwargs.get("url"),
                "kwargs": kwargs,
            }
        for child in value.values():
            found = _find_first_model_info(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_first_model_info(child)
            if found:
                return found
    return None


def _first_model_provider(meta: dict[str, Any], kwargs: dict[str, Any]) -> str | None:
    model_info = meta.get("model_info")
    if isinstance(model_info, dict) and isinstance(model_info.get("provider"), str):
        return model_info["provider"]
    models = kwargs.get("models")
    if isinstance(models, list) and models and isinstance(models[0], str) and ":" in models[0]:
        return models[0].split(":", 1)[0]
    return None


def _first_model_name(meta: dict[str, Any], kwargs: dict[str, Any]) -> str | None:
    model_info = meta.get("model_info")
    if isinstance(model_info, dict) and isinstance(model_info.get("model"), str):
        return model_info["model"]
    models = kwargs.get("models")
    if isinstance(models, list) and models and isinstance(models[0], str):
        return models[0].split(":", 1)[-1]
    return None


def _message_role(item: dict[str, Any]) -> str | None:
    role = _first_string(item, ("role", "from", "speaker", "author", "sender"))
    if not role:
        return None
    normalized = role.strip().lower()
    role_map = {
        "human": "user",
        "customer": "user",
        "client": "user",
        "bot": "assistant",
        "ai": "assistant",
        "gpt": "assistant",
        "model": "assistant",
    }
    return role_map.get(normalized, normalized)


def _message_content(item: dict[str, Any]) -> str | None:
    value = _first_present(item, ("content", "text", "value", "message", "utterance"))
    tool_call_content = _message_tool_call_content(item)
    text_content: str | None = None
    if isinstance(value, str):
        text_content = value
    elif isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        text_content = "\n".join(parts) if parts else None
    pieces = [
        piece
        for piece in (text_content, tool_call_content)
        if isinstance(piece, str) and piece.strip()
    ]
    return "\n".join(pieces) if pieces else None


def _message_tool_call_content(item: dict[str, Any]) -> str | None:
    return render_tool_calls_as_function_wrappers(item.get("tool_calls"))


def render_tool_calls_as_function_wrappers(tool_calls: Any) -> str | None:
    if not isinstance(tool_calls, list):
        return None
    rendered_calls: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict) and not hasattr(call, "function"):
            continue
        function = _object_value(call, "function")
        if function is None:
            continue
        name = _object_value(function, "name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = _object_value(function, "arguments")
        if not isinstance(arguments, str):
            arguments = "" if arguments is None else json.dumps(arguments, ensure_ascii=False)
        rendered_calls.append(f"<function-call>{name.strip()}:{arguments}</function-call>")
    return "\n".join(rendered_calls) if rendered_calls else None


def function_call_wrappers_to_tool_calls(text: str, id_prefix: str = "call") -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not text or "<function-call" not in text.lower():
        return calls
    pattern = re.compile(
        r"<function-call>\s*([^:<>\s]+)\s*:(.*?)</function-call>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for index, match in enumerate(pattern.finditer(text), start=1):
        name = match.group(1).strip()
        arguments = match.group(2).strip()
        if not name:
            continue
        calls.append(
            {
                "id": f"{id_prefix}_{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return calls


def _normalized_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tool_calls, list):
        return None
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls, start=1):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = "" if arguments is None else json.dumps(arguments, ensure_ascii=False)
        normalized_call = {
            "id": str(call.get("id") or f"call_{index}"),
            "type": str(call.get("type") or "function"),
            "function": {
                "name": name.strip(),
                "arguments": arguments,
            },
        }
        normalized.append(normalized_call)
    return normalized or None


def _object_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _first_string(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _first_present(data, keys)
    return value if isinstance(value, str) and value.strip() else None


def _friendly_json_error(error: json.JSONDecodeError) -> str:
    return (
        f"JSON syntax error at line {error.lineno}, column {error.colno}: {error.msg}. "
        "Check for missing commas, unmatched brackets, unescaped quotes inside text, "
        "and use double quotes for JSON strings."
    )


def _friendly_validation_error(error: ValidationError, raw_data: dict[str, Any]) -> str:
    details = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue["loc"])
        details.append(f"{location}: {issue['msg']}")
    detected_keys = ", ".join(raw_data.keys()) or "none"
    return (
        "The JSON structure is invalid. "
        + "; ".join(details)
        + f". Detected top-level keys: {detected_keys}. "
        "Expected either {system_prompt, interactions} or a chat export with messages/dialog/conversation turns."
    )


def _strip_json_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if char == '"' and not escaped:
            in_string = not in_string
        if not in_string and char == "/" and next_char == "/":
            index = text.find("\n", index)
            if index == -1:
                break
            continue
        if not in_string and char == "/" and next_char == "*":
            end = text.find("*/", index + 2)
            if end == -1:
                break
            index = end + 2
            continue
        result.append(char)
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
        index += 1
    return "".join(result)
