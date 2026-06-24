from __future__ import annotations

import json
import os
import re
import difflib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError
import requests

from prompt_optimizer_agent.company_demo_client import (
    generate_with_company_demo,
    get_last_company_response_diagnostics,
    preflight_company_request,
)
from prompt_optimizer_agent.json_utils import ConversationData, Interaction, render_tool_calls_as_function_wrappers


DEFAULT_MODEL = os.getenv("PROMPT_OPTIMIZER_MODEL", "gpt-4o-mini")
DEFAULT_COMPANY_URL = os.getenv("COMPANY_LLM_URL", "http://192.168.101.15:9898")
DEFAULT_COMPANY_PROVIDER = os.getenv("COMPANY_LLM_PROVIDER", "openai_api_like")
DEFAULT_COMPANY_MODEL = os.getenv("COMPANY_LLM_MODEL", "voyager-1.6-gemma4-26b-a4b-it")
AUTO_RERUN_CONTEXT_WINDOW = -1
AUTO_RERUN_CONTEXT_CHAR_BUDGET = 12000
AUTO_RERUN_CONTEXT_MIN_MESSAGES = 6
AUTO_RERUN_CONTEXT_MAX_MESSAGES = 24
PROMPT_EDIT_MAX_NEW_LINE_CHARS = 420
PROMPT_EDIT_MAX_REPEATED_SHINGLE_DELTA = 2
TRANSLATION_TURN_CHAR_LIMIT = 5000
TRANSLATION_TOTAL_CHAR_LIMIT = 24000
PROMPT_EDIT_STYLE_RULES = (
    "Static prompt-edit style rules: do not turn a numbered Step into one long paragraph. "
    "When adding branch logic, use concise numbered substeps or short separate lines. Keep one "
    "condition/action per line. Do not repeat the same branch or fallback instruction in different "
    "words. Do not duplicate semantically equivalent unsure/uncertain branches. Prefer precise "
    "workflow gates such as 'If user is unsure...' followed by exactly one action."
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONCLUSION_DIALOG_LOG_PATH = Path(
    os.getenv(
        "COMPANY_LLM_CONCLUSION_DIALOG_LOG",
        str(PROJECT_ROOT / "logs" / "company_api_conclusion_dialog.jsonl"),
    )
)
LOSS_START = "<|loss_start|>"
LOSS_END = "<|loss_end|>"


@dataclass(frozen=True)
class LLMSettings:
    backend: str = "openai"
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_COMPANY_PROVIDER
    base_url: str = DEFAULT_COMPANY_URL
    max_completion_tokens: int = 4096
    expected_system_prompt_hash: str | None = None
    expected_prompt_version: str | None = None
    rerun_context_window_turns: int | None = AUTO_RERUN_CONTEXT_WINDOW


@dataclass(frozen=True)
class BadCase:
    turn_index: int
    role: str
    error_type: str
    evidence: str
    recommendation: str
    source: str = "ai_judge"


@dataclass(frozen=True)
class PromptOptimization:
    optimized_prompt: str
    rationale: str
    applied_feedback_summary: str


@dataclass(frozen=True)
class JsonRepairResult:
    repaired_json: str
    method: str
    original_text: str
    repaired_text: str


@dataclass(frozen=True)
class RerunTurn:
    user_turn_index: int
    assistant_turn_index: int | None
    user_message: str
    old_assistant_response: str
    new_assistant_response: str
    error: str | None = None
    response_diagnostics: dict[str, Any] | None = None


def translate_interactions_to_zh(
    interactions: list[Interaction],
    llm_settings: LLMSettings | None = None,
) -> dict[int, str]:
    settings = llm_settings or LLMSettings()
    if not _has_llm_access(settings):
        raise RuntimeError(
            "No reachable LLM backend is configured. For OpenAI set OPENAI_API_KEY, "
            "or choose Company API with url/provider/model."
        )
    items = _translation_payload_items(interactions)
    translator_prompt = (
        "You translate multilingual customer-service conversation turns into Simplified Chinese for QA review. "
        "Return JSON only with key translations, whose value is a list of objects with turn_index and translation. "
        "Translate faithfully and concisely without adding analysis, judging, or fixing the conversation. Preserve "
        "numbers, dates, currencies, percentages, product names, file names, tool/function names, and policy labels. "
        "If a turn is already Chinese, keep the Chinese meaning as-is. For function-call wrappers, explain the tool "
        "call in Chinese while preserving the tool name and arguments. For tool outputs, translate readable content "
        "and keep important field names when helpful. If content_truncated is true, begin that translation with "
        "'（内容过长，仅翻译节选）'."
    )
    payload = {
        "target_language": "Simplified Chinese",
        "items": items,
    }
    content = _chat_json(
        settings=settings,
        messages=[
            {"role": "system", "content": translator_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        purpose="translation",
    )
    return _parse_translation_response(content)


def _translation_payload_items(interactions: list[Interaction]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    remaining = TRANSLATION_TOTAL_CHAR_LIMIT
    for index, turn in enumerate(interactions):
        source_text = _translation_source_text(turn)
        if not source_text:
            continue
        char_limit = min(TRANSLATION_TURN_CHAR_LIMIT, max(0, remaining))
        if char_limit <= 0:
            break
        truncated = len(source_text) > char_limit
        content = source_text[:char_limit].rstrip()
        items.append(
            {
                "turn_index": index,
                "role": turn.role,
                "content": content,
                "content_truncated": truncated,
            }
        )
        remaining -= len(content)
    return items


def _translation_source_text(turn: Interaction) -> str:
    if turn.role.lower() != "tool":
        return turn.content
    parsed = _try_parse_json_object(turn.content)
    if not isinstance(parsed, dict):
        return turn.content
    pieces = []
    for key, value in parsed.items():
        if isinstance(value, str):
            pieces.append(f"{key}:\n{value}")
        else:
            pieces.append(f"{key}:\n{json.dumps(value, ensure_ascii=False)}")
    return "\n\n".join(pieces)


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_translation_response(content: str) -> dict[int, str]:
    parsed = json.loads(content)
    translations = parsed.get("translations", [])
    if not isinstance(translations, list):
        raise ValueError("Translation response must contain translations list.")
    result: dict[int, str] = {}
    for item in translations:
        if not isinstance(item, dict):
            continue
        try:
            turn_index = int(item.get("turn_index"))
        except (TypeError, ValueError):
            continue
        translation = str(item.get("translation") or "").strip()
        if translation:
            result[turn_index] = translation
    return result


def analyze_bad_cases(
    data: ConversationData,
    manual_feedback: dict[int, str] | None = None,
    model: str = DEFAULT_MODEL,
    llm_settings: LLMSettings | None = None,
) -> list[BadCase]:
    settings = llm_settings or LLMSettings(model=model)
    manual_feedback = _clean_manual_feedback(manual_feedback)
    ai_cases: list[BadCase] = []

    if _has_llm_access(settings):
        indexed_interactions = [
            {
                "turn_index": index,
                "role": turn.role,
                "content": turn.content,
            }
            for index, turn in enumerate(data.interactions)
        ]
        payload = {
            "system_prompt": data.system_prompt,
            "interactions": indexed_interactions,
        }
        judge_prompt = (
            "You are a strict system-prompt compliance judge, not a general quality reviewer. "
            "Identify bad cases only when an assistant turn clearly violates an explicit requirement, "
            "workflow step, branch condition, negative instruction, tool/function rule, or fact constraint "
            "in the provided system_prompt. Only flag assistant turns. Do not flag a turn merely because "
            "it could be more polite, more detailed, more persuasive, or stylistically better. "
            "Do not flag optional improvements or weak preferences as bad cases. "
            "For vague, partial, uncertain, refused, or unrelated user replies, flag only if the "
            "system_prompt requires clarification, fallback handling, or staying in the current workflow "
            "step and the assistant instead advances, repeats the same blocked request, or skips the "
            "required fallback. If the prompt does not define what to do for ambiguity, omit the case. "
            "For negotiation or debt-collection flows, explicitly check whether the system_prompt forbids "
            "open-ended payment-term questions such as asking the user when or how much they can pay; if a "
            "user rejects a concrete proposal and the assistant asks the user to supply payment terms instead "
            "of proposing the next concrete fallback, flag that assistant turn. For workflow rules with a "
            "maximum attempt limit, count the relevant prior failed clarification/proposal-collection turns; "
            "if the limit is reached or exceeded and the assistant prompts again instead of moving to the "
            "required closing/escalation state, flag it. "
            "For customer-service flows with mandatory fresh retrieval/search tools, inspect assistant "
            "function-call wrappers and tool responses. If the system_prompt says every promotion, discount, "
            "or banking-service customer message requires a fresh search before answering, flag an assistant "
            "natural-language answer that gives product/promotion facts without the required intervening "
            "search tool call. "
            "Return JSON only with key bad_cases. Each bad case must include turn_index, role, "
            "error_type, evidence, recommendation, turn_content_excerpt, violated_prompt_excerpt, "
            "severity, confidence, and hard_violation. severity must be high, medium, or low. confidence "
            "must be a number from 0 to 1. hard_violation must be true only when the assistant violates "
            "a concrete system-prompt rule. Include only hard_violation=true cases with severity high or "
            "medium and confidence >= 0.65. If unsure, return an empty bad_cases list. evidence must "
            "explain the mismatch between the assistant turn and the violated prompt excerpt. evidence must "
            "include a concise analysis basis, not a transcript dump: cite the key prior rejection or "
            "failed-attempt turn indexes, include at most one short quote of the offending assistant wording, "
            "and, when attempt limits are involved, state the counted attempt total. Use the "
            "explicit turn_index value from the provided interactions; do not count turns yourself. "
            "The indexed interaction must be an assistant turn. If an assistant turn is only a "
            "tool/function-call wrapper, flag that wrapper only when the tool call itself is wrong; "
            "otherwise flag the later natural-language assistant reply that exposes the failure to the user."
        )
        try:
            content = _chat_json(
                settings=settings,
                messages=[
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                purpose="judge",
            )
            ai_cases.extend(_parse_judge_bad_cases(data, content))
        except (OpenAIError, requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            if settings.backend == "company_api" and os.getenv("OPENAI_API_KEY"):
                try:
                    fallback_settings = LLMSettings(
                        backend="openai",
                        model=DEFAULT_MODEL,
                        max_completion_tokens=settings.max_completion_tokens,
                    )
                    content = _chat_json(
                        settings=fallback_settings,
                        messages=[
                            {"role": "system", "content": judge_prompt},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        ],
                        purpose="judge_fallback",
                    )
                    ai_cases.extend(_parse_judge_bad_cases(data, content))
                except (OpenAIError, requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError, TypeError) as fallback_exc:
                    ai_cases.append(
                        BadCase(
                            turn_index=-1,
                            role="system",
                            error_type="judge_api_error",
                            evidence=(
                                f"Company API judge failed: {exc}. "
                                f"OpenAI fallback also failed: {fallback_exc}"
                            ),
                            recommendation="AI judge failed. Review manually marked bad cases or retry after checking API settings.",
                        )
                    )
            else:
                ai_cases.append(
                    BadCase(
                        turn_index=-1,
                        role="system",
                        error_type="judge_api_error",
                        evidence=str(exc),
                        recommendation="AI judge failed. Review manually marked bad cases or retry after checking API settings.",
                    )
                )

    ai_cases = _dedupe_bad_cases([*ai_cases, *_local_prompt_rule_bad_cases(data)])

    manual_cases = [
        BadCase(
            turn_index=turn_index,
            role=data.interactions[turn_index].role if 0 <= turn_index < len(data.interactions) else "unknown",
            error_type="human_flagged",
            evidence=feedback,
            recommendation=_manual_recommendation(data, turn_index, feedback, settings),
            source="human",
        )
        for turn_index, feedback in manual_feedback.items()
    ]

    return _merge_manual_and_auto_bad_cases(manual_cases, ai_cases)


def _local_prompt_rule_bad_cases(data: ConversationData) -> list[BadCase]:
    """Reusable deterministic checks for explicit hard rules that LLM judges often under-count."""
    cases: list[BadCase] = []
    cases.extend(_mechanical_detector_bad_cases(data))
    if data.tools is not None:
        cases.extend(_local_undefined_tool_call_cases(data))
    if _prompt_defines_escalation_protocol(data.system_prompt):
        cases.extend(_local_escalation_protocol_cases(data))
    cases.extend(_local_missing_required_tool_call_cases(data))
    return _dedupe_bad_cases(cases)


def _mechanical_detector_bad_cases(data: ConversationData) -> list[BadCase]:
    """Low-false-positive transcript-shape detectors that do not require prompt semantics."""
    cases: list[BadCase] = []
    for index, turn in enumerate(data.interactions):
        if turn.role.lower() != "assistant":
            continue
        marker = _exposed_thought_marker(turn.content)
        if marker is not None:
            cases.append(
                BadCase(
                    turn_index=index,
                    role="assistant",
                    error_type="thought_exposed",
                    evidence=(
                        "Violated rule: Assistant responses must not expose hidden reasoning or internal "
                        "thought-channel markers to the user.\n\n"
                        "Evidence: "
                        f"Assistant turn {index} contains the internal marker `{marker}` "
                        f"({_quote_turn(turn.content)})."
                    ),
                    recommendation=(
                        "Regenerate or clean this assistant turn so only the final user-visible answer remains; "
                        "remove hidden reasoning markers and their enclosed content."
                    ),
                    source="mechanical_detector",
                )
            )
        if not turn.content.strip():
            cases.append(
                BadCase(
                    turn_index=index,
                    role="assistant",
                    error_type="empty_assistant_response",
                    evidence=(
                        "Violated rule: Assistant turns must contain a user-visible response or a valid tool call.\n\n"
                        f"Evidence: Assistant turn {index} has empty content."
                    ),
                    recommendation=(
                        "Regenerate this assistant turn so it contains the required user-visible response or a "
                        "well-formed tool call."
                    ),
                    source="mechanical_detector",
                )
            )
        malformed_tool_call = _malformed_tool_call_evidence(turn)
        if malformed_tool_call:
            cases.append(
                BadCase(
                    turn_index=index,
                    role="assistant",
                    error_type="malformed_tool_call",
                    evidence=(
                        "Violated rule: Assistant tool calls must be syntactically well-formed and use valid JSON "
                        "arguments.\n\n"
                        f"Evidence: Assistant turn {index} has a malformed tool call: {malformed_tool_call} "
                        f"({_quote_turn(turn.content)})."
                    ),
                    recommendation=(
                        "Regenerate or repair the tool call wrapper so it has a function name, a closing "
                        "</function-call> tag, and valid JSON object arguments."
                    ),
                    source="mechanical_detector",
                )
            )
        internal_marker = _internal_marker_leak(turn.content)
        if internal_marker:
            cases.append(
                BadCase(
                    turn_index=index,
                    role="assistant",
                    error_type="internal_marker_leak",
                    evidence=(
                        "Violated rule: Assistant responses must not expose backend/internal protocol markers.\n\n"
                        f"Evidence: Assistant turn {index} contains internal marker `{internal_marker}` "
                        f"({_quote_turn(turn.content)})."
                    ),
                    recommendation=(
                        "Regenerate or clean this assistant turn so internal protocol markers are removed from "
                        "the user-visible output."
                    ),
                    source="mechanical_detector",
                )
            )
        truncated = _truncated_internal_output_evidence(turn.content)
        if truncated:
            cases.append(
                BadCase(
                    turn_index=index,
                    role="assistant",
                    error_type="truncated_internal_output",
                    evidence=(
                        "Violated rule: Assistant output must not stop inside an internal reasoning or loss span.\n\n"
                        f"Evidence: Assistant turn {index} appears truncated: {truncated} "
                        f"({_quote_turn(turn.content)})."
                    ),
                    recommendation=(
                        "Regenerate this assistant turn from the target point and keep only the complete final "
                        "user-visible response."
                    ),
                    source="mechanical_detector",
                )
            )
    return cases


def _exposed_thought_marker(content: str) -> str | None:
    text = content or ""
    for marker in ("thought", "think"):
        if re.search(rf"<\s*{marker}\s*>.*?<\s*/\s*{marker}\s*>", text, flags=re.IGNORECASE | re.DOTALL):
            return f"<{marker}>"
        if re.search(rf"<\s*{marker}\s*>", text, flags=re.IGNORECASE):
            return f"<{marker}>"
    return None


def _malformed_tool_call_evidence(turn: Interaction) -> str | None:
    text = turn.content or ""
    if "<function-call" in text.lower():
        complete_pattern = re.compile(
            r"<function-call>\s*([^:<>\s]+)\s*:(.*?)</function-call>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        matches = list(complete_pattern.finditer(text))
        opening_count = len(re.findall(r"<function-call\b[^>]*>", text, flags=re.IGNORECASE))
        if opening_count != len(matches):
            return "function-call wrapper is missing a valid name, colon, or closing tag"
        for match in matches:
            arguments = match.group(2).strip()
            if arguments and _invalid_tool_arguments_json(arguments):
                return f"`{match.group(1).strip()}` arguments are not valid JSON"
    for call in turn.tool_calls or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            return "tool_calls entry is missing a function object"
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            return "tool_calls function is missing a name"
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments.strip() and _invalid_tool_arguments_json(arguments):
            return f"`{name.strip()}` tool_calls arguments are not valid JSON"
    return None


def _invalid_tool_arguments_json(arguments: str) -> bool:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return True
    return not isinstance(parsed, dict)


def _internal_marker_leak(content: str) -> str | None:
    text = content or ""
    for marker in (LOSS_START, LOSS_END, "<|im_start|>", "<|im_end|>"):
        if marker in text:
            return marker
    match = re.search(r"<\|\s*(?:system|user|assistant|tool|analysis|commentary|final)\s*\|>", text, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _truncated_internal_output_evidence(content: str) -> str | None:
    text = content or ""
    lower = text.lower()
    if lower.count(LOSS_START.lower()) != lower.count(LOSS_END.lower()):
        return f"unbalanced `{LOSS_START}` / `{LOSS_END}` markers"
    for marker in ("thought", "think"):
        if re.search(rf"<\s*{marker}\s*>", text, flags=re.IGNORECASE) and not re.search(
            rf"<\s*/\s*{marker}\s*>", text, flags=re.IGNORECASE
        ):
            return f"unclosed `<{marker}>` span"
    return None


def _local_undefined_tool_call_cases(data: ConversationData) -> list[BadCase]:
    available_tools = _available_tool_names(data)
    if not available_tools:
        return []
    cases: list[BadCase] = []
    for index, turn in enumerate(data.interactions):
        if turn.role.lower() != "assistant":
            continue
        for tool_name in _function_call_names(turn.content):
            if tool_name.lower() in available_tools:
                continue
            cases.append(
                BadCase(
                    turn_index=index,
                    role="assistant",
                    error_type="undefined_tool_call",
                    evidence=(
                        "Violated rule: Assistant function/tool calls must use a tool that is available "
                        "in the current conversation JSON.\n\n"
                        "Evidence: "
                        f"Assistant turn {index} called `{tool_name}`, but the loaded tool definitions only include "
                        f"{_format_tool_name_list(sorted(available_tools))}."
                    ),
                    recommendation=(
                        "Align the prompt/tool configuration so the assistant only calls tools present in the "
                        "current file's tool definitions, or add the missing tool definition before rerunning."
                    ),
                    source="local_scan",
                )
            )
    return cases


def _available_tool_names(data: ConversationData) -> set[str]:
    names: set[str] = set()
    if not data.tools:
        return names
    for key, value in data.tools.items():
        if isinstance(key, str) and key.strip():
            names.add(key.strip().lower())
        if not isinstance(value, dict):
            continue
        function = value.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"].strip().lower())
        elif isinstance(value.get("name"), str):
            names.add(value["name"].strip().lower())
    return names


def _function_call_names(content: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(
        r"<function-call>\s*([^:<>\s]+)\s*:",
        content or "",
        flags=re.IGNORECASE,
    ):
        name = match.group(1).strip()
        if name and name not in names:
            names.append(name)
    return names


def _format_tool_name_list(names: list[str]) -> str:
    if not names:
        return "no tools"
    if len(names) <= 4:
        return ", ".join(f"`{name}`" for name in names)
    return ", ".join(f"`{name}`" for name in names[:4]) + f", and {len(names) - 4} more"


def _prompt_defines_escalation_protocol(system_prompt: str) -> bool:
    lowered = system_prompt.lower()
    return "escalation" in lowered and ("trigger" in lowered or "6.2" in lowered)


def _local_escalation_protocol_cases(data: ConversationData) -> list[BadCase]:
    cases: list[BadCase] = []
    pending_trigger: tuple[int, str] | None = None
    for index, turn in enumerate(data.interactions):
        role = turn.role.lower()
        if role == "user":
            trigger = _user_escalation_trigger(turn.content)
            if trigger:
                pending_trigger = (index, trigger)
            continue
        if role != "assistant" or pending_trigger is None:
            continue
        user_index, trigger = pending_trigger
        pending_trigger = None
        if _assistant_satisfies_escalation_action(data.system_prompt, turn.content):
            continue
        cases.append(
            BadCase(
                turn_index=index,
                role="assistant",
                error_type="escalation_action_not_followed",
                evidence=(
                    f"Violated rule: {_escalation_rule_excerpt(data.system_prompt)}\n\n"
                    "Evidence: "
                    f"User turn {user_index} triggered escalation ({trigger}) with "
                    f"{_quote_turn(data.interactions[user_index].content)}. "
                    f"Assistant turn {index} did not execute the required escalation action "
                    f"({_quote_turn(turn.content)})."
                ),
                recommendation=(
                    "Revise the escalation branch so the first assistant response after any escalation trigger "
                    "stops the negotiation flow and performs the exact configured escalation action, including "
                    "the required spoken text and transfer/hotline ending when specified."
                ),
                source="local_scan",
            )
        )
    return cases


def _user_escalation_trigger(text: str) -> str | None:
    lowered = _compact_text(text).lower()
    if _is_conversation_marker(lowered):
        return None
    trigger_patterns = (
        (
            "dispute_of_debt_or_paid_claim",
            (
                r"\balready paid\b",
                r"\bpaid already\b",
                r"\bamount is wrong\b",
                r"\byour system is wrong\b",
                r"\bi didn'?t borrow\b",
                r"\bsudah bayar\b",
                r"\btelah bayar\b",
                r"\bjumlah(?:nya)? .*?(?:salah|tidak benar|tak betul)\b",
                r"\bsistem .*?(?:salah|masalah|problem|error)\b",
                r"金额.*?(?:不对|不正确|错)",
                r"系统.*?(?:问题|错)",
                r"(?:已经|己经|早就|上周).*?(?:还|付|付款|缴)",
                r"明明只借",
            ),
        ),
        (
            "technical_or_wallet_issue",
            (
                r"\bapp .*?(?:problem|issue|error|cannot|can'?t)\b",
                r"\bwallet .*?(?:problem|issue|error|cannot|can'?t)\b",
                r"\bdompet .*?(?:masalah|problem|error|tidak bisa|tak boleh)\b",
                r"(?:钱包|app|应用).*?(?:问题|故障|不能|无法)",
            ),
        ),
        (
            "overpayment_issue",
            (
                r"\boverpaid\b",
                r"\boverpayment\b",
                r"\bcharged twice\b",
                r"\bdeducted twice\b",
                r"\bterlebih bayar\b",
                r"\bdipotong dua kali\b",
                r"(?:多付|多还|扣了两次|连续扣了两次|付的钱比.*?多)",
            ),
        ),
        (
            "severe_hardship",
            (
                r"\bhospital\b",
                r"\bhospitali[sz]ation\b",
                r"\baccident\b",
                r"\bbankrupt",
                r"\bdeath\b",
                r"\bpassed away\b",
                r"\brumah sakit\b",
                r"\bdirawat\b",
                r"\bkemalangan\b",
                r"\bkecelakaan\b",
                r"(?:医院|住院|车祸|事故|破产|去世|死亡|重病)",
            ),
        ),
        (
            "explicit_agent_request",
            (
                r"\bhuman agent\b",
                r"\bspeak to (?:a )?(?:human|agent|person)\b",
                r"\btransfer\b",
                r"\bagen\b",
                r"\bcustomer service\b",
                r"(?:人工|真人|客服|转接|转人工)",
            ),
        ),
        (
            "alternative_payment_method",
            (
                r"\balternative payment\b",
                r"\banother way to (?:pay|make payment)\b",
                r"\bpayment method\b",
                r"\bcara lain\b.*?\bbayar\b",
                r"\bmetode pembayaran\b",
                r"(?:其他|别的).*?(?:付款|还款|支付).*?(?:方式|方法)",
            ),
        ),
    )
    for label, patterns in trigger_patterns:
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
            return label
    return None


def _assistant_satisfies_escalation_action(system_prompt: str, content: str) -> bool:
    if _prompt_requires_exact_hotline_escalation(system_prompt):
        return _assistant_outputs_hotline_escalation(content)
    if _prompt_requires_spoken_transfer_escalation(system_prompt):
        return _assistant_outputs_spoken_transfer_escalation(content)
    return _assistant_outputs_any_escalation(content)


def _prompt_requires_exact_hotline_escalation(system_prompt: str) -> bool:
    lowered = system_prompt.lower()
    return "deliver the following message exactly" in lowered and "hotline" in lowered


def _prompt_requires_spoken_transfer_escalation(system_prompt: str) -> bool:
    lowered = system_prompt.lower()
    return (
        ("spoken text" in lowered or "empathetic sentence" in lowered or "empathetic phrase" in lowered)
        and ("transfergroup" in lowered or "human officer" in lowered or "human agent" in lowered)
    )


def _assistant_outputs_hotline_escalation(content: str) -> bool:
    lowered = _compact_text(content).lower()
    if _is_tool_only_assistant(content):
        return False
    return "hotline" in lowered and "<dialog-end>" in lowered


def _assistant_outputs_spoken_transfer_escalation(content: str) -> bool:
    if _is_tool_only_assistant(content):
        return False
    lowered = _compact_text(content).lower()
    has_transfer = any(
        term in lowered
        for term in (
            "human officer",
            "human agent",
            "agent",
            "transfer",
            "agen",
            "pegawai",
            "petugas",
            "人工",
            "客服",
            "转接",
        )
    )
    has_forbidden_negotiation = _assistant_mentions_payment_or_debt_details(content)
    return has_transfer and not has_forbidden_negotiation


def _assistant_outputs_any_escalation(content: str) -> bool:
    lowered = _compact_text(content).lower()
    return (
        "hotline" in lowered
        or "dialog-end" in lowered
        or "transfer_to_human" in lowered
        or "transfergroup" in lowered
        or "group1" in lowered
    )


def _is_tool_only_assistant(content: str) -> bool:
    if not _looks_like_tool_wrapper(content):
        return False
    without_wrappers = re.sub(
        r"<function-call>.*?</function-call>",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    without_wrappers = re.sub(
        r"<tool-call>.*?</tool-call>",
        "",
        without_wrappers,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return not without_wrappers.strip()


def _assistant_mentions_payment_or_debt_details(content: str) -> bool:
    lowered = _compact_text(content).lower()
    return any(
        term in lowered
        for term in (
            "pay",
            "payment",
            "settle",
            "loan",
            "overdue",
            "bayar",
            "pembayaran",
            "pinjaman",
            "tunggakan",
            "欠款",
            "逾期",
            "还款",
            "贷款",
        )
    )


def _escalation_rule_excerpt(system_prompt: str) -> str:
    compacted = _compact_text(system_prompt)
    match = re.search(
        r"(?:6\.1\s+)?Escalation Action[:：]?\s*(.{0,520}?)(?:6\.2\s+Escalation Triggers|# PART 2|$)",
        compacted,
        flags=re.IGNORECASE,
    )
    if match:
        return _quote_turn(match.group(1), max_chars=260)
    return "When an escalation trigger occurs, immediately stop the core flow and execute the configured escalation action."


def _assistant_asks_question(text: str) -> bool:
    lowered = _compact_text(text).lower()
    return "?" in lowered or any(
        term in lowered
        for term in (
            "can you",
            "could you",
            "are you",
            "boleh",
            "apakah",
            "adakah",
            "bisa",
            "请问",
            "可以吗",
            "吗",
        )
    )


def _local_missing_required_tool_call_cases(data: ConversationData) -> list[BadCase]:
    rule = _required_tool_rule_from_prompt(data)
    if rule is None:
        return []
    cases: list[BadCase] = []
    pending_user_index: int | None = None
    pending_started_before_name = False
    tool_seen_for_pending = False
    inquiry_context_active = False
    name_gate_required = "customer name gate" in data.system_prompt.lower()
    name_known = not name_gate_required
    name_gate_closed_index: int | None = None
    previous_assistant_asked_name = False

    for index, turn in enumerate(data.interactions):
        role = turn.role.lower()
        if role == "user":
            if previous_assistant_asked_name and not _is_conversation_marker(turn.content):
                name_known = True
                name_gate_closed_index = index
                previous_assistant_asked_name = False
                if pending_user_index is not None:
                    continue
            previous_assistant_asked_name = False
            if _is_required_tool_user_message(turn.content, inquiry_context_active, rule):
                pending_user_index = index
                pending_started_before_name = not name_known
                tool_seen_for_pending = False
                inquiry_context_active = True
            continue

        if role == "assistant":
            if _is_required_tool_call(turn.content, rule.tool_name):
                if pending_user_index is not None:
                    tool_seen_for_pending = True
                inquiry_context_active = True
                previous_assistant_asked_name = False
                continue

            if _looks_like_tool_wrapper(turn.content):
                previous_assistant_asked_name = False
                continue

            if _assistant_asks_for_customer_name(turn.content):
                previous_assistant_asked_name = True
                continue

            previous_assistant_asked_name = False
            if pending_user_index is None:
                continue
            if not name_known:
                continue
            if not _assistant_answers_required_tool_domain(turn.content, rule):
                continue
            if tool_seen_for_pending:
                pending_user_index = None
                tool_seen_for_pending = False
                continue
            cases.append(
                _missing_required_tool_call_bad_case(
                    data=data,
                    rule=rule,
                    assistant_index=index,
                    user_index=pending_user_index,
                    name_gate_closed_index=name_gate_closed_index if pending_started_before_name else None,
                )
            )
            pending_user_index = None
            tool_seen_for_pending = False
            inquiry_context_active = True
            continue

        if role == "tool":
            previous_assistant_asked_name = False

    return cases


@dataclass(frozen=True)
class _RequiredToolRule:
    tool_name: str
    domain_terms: tuple[str, ...]
    rule_excerpt: str


def _required_tool_rule_from_prompt(data: ConversationData) -> _RequiredToolRule | None:
    tool_names = _available_tool_names(data)
    if not tool_names:
        return None
    prompt = _compact_text(data.system_prompt)
    lowered = prompt.lower()
    if not any(term in lowered for term in ("must call", "requires", "required", "before answering", "fresh")):
        return None
    if not any(term in lowered for term in ("every customer message", "every message", "before answering")):
        return None
    for tool_name in sorted(tool_names, key=len, reverse=True):
        if tool_name.lower() not in lowered:
            continue
        domain_terms = _domain_terms_near_tool_rule(prompt, tool_name)
        if not domain_terms:
            continue
        return _RequiredToolRule(
            tool_name=tool_name,
            domain_terms=domain_terms,
            rule_excerpt=_required_tool_rule_excerpt(prompt, tool_name),
        )
    return None


def _domain_terms_near_tool_rule(prompt: str, tool_name: str) -> tuple[str, ...]:
    lowered = prompt.lower()
    tool_index = lowered.find(tool_name.lower())
    window_start = max(0, tool_index - 360) if tool_index >= 0 else 0
    window_end = min(len(prompt), (tool_index if tool_index >= 0 else 0) + len(tool_name) + 360)
    window = prompt[window_start:window_end].lower()
    candidates = set(re.findall(r"\b[a-z][a-z0-9_-]{3,}\b", window))
    generic = {
        "must",
        "call",
        "tool",
        "function",
        "before",
        "after",
        "answer",
        "answering",
        "every",
        "message",
        "customer",
        "requires",
        "required",
        "fresh",
        "search",
        "inquiry",
        "inquiries",
        "handling",
        "request",
        "stated",
        "earlier",
        "immediately",
        "processing",
        "name",
        "gate",
    }
    terms = {term for term in candidates if term not in generic and term not in tool_name.lower()}
    tool_tokens = set(re.findall(r"[a-z0-9]+", tool_name.lower()))
    terms.update(token for token in tool_tokens if token not in generic and len(token) > 3)
    return tuple(sorted(terms))


def _required_tool_rule_excerpt(prompt: str, tool_name: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", prompt)
    matching = [sentence.strip() for sentence in sentences if tool_name.lower() in sentence.lower()]
    if matching:
        return _quote_turn(" ".join(matching[:2]), max_chars=260)
    return _quote_turn(prompt, max_chars=260)


def _missing_required_tool_call_bad_case(
    data: ConversationData,
    rule: _RequiredToolRule,
    assistant_index: int,
    user_index: int,
    name_gate_closed_index: int | None,
) -> BadCase:
    name_gate_note = (
        f" after the name gate closed at user turn {name_gate_closed_index}"
        if name_gate_closed_index is not None
        else ""
    )
    evidence = (
        f"Violated rule: {rule.rule_excerpt}\n\n"
        "Evidence: "
        f"User turn {user_index} matched the required-tool domain{name_gate_note}. "
        f"No `{rule.tool_name}` tool call appeared before assistant turn {assistant_index}, "
        f"which answered with domain-specific details ({_domain_answer_excerpt(data.interactions[assistant_index].content, rule)})."
    )
    return BadCase(
        turn_index=assistant_index,
        role="assistant",
        error_type="missing_required_tool_call",
        evidence=evidence,
        recommendation=(
            "Revise the workflow so every matching customer message triggers the configured required tool "
            "before any factual domain answer, including follow-up confirmations and corrections."
        ),
        source="local_scan",
    )


def _is_required_tool_call(content: str, tool_name: str) -> bool:
    pattern = rf"<function-call>\s*{re.escape(tool_name)}\s*:"
    return bool(re.search(pattern, content, flags=re.IGNORECASE))


def _assistant_asks_for_customer_name(text: str) -> bool:
    lowered = _compact_text(text).lower()
    return (
        ("siapa" in lowered and "berbicara" in lowered)
        or "boleh saya tahu" in lowered and ("nama" in lowered or "siapa" in lowered)
        or "may i know" in lowered and "name" in lowered
    )


def _is_conversation_marker(text: str) -> bool:
    return _compact_text(text).strip().lower() in {"[conversation begins]", "conversation begins"}


def _is_required_tool_user_message(text: str, inquiry_context_active: bool, rule: _RequiredToolRule) -> bool:
    lowered = _compact_text(text).lower()
    if _is_conversation_marker(lowered):
        return False
    if any(term in lowered for term in rule.domain_terms):
        return True
    if not inquiry_context_active:
        return False
    return _looks_like_short_followup(lowered)


def _looks_like_short_followup(lowered: str) -> bool:
    if _assistant_asks_question(lowered):
        return False
    words = re.findall(r"\w+", lowered)
    return 0 < len(words) <= 12


def _assistant_answers_required_tool_domain(text: str, rule: _RequiredToolRule) -> bool:
    lowered = _compact_text(text).lower()
    if any(
        phrase in lowered
        for phrase in (
            "mohon tunggu",
            "saya akan carikan",
            "akan carikan",
            "i will look",
            "let me check",
        )
    ):
        return False
    return any(term in lowered for term in rule.domain_terms)


def _domain_answer_excerpt(text: str, rule: _RequiredToolRule) -> str:
    compacted = _compact_text(text)
    for term in rule.domain_terms:
        pattern = rf"[^.!?]*\b{re.escape(term)}\b[^.!?]*[.!?]?"
        match = re.search(pattern, compacted, flags=re.IGNORECASE)
        if match:
            return _quote_turn(match.group(0), max_chars=120)
    return _quote_turn(compacted, max_chars=120)


def _format_turn_snippets(data: ConversationData, turn_indices: list[int]) -> str:
    return "; ".join(
        f"turn {index}: {_quote_turn(data.interactions[index].content)}"
        for index in turn_indices
        if 0 <= index < len(data.interactions)
    )


def _quote_turn(text: str, max_chars: int = 150) -> str:
    compacted = _compact_text(text)
    if len(compacted) > max_chars:
        compacted = compacted[: max_chars - 3].rstrip() + "..."
    return f'"{compacted}"'


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_bad_cases(cases: list[BadCase]) -> list[BadCase]:
    deduped_by_key: dict[tuple[int, str], BadCase] = {}
    for case in cases:
        key = _bad_case_semantic_key(case)
        existing = deduped_by_key.get(key)
        if existing is None or _bad_case_priority(case) > _bad_case_priority(existing):
            deduped_by_key[key] = case
    return list(deduped_by_key.values())


def _merge_manual_and_auto_bad_cases(
    manual_cases: list[BadCase],
    auto_cases: list[BadCase],
) -> list[BadCase]:
    manual_keys = {_bad_case_semantic_key(case) for case in manual_cases}
    merged = manual_cases[:]
    for case in _dedupe_bad_cases(auto_cases):
        if _bad_case_semantic_key(case) in manual_keys:
            continue
        merged.append(case)
    return merged


def _bad_case_priority(case: BadCase) -> int:
    if case.source == "human":
        return 30
    if case.source in {"local_scan", "mechanical_detector"}:
        return 20
    return 10


def _bad_case_semantic_key(case: BadCase) -> tuple[int, str]:
    return (case.turn_index, _bad_case_category(case))


def _bad_case_category(case: BadCase) -> str:
    text = " ".join(
        [
            case.error_type,
            case.evidence,
            case.recommendation,
        ]
    ).lower()
    if "tool" in text and ("missing" in text or "without" in text or "required" in text or "must call" in text):
        return "missing_required_tool_call"
    if "payment" in text and ("open-ended" in text or "when" in text or "how much" in text):
        return "open_ended_payment_terms_after_rejection"
    if ("attempt" in text or "3-attempt" in text or "three" in text) and ("rtp_closing" in text or "state 4.0" in text):
        return "ptp_attempt_limit_exceeded"
    return re.sub(r"[^a-z0-9]+", "_", case.error_type.lower()).strip("_") or "unknown"


def _parse_judge_bad_cases(data: ConversationData, content: str) -> list[BadCase]:
    parsed = json.loads(content)
    cases: list[BadCase] = []
    for item in parsed.get("bad_cases", []):
        if not _is_actionable_judge_item(item):
            continue
        turn_index = _normalize_ai_bad_case_turn_index(data, item)
        if turn_index is None:
            continue
        evidence = _judge_item_evidence(item)
        cases.append(
            BadCase(
                turn_index=turn_index,
                role=data.interactions[turn_index].role,
                error_type=str(item.get("error_type", "unknown")),
                evidence=evidence,
                recommendation=str(item.get("recommendation", ""))
                or _recommendation_from_judge_item(item),
            )
        )
    return cases


def _is_actionable_judge_item(item: dict[str, Any]) -> bool:
    hard_violation = _parse_bool(item.get("hard_violation"))
    if hard_violation is False:
        return False
    severity = str(item.get("severity", "")).strip().lower()
    if severity and severity not in {"high", "medium"}:
        return False
    confidence = _parse_float(item.get("confidence"))
    if confidence is not None and confidence < 0.65:
        return False
    violated_prompt_excerpt = _judge_violated_prompt_excerpt(item)
    if hard_violation is not True and not violated_prompt_excerpt:
        return False
    return True


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _judge_violated_prompt_excerpt(item: dict[str, Any]) -> str:
    for key in ("violated_prompt_excerpt", "violated_rule", "prompt_rule", "rule_excerpt"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _judge_item_evidence(item: dict[str, Any]) -> str:
    evidence = str(item.get("evidence", "")).strip()
    violated_prompt_excerpt = _judge_violated_prompt_excerpt(item)
    if violated_prompt_excerpt and violated_prompt_excerpt not in evidence:
        if evidence:
            return f"Violated rule: {violated_prompt_excerpt}\n\nEvidence: {evidence}"
        return f"Violated rule: {violated_prompt_excerpt}"
    return evidence


def _normalize_ai_bad_case_turn_index(data: ConversationData, item: dict[str, Any]) -> int | None:
    try:
        raw_index = int(item.get("turn_index", 0))
    except (TypeError, ValueError):
        return None
    role = str(item.get("role", "assistant")).lower()
    candidates = _candidate_bad_case_indices(data, raw_index)
    if role != "assistant":
        candidates.extend(_nearby_assistant_indices(data, raw_index, radius=2))
    best_candidate = _best_bad_case_candidate(data, item, candidates)
    if best_candidate is not None:
        return best_candidate
    return None


def _candidate_bad_case_indices(data: ConversationData, raw_index: int) -> list[int]:
    candidates: list[int] = []

    def add(index: int) -> None:
        if 0 <= index < len(data.interactions) and index not in candidates:
            candidates.append(index)

    add(raw_index)
    add(raw_index - 1)
    candidates.extend(index for index in _nearby_assistant_indices(data, raw_index, radius=3) if index not in candidates)

    assistant_indices = [
        index
        for index, turn in enumerate(data.interactions)
        if turn.role.lower() == "assistant"
    ]
    if 0 <= raw_index < len(assistant_indices):
        add(assistant_indices[raw_index])
    if 0 <= raw_index - 1 < len(assistant_indices):
        add(assistant_indices[raw_index - 1])
    return candidates


def _nearby_assistant_indices(data: ConversationData, raw_index: int, radius: int) -> list[int]:
    nearby: list[int] = []
    for offset in range(0, radius + 1):
        for candidate in (raw_index + offset, raw_index - offset):
            if candidate in nearby:
                continue
            if 0 <= candidate < len(data.interactions) and data.interactions[candidate].role.lower() == "assistant":
                nearby.append(candidate)
    return nearby


def _best_bad_case_candidate(
    data: ConversationData,
    item: dict[str, Any],
    candidates: list[int],
) -> int | None:
    scored: list[tuple[float, int]] = []
    for order, candidate in enumerate(candidates):
        if not 0 <= candidate < len(data.interactions):
            continue
        turn = data.interactions[candidate]
        if turn.role.lower() != "assistant":
            continue
        target_candidate = candidate
        if _looks_like_tool_wrapper(turn.content) and not _judge_item_mentions_tool_failure(item):
            next_natural = _next_natural_assistant_index(data, candidate)
            if next_natural is not None:
                target_candidate = next_natural
        target_turn = data.interactions[target_candidate]
        score = _bad_case_candidate_score(item, target_turn.content) - (order * 0.01)
        if target_candidate == candidate:
            score += 0.25
        else:
            score += 0.5
        scored.append((score, target_candidate))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def _bad_case_candidate_score(item: dict[str, Any], turn_content: str) -> float:
    item_text = " ".join(
        str(item.get(key, ""))
        for key in (
            "turn_content_excerpt",
            "content",
            "assistant_response",
            "quote",
            "evidence",
            "recommendation",
            "error_type",
        )
    )
    score = _token_overlap_score(item_text, turn_content)
    excerpt = str(item.get("turn_content_excerpt") or item.get("quote") or "").strip()
    if excerpt and excerpt.lower() in turn_content.lower():
        score += 3.0
    if _looks_like_tool_wrapper(turn_content) and not _judge_item_mentions_tool_failure(item):
        score -= 2.0
    return score


def _token_overlap_score(source: str, target: str) -> float:
    source_tokens = _meaningful_tokens(source)
    target_tokens = _meaningful_tokens(target)
    if not source_tokens or not target_tokens:
        return 0.0
    overlap = source_tokens.intersection(target_tokens)
    return len(overlap) / max(1, min(len(source_tokens), len(target_tokens)))


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "into", "then", "than",
        "assistant", "user", "should", "would", "could", "have", "has", "had", "was",
        "were", "are", "not", "but", "you", "your", "their", "there", "case",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _looks_like_tool_wrapper(content: str) -> bool:
    stripped = content.strip().lower()
    return (
        stripped.startswith("<function-call")
        or stripped.startswith("<tool-call")
        or stripped.startswith("<function-response")
        or stripped.startswith("<tool-response")
        or (
            stripped.startswith("{\"")
            and any(term in stripped[:120] for term in ("function", "tool", "arguments", "name"))
        )
        or (
            stripped.startswith("[{\"")
            and any(term in stripped[:160] for term in ("function", "tool", "arguments", "name"))
        )
    )


def _judge_item_mentions_tool_failure(item: dict[str, Any]) -> bool:
    text = " ".join(
        str(item.get(key, ""))
        for key in ("evidence", "recommendation", "error_type", "turn_content_excerpt", "quote")
    ).lower()
    return any(term in text for term in ("tool", "function", "function-call", "function call", "tool-call", "api"))


def _next_natural_assistant_index(data: ConversationData, start_index: int) -> int | None:
    for index in range(start_index + 1, min(len(data.interactions), start_index + 5)):
        turn = data.interactions[index]
        if turn.role.lower() == "assistant" and not _looks_like_tool_wrapper(turn.content):
            return index
    return None


def _recommendation_from_judge_item(item: dict[str, Any]) -> str:
    evidence = str(item.get("evidence", "")).strip()
    if evidence:
        return (
            "Revise the relevant system-prompt step so the assistant avoids this failure: "
            f"{evidence}"
        )
    return "Revise the relevant system-prompt step to prevent this bad case."


def optimize_system_prompt(
    data: ConversationData,
    bad_cases: list[BadCase],
    manual_feedback: dict[int, str] | None = None,
    model: str = DEFAULT_MODEL,
    llm_settings: LLMSettings | None = None,
) -> PromptOptimization:
    settings = llm_settings or LLMSettings(model=model)
    manual_feedback = _clean_manual_feedback(manual_feedback)
    if not bad_cases and not manual_feedback:
        return PromptOptimization(
            optimized_prompt=data.system_prompt,
            rationale="No bad cases or manual feedback were provided.",
            applied_feedback_summary="No changes applied.",
        )

    if not _has_llm_access(settings):
        fallback_prompt = _fallback_prompt_update(data.system_prompt, bad_cases, manual_feedback)
        return PromptOptimization(
            optimized_prompt=fallback_prompt,
            rationale="No reachable LLM backend is configured, so a deterministic local prompt update was generated.",
            applied_feedback_summary=_summarize_feedback(bad_cases, manual_feedback),
        )

    optimizer_prompt = (
        "You are a senior system-prompt optimizer. Rewrite the system prompt to prevent the listed failures. "
        "Preserve the original task and useful constraints. Human feedback has priority over AI judge feedback. "
        "Make the prompt directly usable, specific, and not overly long. Return JSON only with keys "
        "optimized_prompt, rationale, and applied_feedback_summary."
    )
    payload = {
        "current_system_prompt": data.system_prompt,
        "bad_cases": [case.__dict__ for case in bad_cases],
        "manual_feedback": manual_feedback,
        "interactions": [turn.model_dump(exclude_none=True) for turn in data.interactions],
    }
    try:
        content = _chat_json(
            settings=settings,
            messages=[
                {"role": "system", "content": optimizer_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            purpose="bulk_prompt_optimize",
        )
        parsed = json.loads(content)
        return PromptOptimization(
            optimized_prompt=str(parsed.get("optimized_prompt") or data.system_prompt),
            rationale=str(parsed.get("rationale") or ""),
            applied_feedback_summary=str(parsed.get("applied_feedback_summary") or ""),
        )
    except (OpenAIError, requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return PromptOptimization(
            optimized_prompt=_fallback_prompt_update(data.system_prompt, bad_cases, manual_feedback),
            rationale=f"LLM optimization failed: {exc}. A local fallback update was generated.",
            applied_feedback_summary=_summarize_feedback(bad_cases, manual_feedback),
        )


def apply_recommendation_to_system_prompt(
    data: ConversationData,
    current_system_prompt: str,
    bad_case: BadCase,
    llm_settings: LLMSettings | None = None,
    force_prompt_edit: bool = False,
) -> PromptOptimization:
    settings = llm_settings or LLMSettings()
    recommendation = bad_case.recommendation.strip() or _recommendation_from_bad_case(bad_case)

    if not _has_llm_access(settings):
        return PromptOptimization(
            optimized_prompt=current_system_prompt,
            rationale=(
                "No reachable LLM backend is configured. The prompt was left unchanged because "
                "step-preserving edits require an LLM to rewrite the relevant numbered step in place."
            ),
            applied_feedback_summary="No changes applied.",
        )

    already_addresses_case = _prompt_already_addresses_case(current_system_prompt, bad_case)
    if already_addresses_case and not force_prompt_edit:
        return PromptOptimization(
            optimized_prompt=current_system_prompt,
            rationale=already_addresses_case,
            applied_feedback_summary="No new prompt change was needed; the working prompt already covers this bad case.",
        )

    turn = (
        data.interactions[bad_case.turn_index].model_dump()
        if 0 <= bad_case.turn_index < len(data.interactions)
        else None
    )
    nearby_turns = _nearby_turn_context(data, bad_case.turn_index)
    payload = {
        "current_system_prompt": current_system_prompt,
        "bad_case": bad_case.__dict__,
        "related_turn": turn,
        "nearby_turns": nearby_turns,
        "recommendation_to_apply": recommendation,
    }
    patch_optimization = _apply_recommendation_patch_to_system_prompt(
        settings=settings,
        current_system_prompt=current_system_prompt,
        payload=payload,
    )
    if patch_optimization is not None and patch_optimization.optimized_prompt != current_system_prompt:
        return patch_optimization
    patch_retry = _retry_prompt_edit_as_patch(
        settings=settings,
        current_system_prompt=current_system_prompt,
        bad_case=bad_case,
        payload=payload,
        failure_reason="Initial exact replacement patch was missing, unchanged, invalid, or did not match the current prompt.",
    )
    if patch_retry is not None:
        return patch_retry
    deterministic_patch = _deterministic_prompt_patch(
        current_system_prompt,
        bad_case,
        force_prompt_edit=force_prompt_edit,
    )
    if deterministic_patch is not None:
        return deterministic_patch
    return PromptOptimization(
        optimized_prompt=current_system_prompt,
        rationale=(
            "LLM prompt edit failed: the backend did not return an applicable exact replacement patch. "
            "The prompt was left unchanged because Apply only supports local in-place prompt patches."
        ),
        applied_feedback_summary="No changes applied.",
    )


def _nearby_turn_context(data: ConversationData, turn_index: int, radius_before: int = 3, radius_after: int = 1) -> list[dict[str, Any]]:
    if not 0 <= turn_index < len(data.interactions):
        return []
    start = max(0, turn_index - radius_before)
    end = min(len(data.interactions), turn_index + radius_after + 1)
    return [
        {
            "turn_index": index,
            **data.interactions[index].model_dump(),
        }
        for index in range(start, end)
    ]


def _apply_recommendation_patch_to_system_prompt(
    settings: LLMSettings,
    current_system_prompt: str,
    payload: dict[str, Any],
) -> PromptOptimization | None:
    patch_prompt = (
        "You are a senior system-prompt patch editor. Apply exactly one bad-case recommendation "
        "by returning a small exact replacement patch, not the full prompt. Preserve the original "
        "format and modify an existing relevant Step/substep/branch in place. Return JSON only "
        "with keys replace, rationale, and applied_feedback_summary. replace must be an object "
        "with old and new strings. old must be copied exactly from current_system_prompt as one "
        "contiguous substring. new is the replacement text. Do not add appendix/addendum sections."
        f" {PROMPT_EDIT_STYLE_RULES} "
        " If the bad case involves a vague, uncertain, partial, refused, or unrelated user reply, "
        "edit the relevant existing workflow branch so the assistant must not advance to the next "
        "step until the required condition is satisfied; it should acknowledge uncertainty and ask "
        "one focused clarification question or use the allowed fallback from that same step."
    )
    try:
        content = _chat_json(
            settings=settings,
            messages=[
                {"role": "system", "content": patch_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            purpose="prompt_edit_patch",
        )
        parsed = json.loads(content)
        patched_prompt = _apply_prompt_replace_patch(current_system_prompt, parsed)
        if patched_prompt is None or patched_prompt == current_system_prompt:
            return None
        if _has_step_append_violation(current_system_prompt, patched_prompt):
            return None
        if _prompt_edit_style_violation(current_system_prompt, patched_prompt):
            return None
        if _prompt_patch_relevance_violation(
            current_system_prompt,
            patched_prompt,
            payload.get("bad_case"),
        ):
            return None
        return PromptOptimization(
            optimized_prompt=patched_prompt,
            rationale=str(parsed.get("rationale") or "Applied exact replacement patch."),
            applied_feedback_summary=str(parsed.get("applied_feedback_summary") or "Applied prompt patch."),
        )
    except (OpenAIError, requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError, TypeError):
        return None


def _apply_prompt_replace_patch(current_system_prompt: str, parsed: dict[str, Any]) -> str | None:
    replace = parsed.get("replace")
    if not isinstance(replace, dict):
        patch = parsed.get("patch")
        if isinstance(patch, dict):
            replace = patch.get("replace")
    if not isinstance(replace, dict):
        return None
    old_text = replace.get("old")
    if not isinstance(old_text, str):
        old_text = replace.get("old_text")
    new_text = replace.get("new")
    if not isinstance(new_text, str):
        new_text = replace.get("new_text")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return None
    if not old_text or old_text not in current_system_prompt or old_text == new_text:
        return None
    return current_system_prompt.replace(old_text, new_text, 1)


def _apply_recommendation_full_prompt(
    settings: LLMSettings,
    current_system_prompt: str,
    bad_case: BadCase,
    related_turn: dict[str, Any] | None,
    payload: dict[str, Any],
) -> PromptOptimization:
    optimizer_prompt = (
        "You are a senior system-prompt editor. Apply exactly one bad-case recommendation to the "
        "current system prompt. Preserve the original task, style, and all useful constraints. "
        "Keep the prompt's existing structure and formatting, especially numbered step-by-step "
        "sections such as Step 1, Step 2, Step 2.1, etc. Do not append a separate 'Additional "
        "reliability requirements', 'Step update note', notes, addendum, patch, or appendix section. "
        "Instead, revise the relevant existing step, substep, branch, or nearby rule in place by "
        "adding, deleting, or replacing wording inside that existing step-by-step flow. Keep the "
        "same Step headings and overall order unless the recommendation explicitly requires "
        "renumbering. Make the smallest clear edit needed to prevent the failure. "
        f"{PROMPT_EDIT_STYLE_RULES} "
        "For vague, uncertain, partial, refused, or unrelated user replies, prefer explicit branch "
        "conditions such as: do not advance to the next workflow step; acknowledge the ambiguity; "
        "ask one focused clarification question or use the allowed fallback; continue only after "
        "the required information or confirmation is clearly provided. "
        "Return JSON only with keys optimized_prompt, rationale, and applied_feedback_summary."
    )
    try:
        content = _chat_json(
            settings=settings,
            messages=[
                {"role": "system", "content": optimizer_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            purpose="prompt_edit",
        )
        parsed = json.loads(content)
        optimized_prompt = str(parsed.get("optimized_prompt") or current_system_prompt)
        integrity_violation = _prompt_edit_integrity_violation(current_system_prompt, optimized_prompt)
        style_violation = _prompt_edit_style_violation(current_system_prompt, optimized_prompt)
        if (
            integrity_violation
            or _has_step_append_violation(current_system_prompt, optimized_prompt)
            or style_violation
        ):
            rejection_reason = (
                integrity_violation
                or style_violation
                or "appended a separate note/requirements section"
            )
            optimized_prompt = _retry_step_preserving_prompt_edit(
                settings=settings,
                current_system_prompt=current_system_prompt,
                bad_case=bad_case,
                related_turn=related_turn,
                rejected_prompt=optimized_prompt,
                rejection_reason=rejection_reason,
            )
            if optimized_prompt == current_system_prompt and integrity_violation:
                return PromptOptimization(
                    optimized_prompt=current_system_prompt,
                    rationale=(
                        f"LLM prompt edit rejected: {rejection_reason}. "
                        "The prompt was left unchanged to avoid dropping existing workflow sections."
                    ),
                    applied_feedback_summary="No changes applied.",
                )
        integrity_violation = _prompt_edit_integrity_violation(current_system_prompt, optimized_prompt)
        if integrity_violation:
            return PromptOptimization(
                optimized_prompt=current_system_prompt,
                rationale=(
                    f"LLM prompt edit rejected: {integrity_violation}. "
                    "The prompt was left unchanged to avoid dropping existing workflow sections."
                ),
                applied_feedback_summary="No changes applied.",
            )
        return PromptOptimization(
            optimized_prompt=optimized_prompt,
            rationale=str(parsed.get("rationale") or ""),
            applied_feedback_summary=str(parsed.get("applied_feedback_summary") or ""),
        )
    except (OpenAIError, requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        patch_retry = _retry_prompt_edit_as_patch(
            settings=settings,
            current_system_prompt=current_system_prompt,
            bad_case=bad_case,
            payload=payload,
            failure_reason=str(exc),
        )
        if patch_retry is not None:
            return patch_retry
        return PromptOptimization(
            optimized_prompt=current_system_prompt,
            rationale=(
                f"LLM prompt edit failed: {exc}. The prompt was left unchanged because this app "
                "now requires in-place step edits rather than appended notes."
            ),
            applied_feedback_summary="No changes applied.",
        )


def _retry_prompt_edit_as_patch(
    settings: LLMSettings,
    current_system_prompt: str,
    bad_case: BadCase,
    payload: dict[str, Any],
    failure_reason: str,
) -> PromptOptimization | None:
    retry_prompt = (
        "The previous prompt patch attempt failed or did not apply cleanly. "
        "Do not return the full prompt. Return exactly one small JSON object containing an exact "
        "replacement patch with keys replace, rationale, and applied_feedback_summary. "
        "replace.old must be copied exactly from current_system_prompt as one contiguous substring. "
        "replace.new must be the revised replacement text. Use JSON string escaping for all newlines "
        "and quotes. No markdown fences, no prose, no comments. Preserve the existing step structure "
        "and modify the relevant existing branch in place."
    )
    retry_payload = {
        **payload,
        "failed_patch_reason": failure_reason,
    }
    try:
        content = _chat_json(
            settings=settings,
            messages=[
                {"role": "system", "content": retry_prompt},
                {"role": "user", "content": json.dumps(retry_payload, ensure_ascii=False)},
            ],
            purpose="prompt_edit_json_patch_retry",
        )
        parsed = json.loads(content)
        patched_prompt = _apply_prompt_replace_patch(current_system_prompt, parsed)
        if patched_prompt is None or patched_prompt == current_system_prompt:
            return None
        if _prompt_edit_integrity_violation(current_system_prompt, patched_prompt):
            return None
        if _prompt_edit_integrity_violation(current_system_prompt, patched_prompt):
            return None
        if _has_step_append_violation(current_system_prompt, patched_prompt):
            return None
        if _prompt_edit_style_violation(current_system_prompt, patched_prompt):
            return None
        if _prompt_patch_relevance_violation(
            current_system_prompt,
            patched_prompt,
            payload.get("bad_case"),
        ):
            return None
        return PromptOptimization(
            optimized_prompt=patched_prompt,
            rationale=str(parsed.get("rationale") or "Applied JSON repair patch retry."),
            applied_feedback_summary=str(parsed.get("applied_feedback_summary") or "Applied prompt patch retry."),
        )
    except (OpenAIError, requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError, TypeError):
        return None


def rerun_conversation(
    data: ConversationData,
    optimized_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    llm_settings: LLMSettings | None = None,
    target_assistant_turn_indices: set[int] | None = None,
    required_tools_by_turn: dict[int, str] | None = None,
    required_exact_responses_by_turn: dict[int, str] | None = None,
    expected_system_prompt_hash: str | None = None,
    expected_prompt_version: str | None = None,
) -> list[RerunTurn]:
    settings = llm_settings or LLMSettings(model=model)
    results: list[RerunTurn] = []
    replay_messages: list[dict[str, str]] = [{"role": "system", "content": optimized_prompt}]
    rerun_all = target_assistant_turn_indices is None
    explicit_targeted_rerun = target_assistant_turn_indices is not None
    if target_assistant_turn_indices is None:
        target_assistant_turn_indices = {
            index for index, turn in enumerate(data.interactions) if turn.role == "assistant"
        }
    required_tools_by_turn = required_tools_by_turn or {}
    required_exact_responses_by_turn = required_exact_responses_by_turn or {}

    if not _has_llm_access(settings):
        return [
            RerunTurn(
                user_turn_index=-1,
                assistant_turn_index=None,
                user_message="",
                old_assistant_response="",
                new_assistant_response="",
                error="No reachable LLM backend is configured. For OpenAI set OPENAI_API_KEY, or choose Company API with url/provider/model.",
            )
        ]

    pending_user: tuple[int, str] | None = None
    latest_user: tuple[int, str] | None = None
    for index, turn in enumerate(data.interactions):
        if turn.role == "user":
            pending_user = (index, turn.content)
            latest_user = pending_user
            replay_messages.append(_interaction_replay_message(turn))
            continue

        if turn.role == "assistant" and (
            pending_user is not None
            or (explicit_targeted_rerun and index in target_assistant_turn_indices)
        ):
            has_immediate_user = pending_user is not None
            user_index, user_message = pending_user or latest_user or (-1, "")
            if index not in target_assistant_turn_indices:
                replay_messages.append(_interaction_replay_message(turn))
                pending_user = None
                continue
            try:
                request_messages = _rerun_request_messages(
                    replay_messages,
                    settings.rerun_context_window_turns,
                )
                required_tool = required_tools_by_turn.get(index)
                required_exact_response = required_exact_responses_by_turn.get(index)
                request_messages = _with_required_tool_instruction(request_messages, required_tool)
                request_messages = _with_required_exact_response_instruction(
                    request_messages,
                    required_exact_response,
                )
                tool_choice = _tool_choice_for_required_tool(required_tool, data.tools)
                _preflight_company_rerun_request(
                    settings=settings,
                    messages=request_messages,
                    tools=data.tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    purpose="targeted_rerun",
                    expected_system_prompt_hash=expected_system_prompt_hash,
                    expected_prompt_version=expected_prompt_version,
                )
                new_response = _chat_text(
                    settings=settings,
                    messages=request_messages,
                    temperature=temperature,
                    tools=data.tools,
                    tool_choice=tool_choice,
                    purpose="targeted_rerun",
                    expected_system_prompt_hash=expected_system_prompt_hash,
                    expected_prompt_version=expected_prompt_version,
                )
                response_diagnostics = _last_response_diagnostics(settings)
                if required_tool and not _contains_required_tool_call(new_response, required_tool):
                    forced_query_source = (
                        user_message
                        if has_immediate_user
                        else _forced_tool_query_source(request_messages, user_message, turn.content)
                    )
                    new_response = _forced_required_tool_call(required_tool, forced_query_source)
                    response_diagnostics = {
                        **(response_diagnostics or {}),
                        "forced_required_tool_call": {
                            "tool_name": required_tool,
                            "reason": "Model rerun returned natural language despite required tool-call hint/tool_choice.",
                        },
                    }
                replay_messages.append({"role": "assistant", "content": new_response})
                results.append(
                    RerunTurn(
                        user_turn_index=user_index,
                        assistant_turn_index=index,
                        user_message=user_message,
                        old_assistant_response=turn.content,
                        new_assistant_response=new_response,
                        response_diagnostics=response_diagnostics,
                    )
                )
            except (OpenAIError, requests.RequestException, RuntimeError) as exc:
                results.append(
                    RerunTurn(
                        user_turn_index=user_index,
                        assistant_turn_index=index,
                        user_message=user_message,
                        old_assistant_response=turn.content,
                        new_assistant_response="",
                        error=str(exc),
                    )
                )
            pending_user = None
            continue

        replay_messages.append(_interaction_replay_message(turn))

    if pending_user is not None and rerun_all:
        user_index, user_message = pending_user
        try:
            request_messages = _rerun_request_messages(
                replay_messages,
                settings.rerun_context_window_turns,
            )
            required_tool = required_tools_by_turn.get(-1)
            required_exact_response = required_exact_responses_by_turn.get(-1)
            request_messages = _with_required_tool_instruction(request_messages, required_tool)
            request_messages = _with_required_exact_response_instruction(
                request_messages,
                required_exact_response,
            )
            tool_choice = _tool_choice_for_required_tool(required_tool, data.tools)
            _preflight_company_rerun_request(
                settings=settings,
                messages=request_messages,
                tools=data.tools,
                tool_choice=tool_choice,
                temperature=temperature,
                purpose="full_rerun",
                expected_system_prompt_hash=expected_system_prompt_hash,
                expected_prompt_version=expected_prompt_version,
            )
            new_response = _chat_text(
                settings=settings,
                messages=request_messages,
                temperature=temperature,
                tools=data.tools,
                tool_choice=tool_choice,
                purpose="full_rerun",
                expected_system_prompt_hash=expected_system_prompt_hash,
                expected_prompt_version=expected_prompt_version,
            )
            response_diagnostics = _last_response_diagnostics(settings)
            if required_tool and not _contains_required_tool_call(new_response, required_tool):
                new_response = _forced_required_tool_call(required_tool, user_message)
                response_diagnostics = {
                    **(response_diagnostics or {}),
                    "forced_required_tool_call": {
                        "tool_name": required_tool,
                        "reason": "Model rerun returned natural language despite required tool-call hint/tool_choice.",
                    },
                }
            results.append(
                RerunTurn(
                    user_turn_index=user_index,
                    assistant_turn_index=None,
                    user_message=user_message,
                    old_assistant_response="",
                    new_assistant_response=new_response,
                    response_diagnostics=response_diagnostics,
                )
            )
        except (OpenAIError, requests.RequestException, RuntimeError) as exc:
            results.append(
                RerunTurn(
                    user_turn_index=user_index,
                    assistant_turn_index=None,
                    user_message=user_message,
                    old_assistant_response="",
                    new_assistant_response="",
                    error=str(exc),
                )
            )
    return results


def _rerun_request_messages(
    replay_messages: list[dict[str, str]],
    context_window_turns: int | None,
) -> list[dict[str, str]]:
    if context_window_turns == AUTO_RERUN_CONTEXT_WINDOW:
        return _auto_rerun_request_messages(replay_messages)
    if context_window_turns is None or context_window_turns <= 0 or len(replay_messages) <= 1:
        return replay_messages
    system_message = replay_messages[0]
    recent_messages = replay_messages[1:][-context_window_turns:]
    return [system_message, *recent_messages]


def _interaction_replay_message(turn: Interaction) -> dict[str, str]:
    return {
        "role": turn.role,
        "content": turn.content,
    }


def _with_required_tool_instruction(
    messages: list[dict[str, str]],
    required_tool: str | None,
) -> list[dict[str, str]]:
    if not required_tool:
        return messages
    instruction = (
        "For the next assistant turn, you MUST call the function/tool "
        f"`{required_tool}`. Do not answer in natural language before this tool call. "
        "Use the latest customer message and relevant immediate context to populate the arguments."
    )
    return [*messages, {"role": "system", "content": instruction}]


def _with_required_exact_response_instruction(
    messages: list[dict[str, str]],
    required_response: str | None,
) -> list[dict[str, str]]:
    if not required_response:
        return messages
    instruction = (
        "For the next assistant turn, an exact spoken escalation action is required. "
        "Return exactly this text, with no extra words, no translation, no question, and no tool call:\n"
        f"{required_response}"
    )
    return [*messages, {"role": "system", "content": instruction}]


def _tool_choice_for_required_tool(
    required_tool: str | None,
    tools: dict[str, Any] | None,
) -> dict[str, Any] | str | None:
    if not required_tool or not tools:
        return None
    if required_tool not in tools:
        return None
    return {
        "type": "function",
        "function": {
            "name": required_tool,
        },
    }


def _contains_required_tool_call(response: str, required_tool: str) -> bool:
    if not response:
        return False
    pattern = rf"<function-call>\s*{re.escape(required_tool)}\s*:"
    return bool(re.search(pattern, response, flags=re.IGNORECASE))


def _forced_required_tool_call(required_tool: str, user_message: str) -> str:
    arguments = {"query": _forced_tool_query(user_message)}
    return f"<function-call>{required_tool}:{json.dumps(arguments, ensure_ascii=False)}</function-call>"


def _forced_tool_query_source(
    request_messages: list[dict[str, str]],
    fallback_user_message: str,
    old_assistant_response: str,
) -> str:
    user_messages = [
        str(message.get("content") or "")
        for message in request_messages
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    ]
    pieces = user_messages[-3:]
    if old_assistant_response.strip():
        pieces.append(old_assistant_response)
    if not pieces and fallback_user_message.strip():
        pieces.append(fallback_user_message)
    return " ".join(pieces)


def _forced_tool_query(user_message: str) -> str:
    compacted = _compact_text(user_message)
    return compacted[:240] if compacted else ""


def _auto_rerun_request_messages(replay_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(replay_messages) <= 1:
        return replay_messages
    system_message = replay_messages[0]
    context_messages = replay_messages[1:]
    if len(context_messages) <= AUTO_RERUN_CONTEXT_MIN_MESSAGES:
        return replay_messages

    selected_reversed: list[dict[str, str]] = []
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


def generate_experiment_conclusion(
    data: ConversationData,
    before_prompt: str,
    optimized_prompt: str,
    rerun_results: list[RerunTurn],
    updated_interactions: list[Interaction],
    post_rerun_bad_cases: list[BadCase],
    applied_feedback_summary: str,
    llm_settings: LLMSettings | None = None,
    post_rerun_scan_status: str = "completed",
    prompt_version_history: list[dict[str, Any]] | None = None,
) -> str:
    settings = llm_settings or LLMSettings()
    failed = [result for result in rerun_results if result.error]
    if not _has_llm_access(settings):
        return (
            "Conclusion could not be generated because no reachable LLM backend is configured. "
            "Configure OpenAI or Company API and rerun the experiment."
        )
    if failed:
        return (
            "Conclusion was skipped because one or more assistant turns failed during rerun. "
            "Fix the rerun errors first so the experiment result is complete."
        )

    conclusion_prompt = (
        "You are an experiment analyst for prompt-optimization runs. Compare the original conversation "
        "with the newly generated conversation after a system-prompt edit. The conclusion should be in English only, "
        "except when quoting evidence from the conversation. Return JSON only in this exact shape: "
        '{"conclusion":{"paragraph_1":"...","paragraph_2":"...","paragraph_3":"..."}}. '
        "Each value must be one short paragraph. Do not include headings, bullets, numbered labels, or markdown lists. "
        "Keep all three paragraphs together under 180 words.\n\n"
        "paragraph_1 must give the verification verdict: state whether the approved badcase was actually fixed, "
        "partially fixed, or not fixed. Base the verdict on residual scan results and target-turn behavior, not on "
        "whether a prompt edit or rerun merely executed.\n\n"
        "paragraph_2 must diagnose why the badcase passed or failed and connect that diagnosis to prompt-change and "
        "backend evidence using paragraph_inputs.paragraph_2_improvement_evidence and "
        "paragraph_inputs.paragraph_1_changes. Mention the concrete violated behavior, whether the prompt changed, "
        "and available rerun/tool/token/backend metadata. Do not claim the response is identical when "
        "target_turn_comparisons says only wording changed; instead say wording changed but target behavior did or did "
        "not improve. Use token/logprob evidence only when token_probability.available is true. If it is false, say "
        "that token logprobs were requested but are unavailable, empty, unsupported, or missing according to "
        "token_probability.reasons, so token-probability analysis cannot be performed. Then use behavior/meta/tool "
        "evidence instead. Treat diagnostic_evidence.response_metadata as backend execution metadata, not "
        "token-confidence evidence. High-confidence logprobs on a wrong response mean the backend strongly preferred "
        "the wrong behavior; they do not prove compliance.\n\n"
        "paragraph_3 must give the single best next action for the current verified state using "
        "paragraph_inputs.paragraph_3_next_recommendation. If verification passed, say to keep the version and stop. "
        "If it failed, tie the action to the observed root cause and backend evidence. If repair_saturation says "
        "unfixable_assessment_required or likely_not_fixable_by_prompt_only, explicitly assess whether another "
        "prompt-only version is no longer justified; recommend unsupported/missing-ground-truth handling, a "
        "deterministic conversation replacement, or a non-prompt workflow/model change instead of another generic "
        "prompt patch. Include why, implementation steps, acceptance criteria, and a fallback; avoid generic advice "
        "or repeating an unchanged failed experiment."
    )
    payload = _build_conclusion_payload(
        data=data,
        before_prompt=before_prompt,
        optimized_prompt=optimized_prompt,
        rerun_results=rerun_results,
        updated_interactions=updated_interactions,
        post_rerun_bad_cases=post_rerun_bad_cases,
        applied_feedback_summary=applied_feedback_summary,
        post_rerun_scan_status=post_rerun_scan_status,
        prompt_version_history=prompt_version_history,
    )
    payload_json = json.dumps(payload, ensure_ascii=False)
    if len(payload_json) > 45000:
        return _deterministic_experiment_conclusion(
            before_prompt=before_prompt,
            optimized_prompt=optimized_prompt,
            rerun_results=rerun_results,
            post_rerun_bad_cases=post_rerun_bad_cases,
            applied_feedback_summary=applied_feedback_summary,
            post_rerun_scan_status=post_rerun_scan_status,
            prompt_version_history=prompt_version_history,
        )
    messages = [
        {"role": "system", "content": conclusion_prompt},
        {"role": "user", "content": payload_json},
    ]
    _write_conclusion_dialog_log(messages)
    try:
        conclusion_settings = replace(
            settings,
            max_completion_tokens=min(settings.max_completion_tokens, 1024),
        )
        content = _chat_json(
            settings=conclusion_settings,
            messages=messages,
            purpose="conclusion",
        )
        parsed = json.loads(content)
        return _format_conclusion_value(parsed.get("conclusion"))
    except (OpenAIError, requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        if "maximum context length" in str(exc).lower() or "input_tokens" in str(exc).lower():
            return _deterministic_experiment_conclusion(
                before_prompt=before_prompt,
                optimized_prompt=optimized_prompt,
                rerun_results=rerun_results,
                post_rerun_bad_cases=post_rerun_bad_cases,
                applied_feedback_summary=applied_feedback_summary,
                post_rerun_scan_status=post_rerun_scan_status,
                prompt_version_history=prompt_version_history,
            )
        return f"Conclusion generation failed: {exc}"


def _deterministic_experiment_conclusion(
    *,
    before_prompt: str,
    optimized_prompt: str,
    rerun_results: list[RerunTurn],
    post_rerun_bad_cases: list[BadCase],
    applied_feedback_summary: str,
    post_rerun_scan_status: str,
    prompt_version_history: list[dict[str, Any]] | None = None,
) -> str:
    prompt_changed = before_prompt.strip() != optimized_prompt.strip()
    target_turns = [
        str(result.assistant_turn_index)
        for result in rerun_results
        if result.assistant_turn_index is not None
    ]
    token_probability = _token_probability_diagnostics(rerun_results)
    logprob_available = token_probability.get("available") is True
    residual_count = len(post_rerun_bad_cases)
    repair_saturation = _prompt_repair_saturation_diagnostics(
        prompt_version_history,
        post_rerun_bad_cases,
        post_rerun_scan_status,
    )
    if post_rerun_scan_status != "completed":
        first = "Verification is incomplete because the post-rerun badcase scan did not complete."
    elif residual_count == 0:
        first = f"Verified fixed: target assistant turn(s) {', '.join(target_turns) or '-'} have no residual badcases."
    else:
        first = (
            f"Not verified fixed: the post-rerun scan still found {residual_count} residual badcase(s) "
            f"on target assistant turn(s) {', '.join(target_turns) or '-'}."
        )
    prompt_evidence = (
        f"The system prompt changed: {applied_feedback_summary}"
        if prompt_changed
        else "No new system-prompt version was created; the target turn was rerun with the existing prompt."
    )
    if logprob_available:
        token_evidence = "Token-probability evidence was available and can be used only as backend confidence context"
    else:
        unavailable_reasons = token_probability.get("reasons") or [token_probability.get("reason")]
        reason_text = "; ".join(str(reason) for reason in unavailable_reasons if reason)
        if reason_text:
            token_evidence = (
                "Token logprobs were requested but are unavailable for analysis "
                f"({reason_text})"
            )
        else:
            token_evidence = "Token logprobs are unavailable for analysis"
    second = f"{prompt_evidence} {token_evidence}; the residual scan is the verification source of truth."
    third = (
        "The cycle is complete; keep this prompt version and do not run another scan unless requested."
        if residual_count == 0
        else (
            "Residual badcases remain after multiple prompt versions; treat prompt-only repair as saturated, "
            "assess whether the case is unsupported or not fixable with the current model/workflow, then use a "
            "deterministic conversation replacement or missing-ground-truth workflow with acceptance criteria of "
            "zero residual badcases."
            if repair_saturation.get("likely_not_fixable_by_prompt_only")
            else "Residual badcases remain; review the remaining traces before applying another targeted prompt patch."
        )
    )
    return "\n\n".join([first, second, third])


def _write_conclusion_dialog_log(
    messages: list[dict[str, str]],
    path: Path | None = None,
) -> None:
    output_path = path or DEFAULT_CONCLUSION_DIALOG_LOG_PATH
    if len(messages) < 2:
        return
    dialog = [
        {
            "turn_index": 0,
            "role": str(messages[0].get("role", "system")),
            "content": str(messages[0].get("content", "")),
        },
        {
            "turn_index": 1,
            "role": str(messages[1].get("role", "user")),
            "content": str(messages[1].get("content", "")),
        },
        {
            "turn_index": 2,
            "role": "assistant",
            "content": f'{LOSS_START}{{"conclusion": ""}}{LOSS_END}',
        },
    ]
    record = {
        "type": "compress",
        "dialog": dialog,
        "tools": {},
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        pass


def _build_conclusion_payload(
    data: ConversationData,
    before_prompt: str,
    optimized_prompt: str,
    rerun_results: list[RerunTurn],
    updated_interactions: list[Interaction],
    post_rerun_bad_cases: list[BadCase],
    applied_feedback_summary: str,
    post_rerun_scan_status: str,
    prompt_version_history: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    prompt_diff = _prompt_unified_diff(before_prompt, optimized_prompt)
    prompt_changed = before_prompt.strip() != optimized_prompt.strip()
    rerun_evidence = [_rerun_evidence_item(result) for result in rerun_results]
    diagnostic_evidence = _experiment_diagnostic_evidence(
        data=data,
        rerun_results=rerun_results,
        post_rerun_bad_cases=post_rerun_bad_cases,
        post_rerun_scan_status=post_rerun_scan_status,
    )
    token_probability = diagnostic_evidence.get("token_probability", {})
    target_turn_comparisons = [_target_turn_comparison(item) for item in rerun_evidence]
    prompt_change_quality = _prompt_change_quality(before_prompt, optimized_prompt)
    repair_saturation = _prompt_repair_saturation_diagnostics(
        prompt_version_history,
        post_rerun_bad_cases,
        post_rerun_scan_status,
    )
    return {
        "output_contract": {
            "format": {
                "conclusion": {
                    "paragraph_1": "verification verdict: whether it was actually fixed",
                    "paragraph_2": "badcase diagnosis connected to prompt and backend evidence",
                    "paragraph_3": "best next action for the verified state",
                }
            },
            "constraints": [
                "English only except quoted conversation evidence.",
                "Exactly three short paragraphs after formatting.",
                "No section titles, bullets, numbered labels, or markdown lists.",
                "Under 180 words total.",
            ],
        },
        "paragraph_inputs": {
            "paragraph_1_changes": {
                "applied_feedback_summary": applied_feedback_summary,
                "prompt_changed": prompt_changed,
                "prompt_diff": prompt_diff,
                "prompt_change_quality": prompt_change_quality,
                "instruction": (
                    "State the exact changed rule. If prompt_changed is false, state that no new "
                    "system-prompt version was needed and the existing prompt was used for a conversation-only "
                    "target rerun. If the edit only repeats an abstract instruction or duplicates wording, "
                    "say that clearly."
                ),
            },
            "paragraph_2_improvement_evidence": {
                "target_turn_comparisons": target_turn_comparisons,
                "diagnostic_evidence": diagnostic_evidence,
                "post_rerun_bad_cases": [case.__dict__ for case in post_rerun_bad_cases],
                "token_probability": token_probability,
                "evidence_rules": [
                    "Do not say old and new responses are identical unless exact_text_changed is false.",
                    "If exact_text_changed is true but approximate_similarity is high, say wording changed and judge the behavior.",
                    "Use token probability only when available is true; otherwise say there is no token-probability evidence.",
                    "If token probability is unavailable, mention whether logprobs were requested and cite the unavailable reason.",
                    "Prefer behavior, tool-call state, metadata, and residual scan evidence over prompt wording alone.",
                ],
            },
            "paragraph_3_next_recommendation": {
                "repair_saturation": repair_saturation,
                "observed_root_cause_hints": _conclusion_root_cause_hints(
                    target_turn_comparisons=target_turn_comparisons,
                    prompt_change_quality=prompt_change_quality,
                    token_probability=token_probability,
                    post_rerun_bad_cases=post_rerun_bad_cases,
                    post_rerun_scan_status=post_rerun_scan_status,
                    repair_saturation=repair_saturation,
                ),
                "recommendation_rules": [
                    "Recommend one concrete prompt or workflow edit.",
                    "If the current edit duplicated abstract guidance, recommend replacing it with a concrete response pattern.",
                    "If rerun behavior still misses the required action, recommend an explicit branch/action gate.",
                    "If repair_saturation.likely_not_fixable_by_prompt_only is true, do not recommend another prompt-only patch unless new evidence changes the experiment; recommend unsupported/missing-ground-truth handling, deterministic replacement, workflow/tool correction, or model/backend escalation.",
                ],
            },
        },
        "raw_evidence": {
            "before_prompt_preview": _truncate_for_conclusion(before_prompt),
            "optimized_prompt_preview": _truncate_for_conclusion(optimized_prompt),
            "prompt_diff": prompt_diff,
            "source_meta": data.source_meta,
            "tool_names": sorted(str(key) for key in (data.tools or {}).keys()),
            "original_turn_count": len(data.interactions),
            "updated_turn_count": len(updated_interactions),
            "rerun_results": [_rerun_result_summary(result) for result in rerun_results],
            "post_rerun_bad_cases": [case.__dict__ for case in post_rerun_bad_cases],
        },
    }


def _truncate_for_conclusion(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    head = max(0, limit // 2)
    tail = max(0, limit - head)
    return (
        text[:head]
        + f"\n...[truncated {len(text) - limit} characters for conclusion payload]...\n"
        + text[-tail:]
    )


def _rerun_result_summary(result: RerunTurn) -> dict[str, object]:
    diagnostics = result.response_diagnostics or {}
    logprobs = diagnostics.get("logprobs") if isinstance(diagnostics, dict) else None
    response_meta = diagnostics.get("response_meta") if isinstance(diagnostics, dict) else None
    return {
        "user_turn_index": result.user_turn_index,
        "assistant_turn_index": result.assistant_turn_index,
        "old_assistant_response_preview": _truncate_for_conclusion(result.old_assistant_response, limit=1200),
        "new_assistant_response_preview": _truncate_for_conclusion(result.new_assistant_response, limit=1200),
        "error": result.error,
        "logprobs": logprobs,
        "response_meta": response_meta,
    }


def _target_turn_comparison(item: dict[str, object]) -> dict[str, object]:
    old_response = str(item.get("old_assistant_response") or "")
    new_response = str(item.get("new_assistant_response") or "")
    exact_text_changed = old_response.strip() != new_response.strip()
    if old_response or new_response:
        approximate_similarity = round(
            difflib.SequenceMatcher(None, old_response.strip(), new_response.strip()).ratio(),
            3,
        )
    else:
        approximate_similarity = 1.0
    if not exact_text_changed:
        behavior_observation = "No text change was observed in the rerun response."
    elif approximate_similarity >= 0.82:
        behavior_observation = (
            "The rerun response changed wording but is highly similar; evaluate whether the required action changed."
        )
    else:
        behavior_observation = "The rerun response changed substantially; evaluate whether it now follows the target rule."
    return {
        "user_turn_index": item.get("user_turn_index"),
        "assistant_turn_index": item.get("assistant_turn_index"),
        "user_message": item.get("user_message"),
        "old_assistant_response": old_response,
        "new_assistant_response": new_response,
        "exact_text_changed": exact_text_changed,
        "approximate_similarity": approximate_similarity,
        "behavior_observation": behavior_observation,
        "error": item.get("error"),
    }


def _prompt_change_quality(before_prompt: str, optimized_prompt: str) -> dict[str, object]:
    before_counts = _line_counts(before_prompt)
    optimized_counts = _line_counts(optimized_prompt)
    added_lines = [
        line[1:].strip()
        for line in difflib.unified_diff(
            before_prompt.splitlines(),
            optimized_prompt.splitlines(),
            fromfile="before_prompt",
            tofile="optimized_prompt",
            lineterm="",
        )
        if line.startswith("+") and not line.startswith("+++")
    ]
    repeated_added_lines = [
        line
        for line in added_lines
        if len(line) >= 24 and optimized_counts.get(line, 0) > max(before_counts.get(line, 0), 1)
    ]
    repeated_added_fragments = [
        fragment
        for line in added_lines
        for fragment in _repeated_sentence_fragments(line)
    ]
    return {
        "added_lines": added_lines[:12],
        "repeated_added_lines": repeated_added_lines[:8],
        "repeated_added_fragments": repeated_added_fragments[:8],
        "has_repeated_added_instruction": bool(repeated_added_lines or repeated_added_fragments),
        "style_violation": _prompt_edit_style_violation(before_prompt, optimized_prompt),
    }


def _prompt_repair_saturation_diagnostics(
    prompt_version_history: list[dict[str, Any]] | None,
    post_rerun_bad_cases: list[BadCase],
    post_rerun_scan_status: str,
) -> dict[str, object]:
    versions = [version for version in (prompt_version_history or []) if isinstance(version, dict)]
    residual_count = len(post_rerun_bad_cases)
    applied_version_count = max(0, len(versions) - 1)
    scan_trace_items: list[dict[str, object]] = []
    for version in versions:
        label = str(version.get("label") or "")
        scan_cases = version.get("scan_trace_cases") or []
        if not isinstance(scan_cases, list):
            continue
        for item in scan_cases:
            if isinstance(item, dict):
                scan_trace_items.append(
                    {
                        "version_label": label,
                        "turn_index": item.get("turn_index"),
                        "error_type": item.get("error_type"),
                    }
                )
    residual_error_types = sorted({case.error_type for case in post_rerun_bad_cases})
    residual_turn_error_keys = {
        (case.turn_index, case.error_type)
        for case in post_rerun_bad_cases
    }
    repeated_error_type_counts = {
        error_type: 1
        + sum(1 for item in scan_trace_items if item.get("error_type") == error_type)
        for error_type in residual_error_types
    }
    repeated_turn_error_counts = {
        f"{turn_index}:{error_type}": 1
        + sum(
            1
            for item in scan_trace_items
            if item.get("turn_index") == turn_index and item.get("error_type") == error_type
        )
        for turn_index, error_type in residual_turn_error_keys
    }
    unfixable_assessment_required = (
        post_rerun_scan_status == "completed"
        and residual_count > 0
        and applied_version_count >= 2
    )
    repeated_same_failure = any(count >= 3 for count in repeated_error_type_counts.values()) or any(
        count >= 2 for count in repeated_turn_error_counts.values()
    )
    likely_not_fixable_by_prompt_only = unfixable_assessment_required and repeated_same_failure
    if residual_count == 0:
        status = "resolved"
    elif likely_not_fixable_by_prompt_only:
        status = "likely_not_fixable_by_prompt_only"
    elif unfixable_assessment_required:
        status = "unfixable_assessment_required"
    else:
        status = "prompt_repair_still_plausible"
    return {
        "status": status,
        "prompt_version_count": len(versions),
        "applied_version_count": applied_version_count,
        "scanned_version_count": sum(1 for version in versions if version.get("scan_trace_recorded")),
        "residual_badcase_count": residual_count,
        "residual_error_types": residual_error_types,
        "repeated_residual_error_type_counts": repeated_error_type_counts,
        "repeated_residual_turn_error_counts": repeated_turn_error_counts,
        "unfixable_assessment_required": unfixable_assessment_required,
        "likely_not_fixable_by_prompt_only": likely_not_fixable_by_prompt_only,
        "recommended_next_action": (
            "Stop repeating prompt-only patches for this residual case. Assess whether required ground truth, "
            "tool behavior, deterministic replacement, or model/backend escalation is needed."
            if likely_not_fixable_by_prompt_only
            else "Continue only with a materially different repair plan tied to the residual evidence."
        ),
    }


def _line_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        key = line.strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _repeated_sentence_fragments(text: str) -> list[str]:
    parts = [
        re.sub(r"\s+", " ", part).strip(" .;:")
        for part in re.split(r"(?<=[.!?])\s+|;\s+", text)
    ]
    counts: dict[str, int] = {}
    for part in parts:
        if len(part) < 24:
            continue
        lowered = part.lower()
        counts[lowered] = counts.get(lowered, 0) + 1
    repeated = []
    for part in parts:
        lowered = part.lower()
        if counts.get(lowered, 0) > 1 and part not in repeated:
            repeated.append(part)
    return repeated


def _conclusion_root_cause_hints(
    target_turn_comparisons: list[dict[str, object]],
    prompt_change_quality: dict[str, object],
    token_probability: object,
    post_rerun_bad_cases: list[BadCase],
    post_rerun_scan_status: str,
    repair_saturation: dict[str, object] | None = None,
) -> list[str]:
    hints: list[str] = []
    if prompt_change_quality.get("has_repeated_added_instruction"):
        hints.append("The prompt edit appears to duplicate an abstract instruction instead of adding a concrete action pattern.")
    if any(item.get("exact_text_changed") and float(item.get("approximate_similarity") or 0) >= 0.82 for item in target_turn_comparisons):
        hints.append("The rerun changed wording but remained behaviorally similar, so the root cause may be instruction weakness.")
    if isinstance(token_probability, dict) and not token_probability.get("available"):
        hints.append("No token-probability evidence is available; do not attribute the failure to token-level uncertainty.")
    if post_rerun_scan_status == "completed" and post_rerun_bad_cases:
        turns = ", ".join(str(case.turn_index) for case in post_rerun_bad_cases[:6])
        hints.append(f"Post-rerun scan still found residual badcase turns: {turns}.")
    if repair_saturation and repair_saturation.get("likely_not_fixable_by_prompt_only"):
        hints.append(
            "The same residual failure persisted across multiple prompt versions; assess whether prompt-only repair is saturated and whether the case should be handled as unsupported, deterministic replacement, workflow/tool correction, or model/backend escalation."
        )
    elif repair_saturation and repair_saturation.get("unfixable_assessment_required"):
        hints.append(
            "Multiple prompt versions have already been tried with residual failures; explicitly assess whether another prompt-only edit has new evidence or should stop."
        )
    return hints


def _rerun_result_dict(result: RerunTurn) -> dict[str, object]:
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    return {
        "user_turn_index": getattr(result, "user_turn_index", None),
        "assistant_turn_index": getattr(result, "assistant_turn_index", None),
        "user_message": getattr(result, "user_message", None),
        "old_assistant_response": getattr(result, "old_assistant_response", None),
        "new_assistant_response": getattr(result, "new_assistant_response", None),
        "error": getattr(result, "error", None),
        "response_diagnostics": getattr(result, "response_diagnostics", None),
    }


def _format_conclusion_value(value: object) -> str:
    if value is None:
        return "No conclusion was returned by the model."
    if isinstance(value, str):
        return _normalize_conclusion_paragraphs(value)
    if isinstance(value, list):
        paragraphs = [str(item).strip() for item in value if str(item).strip()]
        if not paragraphs:
            return "No conclusion was returned by the model."
        return _normalize_conclusion_paragraphs("\n\n".join(paragraphs))
    if isinstance(value, dict):
        paragraph_keys = [
            ("paragraph_1", "modifications", "previous_step_changes", "previous_optimization", "changes"),
            ("paragraph_2", "improvement_evidence", "improvement", "evaluation", "evidence"),
            ("paragraph_3", "next_steps", "next_optimization", "recommendation", "next_recommendation"),
        ]
        paragraphs = []
        used_keys = set()
        for key_group in paragraph_keys:
            for key in key_group:
                if key not in value:
                    continue
                formatted = _format_conclusion_section(value[key])
                if formatted:
                    paragraphs.append(formatted)
                    used_keys.add(key)
                    break
        if paragraphs:
            extra_items = [
                _format_conclusion_section(item)
                for key, item in value.items()
                if key not in used_keys and _format_conclusion_section(item)
            ]
            if extra_items:
                paragraphs.append(" ".join(extra_items))
            return _normalize_conclusion_paragraphs("\n\n".join(paragraphs))
        items = [_format_conclusion_section(item) for item in value.values()]
        items = [item for item in items if item]
        if items:
            return _normalize_conclusion_paragraphs("\n\n".join(items))
    return _normalize_conclusion_paragraphs(str(value))


def _format_conclusion_section(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        items = [
            f"{str(key).strip()}: {str(item).strip()}"
            for key, item in value.items()
            if str(key).strip() and str(item).strip()
        ]
        return " ".join(items)
    text = str(value).strip()
    return text


def _normalize_conclusion_paragraphs(value: str) -> str:
    text = value.strip()
    if not text:
        return "No conclusion was returned by the model."
    normalized_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            continue
        if re.match(r"^#{1,6}\s+", line):
            continue
        if re.match(r"^(paragraph\s*)?[123]\s*[:：.-]\s*$", line, flags=re.IGNORECASE):
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^(paragraph\s*)?[123]\s*[:：.)-]\s+", "", line, flags=re.IGNORECASE)
        if line:
            normalized_lines.append(line)
    cleaned = "\n".join(normalized_lines).strip()
    paragraphs = [
        re.sub(r"\s*\n\s*", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", cleaned)
        if paragraph.strip()
    ]
    if not paragraphs:
        return "No conclusion was returned by the model."
    if len(paragraphs) == 1:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraphs[0])
        if len(sentences) >= 3:
            paragraphs = [sentences[0], sentences[1], " ".join(sentences[2:])]
    if len(paragraphs) == 2:
        first_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraphs[0])
        second_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraphs[1])
        if len(first_sentences) >= 2:
            paragraphs = [first_sentences[0], " ".join(first_sentences[1:]), paragraphs[1]]
        elif len(second_sentences) >= 2:
            paragraphs = [paragraphs[0], second_sentences[0], " ".join(second_sentences[1:])]
    if len(paragraphs) > 3:
        paragraphs = [paragraphs[0], paragraphs[1], " ".join(paragraphs[2:])]
    return "\n\n".join(_trim_conclusion_to_word_budget(paragraphs, max_words=180))


def _trim_conclusion_to_word_budget(paragraphs: list[str], max_words: int) -> list[str]:
    trimmed: list[str] = []
    remaining = max_words
    for index, paragraph in enumerate(paragraphs):
        words = paragraph.split()
        future_paragraphs = max(len(paragraphs) - index - 1, 0)
        allowance = max(1, remaining - future_paragraphs)
        if len(words) > allowance:
            paragraph = " ".join(words[:allowance]).rstrip(".,;:") + "..."
            words = paragraph.split()
        trimmed.append(paragraph)
        remaining -= len(words)
        if remaining <= 0:
            break
    return trimmed


def _prompt_unified_diff(before_prompt: str, optimized_prompt: str, max_chars: int = 6000) -> str:
    diff = "\n".join(
        difflib.unified_diff(
            before_prompt.splitlines(),
            optimized_prompt.splitlines(),
            fromfile="before_prompt",
            tofile="optimized_prompt",
            lineterm="",
        )
    )
    if len(diff) <= max_chars:
        return diff
    return diff[:max_chars].rstrip() + "\n... [diff truncated]"


def _rerun_evidence_item(result: RerunTurn) -> dict[str, object]:
    old_response = result.old_assistant_response or ""
    new_response = result.new_assistant_response or ""
    response_diagnostics = getattr(result, "response_diagnostics", None)
    return {
        "user_turn_index": result.user_turn_index,
        "assistant_turn_index": result.assistant_turn_index,
        "user_message": result.user_message,
        "old_assistant_response": old_response,
        "new_assistant_response": new_response,
        "changed": old_response.strip() != new_response.strip(),
        "old_response_len": len(old_response),
        "new_response_len": len(new_response),
        "error": result.error,
        "response_diagnostics": response_diagnostics,
    }


def _experiment_diagnostic_evidence(
    data: ConversationData,
    rerun_results: list[RerunTurn],
    post_rerun_bad_cases: list[BadCase],
    post_rerun_scan_status: str,
) -> dict[str, object]:
    source_meta = data.source_meta or {}
    tools = data.tools or {}
    model_info = _compact_meta_value(source_meta.get("model_info"))
    kwargs = source_meta.get("kwargs") if isinstance(source_meta.get("kwargs"), dict) else {}
    rerun_items = [_rerun_evidence_item(result) for result in rerun_results]
    token_probability = _token_probability_diagnostics(rerun_results)
    response_metadata = _response_metadata_diagnostics(rerun_results)
    old_text = "\n".join(str(item.get("old_assistant_response") or "") for item in rerun_items)
    new_text = "\n".join(str(item.get("new_assistant_response") or "") for item in rerun_items)
    return {
        "behavior_change": {
            "rerun_turn_count": len(rerun_results),
            "changed_turn_count": sum(1 for item in rerun_items if item.get("changed")),
            "failed_turn_count": sum(1 for result in rerun_results if result.error),
            "old_response_function_call_count": _function_tag_count(old_text),
            "new_response_function_call_count": _function_tag_count(new_text),
            "old_response_contains_function_call": "<function-call" in old_text,
            "new_response_contains_function_call": "<function-call" in new_text,
        },
        "token_probability": token_probability,
        "response_metadata": response_metadata,
        "metadata": {
            "source_meta_available": bool(source_meta),
            "source_meta_keys": sorted(str(key) for key in source_meta.keys()),
            "model_info": model_info,
            "provider": source_meta.get("provider") or kwargs.get("provider"),
            "model": source_meta.get("model") or kwargs.get("model"),
            "url": source_meta.get("url") or kwargs.get("url"),
            "tools_available": bool(tools),
            "tool_count": len(tools),
            "tool_names": sorted(str(key) for key in tools.keys())[:20],
        },
        "post_rerun_scan": {
            "status": post_rerun_scan_status,
            "residual_badcase_count": len(post_rerun_bad_cases),
            "residual_badcase_turns": [case.turn_index for case in post_rerun_bad_cases],
            "residual_error_types": sorted({case.error_type for case in post_rerun_bad_cases}),
        },
    }


def _compact_meta_value(value: object, max_chars: int = 1200) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) <= max_chars:
        return value
    return text[:max_chars].rstrip() + "... [truncated]"


def _function_tag_count(text: str) -> int:
    return len(re.findall(r"<function-call\b", text))


def _last_response_diagnostics(settings: LLMSettings) -> dict[str, Any] | None:
    if settings.backend != "company_api":
        return None
    diagnostics = get_last_company_response_diagnostics()
    return diagnostics or None


def _token_probability_diagnostics(rerun_results: list[RerunTurn]) -> dict[str, object]:
    logprob_items = []
    for result in rerun_results:
        diagnostics = getattr(result, "response_diagnostics", None) or {}
        logprobs = diagnostics.get("logprobs") if isinstance(diagnostics, dict) else None
        if isinstance(logprobs, dict):
            logprob_items.append(logprobs)
    available_items = [item for item in logprob_items if item.get("available")]
    if available_items:
        return {
            "available": True,
            "request_count": len(logprob_items),
            "available_count": len(available_items),
            "avg_logprob_by_turn": [item.get("avg_logprob") for item in available_items],
            "min_logprob_by_turn": [item.get("min_logprob") for item in available_items],
            "low_confidence_tokens": [
                token
                for item in available_items
                for token in (item.get("low_confidence_tokens") or [])
            ][:20],
        }
    if logprob_items:
        return {
            "available": False,
            "request_count": len(logprob_items),
            "reasons": [item.get("reason") for item in logprob_items if item.get("reason")],
            "unsupported_errors": [
                item.get("unsupported_error")
                for item in logprob_items
                if item.get("unsupported_error")
            ],
        }
    return {
        "available": False,
        "request_count": 0,
        "reason": (
            "No token logprob diagnostics were attached to rerun results. "
            "Do not attribute improvement or failure to token error probability."
        ),
    }


def _response_metadata_diagnostics(rerun_results: list[RerunTurn]) -> dict[str, object]:
    metadata_items = []
    for result in rerun_results:
        diagnostics = getattr(result, "response_diagnostics", None) or {}
        if not isinstance(diagnostics, dict):
            continue
        item = {
            "request_id": diagnostics.get("request_id"),
            "purpose": diagnostics.get("purpose"),
            "provider": diagnostics.get("provider"),
            "model": diagnostics.get("model"),
            "endpoint": diagnostics.get("endpoint"),
        }
        response_meta = diagnostics.get("response_meta")
        if isinstance(response_meta, dict):
            item["response_meta"] = response_meta
        metadata_items.append(item)
    return {
        "available": bool(metadata_items),
        "request_count": len(metadata_items),
        "items": metadata_items[:12],
    }


def recommend_for_manual_bad_case(
    data: ConversationData,
    turn_index: int,
    feedback: str,
    llm_settings: LLMSettings | None = None,
) -> str:
    settings = llm_settings or LLMSettings()
    return _manual_recommendation(data, turn_index, feedback, settings)


def _recommendation_from_bad_case(bad_case: BadCase) -> str:
    evidence = bad_case.evidence.strip()
    if evidence:
        return (
            "Revise the relevant existing numbered Step/substep/branch in place so this failure "
            f"does not happen again: {evidence}"
        )
    return (
        "Revise the relevant existing numbered Step/substep/branch in place to address this "
        f"{bad_case.error_type} bad case."
    )


def _has_step_append_violation(original_prompt: str, optimized_prompt: str) -> bool:
    forbidden_sections = (
        "additional reliability requirements",
        "step update note",
        "addendum",
        "appendix",
        "patch note",
    )
    original_lower = original_prompt.lower()
    optimized_lower = optimized_prompt.lower()
    return any(
        section in optimized_lower and section not in original_lower
        for section in forbidden_sections
    )


def _prompt_patch_relevance_violation(
    original_prompt: str,
    optimized_prompt: str,
    bad_case: Any,
) -> str | None:
    error_type = _bad_case_error_type(bad_case)
    changed_text = _changed_prompt_text(original_prompt, optimized_prompt).lower()
    if error_type == "escalation_action_not_followed" and _bad_case_mentions_exact_message(bad_case):
        required_markers = (
            "exact",
            "message exactly",
            "hotline",
            "required message",
            "spoken text",
            "configured escalation action",
        )
        if any(marker in changed_text for marker in required_markers):
            return None
        return "patch did not edit the exact escalation/action conflict"
    if error_type != "late_payment_proposal_not_rtp_closing":
        return None
    required_markers = (
        "rtp_closing",
        "maximum payment date",
        "payment date",
        "date validation",
        "later than",
        "after the maximum",
        "deadline",
        "do not negotiate",
        "not negotiate",
        "late-date",
        "late date",
    )
    if any(marker in changed_text for marker in required_markers):
        return None
    return "patch did not edit the late-date/RTP_Closing rule"


def _bad_case_error_type(bad_case: Any) -> str:
    if isinstance(bad_case, BadCase):
        return bad_case.error_type
    if isinstance(bad_case, dict):
        return str(bad_case.get("error_type") or "")
    return ""


def _bad_case_mentions_exact_message(bad_case: Any) -> bool:
    if isinstance(bad_case, BadCase):
        text = " ".join((bad_case.error_type, bad_case.evidence, bad_case.recommendation))
    elif isinstance(bad_case, dict):
        text = " ".join(
            str(bad_case.get(key) or "")
            for key in ("error_type", "evidence", "recommendation")
        )
    else:
        text = str(bad_case or "")
    lowered = text.lower()
    return "exact" in lowered and (
        "message" in lowered
        or "spoken text" in lowered
        or "escalation action" in lowered
        or "hotline" in lowered
    )


def _changed_prompt_text(original_prompt: str, optimized_prompt: str) -> str:
    changed_lines = []
    for line in difflib.ndiff(original_prompt.splitlines(), optimized_prompt.splitlines()):
        if line.startswith("+ ") or line.startswith("- "):
            changed_lines.append(line[2:])
    return "\n".join(changed_lines)


def _deterministic_prompt_patch(
    current_system_prompt: str,
    bad_case: BadCase,
    force_prompt_edit: bool = False,
) -> PromptOptimization | None:
    if bad_case.error_type == "escalation_action_not_followed":
        return _deterministic_exact_escalation_prompt_patch(current_system_prompt, bad_case)
    if bad_case.error_type == "late_payment_proposal_not_rtp_closing":
        return _deterministic_late_date_prompt_patch(
            current_system_prompt,
            force_prompt_edit=force_prompt_edit,
        )
    return None


def _deterministic_late_date_prompt_patch(
    current_system_prompt: str,
    force_prompt_edit: bool = False,
) -> PromptOptimization | None:
    if "Rule conflict resolver for late-date proposals:" in current_system_prompt:
        if force_prompt_edit:
            return _reinforce_existing_late_date_resolver(current_system_prompt)
        return None
    date_section = _find_late_date_validation_section(current_system_prompt)
    if date_section is None:
        return None
    proposal_section = _find_proposal_collection_section(current_system_prompt)
    closing_section = _find_rtp_closing_section(current_system_prompt)
    insertions: list[tuple[int, str]] = []
    insertions.append((date_section.end, _late_date_validation_conflict_guard()))
    if proposal_section is not None:
        insertions.append((proposal_section.end, _proposal_clarification_conflict_guard()))
    if closing_section is not None:
        insertions.append((closing_section.end, _rtp_closing_conflict_guard()))
    patched_prompt = _apply_prompt_insertions(current_system_prompt, insertions)
    return PromptOptimization(
        optimized_prompt=patched_prompt,
        rationale=(
            "Applied deterministic local rule-conflict patch to the existing late-date, proposal "
            "clarification, and RTP_Closing branches using only values already configured in the "
            "current prompt."
        ),
        applied_feedback_summary=(
            "Added generic rule-conflict guards so late-date routing to RTP_Closing takes priority "
            "over proposal clarification, negotiation, and commitment questions."
        ),
    )


def _reinforce_existing_late_date_resolver(current_system_prompt: str) -> PromptOptimization | None:
    marker = "Residual-verification reinforcement for late-date proposals:"
    if marker in current_system_prompt:
        return None
    resolver = _find_named_rule_block(
        current_system_prompt,
        "Rule conflict resolver for late-date proposals:",
    )
    if resolver is None:
        return None
    reinforcement = (
        "\n\nResidual-verification reinforcement for late-date proposals:\n"
        "- A late-date proposal is a deterministic terminal-routing condition. Once detected, do not "
        "generate any content from proposal collection, negotiation, persuasion, deadline-adjustment, "
        "or payment-commitment flows.\n"
        "- The response may only acknowledge the user's stated timing, state that the late proposal "
        "cannot be accepted, deliver the configured RTP_Closing assertions, and end with `<dialog-end>`.\n"
        "- Forbidden outputs include suggesting payment before the deadline, asking the user to try, "
        "asking whether payment is possible, offering another date, or ending with any question."
    )
    patched_prompt = (
        current_system_prompt[: resolver.end]
        + reinforcement
        + current_system_prompt[resolver.end :]
    )
    return PromptOptimization(
        optimized_prompt=patched_prompt,
        rationale=(
            "Strengthened the existing late-date resolver after an unchanged or failed-verification "
            "apply attempt by adding deterministic allowed and forbidden output constraints."
        ),
        applied_feedback_summary=(
            "Reinforced the existing late-date resolver with a deterministic terminal-response contract."
        ),
    )


def _find_named_rule_block(system_prompt: str, heading: str) -> _PromptSection | None:
    start = system_prompt.find(heading)
    if start < 0:
        return None
    next_block = re.search(r"\n\n[A-Z][^\n]{3,120}:\n", system_prompt[start + len(heading) :])
    end = (
        start + len(heading) + next_block.start()
        if next_block
        else len(system_prompt)
    )
    return _PromptSection(start=start, end=end)


@dataclass(frozen=True)
class _PromptSection:
    start: int
    end: int


def _find_late_date_validation_section(system_prompt: str) -> _PromptSection | None:
    heading_match = re.search(
        r"\*\*[^*\n]*(?:maximum payment date validation|date validation)[^*\n]*\*\*:?",
        system_prompt,
        flags=re.IGNORECASE,
    )
    if heading_match:
        next_heading = re.search(
            r"\n(?:\*\*\d+(?:\.\d+)*[^*\n]*\*\*:?|#{1,6}\s+\S)",
            system_prompt[heading_match.end() :],
            flags=re.IGNORECASE,
        )
        end = heading_match.end() + next_heading.start() if next_heading else len(system_prompt)
        section_text = system_prompt[heading_match.start() : end].lower()
        if "rtp_closing" in section_text and (
            "later than" in section_text
            or "maximum payment date" in section_text
            or ">" in section_text
        ):
            return _PromptSection(heading_match.start(), end)

    branch_match = re.search(
        r"if\s+the\s+user\s+proposes\s+a\s+payment\s+date.{0,1800}?rtp_closing.{0,800}?(?=\n\n|\Z)",
        system_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if branch_match:
        return _PromptSection(branch_match.start(), branch_match.end())
    return None


def _find_proposal_collection_section(system_prompt: str) -> _PromptSection | None:
    heading_pattern = re.compile(
        r"(?m)^(?:\*\*[^*\n]*(?:clarify|collect|proposal|payment proposal)[^*\n]*\*\*:?|"
        r"#{1,6}\s+[^\n]*(?:clarify|collect|proposal)[^\n]*)",
        flags=re.IGNORECASE,
    )
    for heading_match in heading_pattern.finditer(system_prompt):
        next_heading = re.search(
            r"\n(?:\*\*\d+(?:\.\d+)*[^*\n]*\*\*:?|#{1,6}\s+\S)",
            system_prompt[heading_match.end() :],
            flags=re.IGNORECASE,
        )
        end = heading_match.end() + next_heading.start() if next_heading else len(system_prompt)
        section_text = system_prompt[heading_match.start() : end].lower()
        if _looks_like_proposal_clarification_section(section_text):
            return _PromptSection(heading_match.start(), end)

    branch_match = re.search(
        r"(?:if\s+the\s+user\s+[^.\n]{0,260}?(?:does\s+not\s+mention\s+a\s+date|vague\s+date)"
        r"[^.\n]{0,260}?(?:ask|prompt)[^.\n]{0,120}?date)",
        system_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if branch_match:
        return _PromptSection(branch_match.start(), branch_match.end())
    return None


def _find_rtp_closing_section(system_prompt: str) -> _PromptSection | None:
    heading_match = re.search(
        r"(?m)^(?:\*\*[^*\n]*rtp_closing[^*\n]*\*\*:?|#{1,6}\s+[^\n]*rtp_closing[^\n]*)",
        system_prompt,
        flags=re.IGNORECASE,
    )
    if not heading_match:
        return None
    next_heading = re.search(
        r"\n(?:\*\*\d+(?:\.\d+)*[^*\n]*\*\*:?|#{1,6}\s+\S)",
        system_prompt[heading_match.end() :],
        flags=re.IGNORECASE,
    )
    end = heading_match.end() + next_heading.start() if next_heading else len(system_prompt)
    return _PromptSection(heading_match.start(), end)


def _looks_like_proposal_clarification_section(section_text: str) -> bool:
    has_proposal_language = any(
        term in section_text
        for term in (
            "proposal",
            "payment date",
            "clarify",
            "collect",
            "ask the user when",
            "prompt the user for a date",
        )
    )
    has_date_language = "date" in section_text or "tanggal" in section_text
    asks_for_more = any(term in section_text for term in ("ask", "prompt", "clarify", "collect"))
    return has_proposal_language and has_date_language and asks_for_more


def _late_date_validation_conflict_guard() -> str:
    return (
        "\n\nRule conflict resolver for late-date proposals:\n"
        "- This date-validation branch has priority over proposal clarification, negotiation, "
        "validation-tool calls, and payment-commitment questions whenever the user's timing is "
        "later than the configured maximum payment date.\n"
        "- Treat any user payment timing that can be resolved to later than the configured maximum "
        "payment date as terminal for negotiation, including relative or natural-language timing "
        "resolved from the prompt's reference date.\n"
        "- The next assistant response must follow the configured RTP_Closing state only. Use the "
        "amount, deadline, consequences, payment channels, and closing wording already defined in "
        "the current system prompt and RTP_Closing section.\n"
        "- Do not negotiate, ask for another date, ask for commitment or confirmation, call validation "
        "tools, or include any follow-up question before closing. The response must be a final closing "
        "response with `<dialog-end>` and must not contain a question mark or interrogative sentence."
    )


def _proposal_clarification_conflict_guard() -> str:
    return (
        "\n\nRule conflict resolver before proposal clarification:\n"
        "- Before this section asks for missing, vague, relative, or more specific payment-date details, "
        "first apply any higher-priority date-validation rule in the current system prompt.\n"
        "- If the user's timing already gives enough information to determine that it is later than "
        "the configured maximum payment date, do not clarify, negotiate, ask for commitment, or ask "
        "whether the user can pay by the configured maximum date.\n"
        "- In that conflict, skip this clarification flow and route the next assistant response directly "
        "to the configured RTP_Closing state as a final `<dialog-end>` closing response."
    )


def _rtp_closing_conflict_guard() -> str:
    return (
        "\n\nRule conflict resolver inside RTP_Closing:\n"
        "- When RTP_Closing is reached from a late-date, maximum-date, no-agreement, or failed-negotiation "
        "route, the response is terminal and must be a closing statement, not a new negotiation turn.\n"
        "- State the configured payment requirement and consequences as assertions using the values already "
        "defined in this prompt. Do not ask whether the customer can commit, can try, can consider, or can "
        "make payment by the deadline.\n"
        "- End with the configured polite closing and `<dialog-end>`. Do not include any question mark or "
        "interrogative sentence."
    )


def _apply_prompt_insertions(prompt: str, insertions: list[tuple[int, str]]) -> str:
    patched = prompt
    for position, text in sorted(insertions, key=lambda item: item[0], reverse=True):
        if text.strip() in patched:
            continue
        patched = patched[:position] + text + patched[position:]
    return patched


def _deterministic_exact_escalation_prompt_patch(
    current_system_prompt: str,
    bad_case: BadCase,
) -> PromptOptimization | None:
    if not _bad_case_mentions_exact_message(bad_case):
        return None
    exact_message = extract_exact_escalation_message(current_system_prompt)
    if not exact_message:
        return None
    if "Operational guard for exact escalation actions:" in current_system_prompt:
        return None
    match = re.search(
        r"When an escalation is triggered, you must \*\*immediately stop\*\*.*?"
        r"deliver the following message exactly [\"“].*?[\"”]\.",
        current_system_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    action_text = match.group(0)
    guard = (
        "\n\nOperational guard for exact escalation actions:\n"
        "- This escalation action has the highest priority over negotiation, payment confirmation, "
        "ordinary call closure, and transfer branches.\n"
        "- If the latest user message matches any configured escalation trigger, including a debt "
        "dispute or already-paid claim when those triggers are defined, execute this exact escalation "
        "action immediately.\n"
        "- The next assistant response must be exactly: "
        f"\"{exact_message}\"\n"
        "- Do not ask for additional details, proof, confirmation, dates, payment method, or whether "
        "the user wants to speak with an agent. Do not continue negotiation. "
        "Do not call a transfer tool unless the exact message itself requires a tool call."
    )
    patched_prompt = current_system_prompt.replace(action_text, action_text + guard, 1)
    already_paid_anchor = (
        "    - If the user informs that they have already paid, thank the user for the payment "
        "and end the call <dialog-end>."
    )
    if already_paid_anchor in patched_prompt:
        patched_prompt = patched_prompt.replace(
            already_paid_anchor,
            (
                "    - If the user informs that they have already paid, first check the configured "
                "Escalation Triggers. If the current prompt classifies the statement as an escalation "
                "trigger, execute the configured Escalation Action exactly. Only use the thank-and-end "
                "behavior when the statement is not an escalation trigger."
            ),
            1,
        )
    return PromptOptimization(
        optimized_prompt=patched_prompt,
        rationale=(
            "Applied deterministic local patch to make exact escalation action highest priority "
            "and resolve the already-paid branch conflict."
        ),
        applied_feedback_summary=(
            "Added an operational guard for exact escalation actions and clarified that already-paid "
            "claims route to the configured escalation action."
        ),
    )


def extract_exact_escalation_message(system_prompt: str) -> str | None:
    match = re.search(
        r"deliver the following message exactly\s*[:：]?\s*[\"“](?P<message>.*?)[\"”]",
        system_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    message = re.sub(r"\s+", " ", match.group("message")).strip()
    return message or None


def _prompt_already_addresses_case(current_system_prompt: str, bad_case: BadCase) -> str | None:
    if bad_case.error_type == "escalation_action_not_followed":
        if "Operational guard for exact escalation actions:" in current_system_prompt:
            return "Existing working prompt already contains the exact-escalation operational guard."
    if bad_case.error_type == "late_payment_proposal_not_rtp_closing":
        if "Rule conflict resolver for late-date proposals:" in current_system_prompt:
            return "Existing working prompt already contains the late-date rule-conflict resolver."
    return None


def _prompt_edit_integrity_violation(original_prompt: str, optimized_prompt: str) -> str | None:
    if not optimized_prompt.strip():
        return "returned an empty prompt"

    original_length = len(original_prompt)
    optimized_length = len(optimized_prompt)
    if original_length >= 4000 and optimized_length < int(original_length * 0.85):
        return (
            "returned a substantially shorter prompt "
            f"({optimized_length} vs {original_length} characters)"
        )

    original_headings = _prompt_structure_headings(original_prompt)
    if len(original_headings) >= 8:
        missing_headings = [heading for heading in original_headings if heading not in optimized_prompt]
        allowed_missing = max(2, len(original_headings) // 10)
        if len(missing_headings) > allowed_missing:
            return f"dropped existing prompt headings such as {missing_headings[0]!r}"

    return None


def _prompt_structure_headings(prompt: str) -> list[str]:
    headings: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+\S", stripped):
            headings.append(stripped)
        elif re.match(r"^\*\*\d+(?:\.\d+)*[^\n]*\*\*", stripped):
            headings.append(stripped)
    return headings


def _prompt_edit_style_violation(original_prompt: str, optimized_prompt: str) -> str | None:
    original_lines = {line.strip() for line in original_prompt.splitlines() if line.strip()}
    for line in optimized_prompt.splitlines():
        stripped = line.strip()
        if not stripped or stripped in original_lines:
            continue
        if len(stripped) > PROMPT_EDIT_MAX_NEW_LINE_CHARS and _looks_like_branch_paragraph(stripped):
            return "introduced an overlong workflow branch paragraph instead of concise substeps"

    repeated_delta = _max_repeated_shingle_delta(original_prompt, optimized_prompt)
    if repeated_delta > PROMPT_EDIT_MAX_REPEATED_SHINGLE_DELTA:
        return "introduced repeated branch/fallback wording"
    return None


def _looks_like_branch_paragraph(line: str) -> bool:
    lowered = line.lower()
    branch_markers = (
        " if ",
        " otherwise",
        " alternatively",
        " no matter what",
        " not sure",
        " unsure",
        " refuses",
        " refusal",
        " follow up",
    )
    return sum(1 for marker in branch_markers if marker in f" {lowered}") >= 2


def _max_repeated_shingle_delta(original_prompt: str, optimized_prompt: str, size: int = 7) -> int:
    original_counts = _shingle_counts(original_prompt, size)
    optimized_counts = _shingle_counts(optimized_prompt, size)
    max_delta = 0
    for shingle, optimized_count in optimized_counts.items():
        if _uninformative_shingle(shingle):
            continue
        delta = optimized_count - original_counts.get(shingle, 0)
        if delta > max_delta:
            max_delta = delta
    return max_delta


def _shingle_counts(text: str, size: int) -> dict[tuple[str, ...], int]:
    tokens = re.findall(r"[A-Za-z0-9_']+", text.lower())
    counts: dict[tuple[str, ...], int] = {}
    for index in range(0, max(0, len(tokens) - size + 1)):
        shingle = tuple(tokens[index:index + size])
        counts[shingle] = counts.get(shingle, 0) + 1
    return counts


def _uninformative_shingle(shingle: tuple[str, ...]) -> bool:
    stopwords = {
        "the", "and", "or", "to", "a", "an", "of", "in", "on", "for", "with", "if", "is", "are",
    }
    return all(token in stopwords or len(token) <= 2 for token in shingle)


def _retry_step_preserving_prompt_edit(
    settings: LLMSettings,
    current_system_prompt: str,
    bad_case: BadCase,
    related_turn: dict[str, Any] | None,
    rejected_prompt: str,
    rejection_reason: str = "appended a separate note/requirements section",
) -> str:
    retry_prompt = (
        f"The previous edit was rejected because it {rejection_reason}. "
        "Rewrite again. You must preserve the original step-by-step format and modify the existing "
        "numbered Step/substep text in place. Do not add any new section at the end. If the failure "
        "comes from an ambiguous, uncertain, partial, refused, or unrelated user reply, add the "
        "clarification/fallback condition inside the relevant existing step instead of creating a "
        f"generic rule. {PROMPT_EDIT_STYLE_RULES} Return JSON only with key optimized_prompt."
    )
    payload = {
        "current_system_prompt": current_system_prompt,
        "bad_case": bad_case.__dict__,
        "related_turn": related_turn,
        "recommendation_to_apply": bad_case.recommendation.strip() or _recommendation_from_bad_case(bad_case),
        "rejected_prompt": rejected_prompt,
    }
    content = _chat_json(
        settings=settings,
        messages=[
            {"role": "system", "content": retry_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        purpose="prompt_edit_retry",
    )
    parsed = json.loads(content)
    retried_prompt = str(parsed.get("optimized_prompt") or current_system_prompt)
    if (
        _prompt_edit_integrity_violation(current_system_prompt, retried_prompt)
        or
        _has_step_append_violation(current_system_prompt, retried_prompt)
        or _prompt_edit_style_violation(current_system_prompt, retried_prompt)
    ):
        return current_system_prompt
    return retried_prompt


def _manual_recommendation(data: ConversationData, turn_index: int, feedback: str, settings: LLMSettings) -> str:
    if not _has_llm_access(settings):
        return f"Add an explicit instruction addressing this human-reported issue: {feedback}"
    turn = data.interactions[turn_index] if 0 <= turn_index < len(data.interactions) else Interaction(role="assistant", content="")
    prompt = (
        "Convert this human bad-case note into one concise system-prompt modification recommendation. "
        "Prefer recommendations that edit the relevant existing numbered Step/substep/branch in place, "
        "rather than appending a new generic rule. If the note involves vague, uncertain, partial, refused, "
        "or unrelated user replies, recommend a branch that keeps the assistant in the current workflow "
        "step, asks one focused clarification question, or uses the allowed fallback. Return only the "
        "recommendation sentence."
    )
    try:
        return _chat_text(
            settings=settings,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "system_prompt": data.system_prompt,
                            "turn_index": turn_index,
                            "turn": turn.model_dump(exclude_none=True),
                            "human_feedback": feedback,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.1,
            purpose="manual_recommendation",
        )
    except (OpenAIError, requests.RequestException, RuntimeError):
        return f"Add an explicit instruction addressing this human-reported issue: {feedback}"


def _chat_json(settings: LLMSettings, messages: list[dict[str, str]], purpose: str = "json") -> str:
    if settings.backend == "company_api":
        text = _chat_text(settings=settings, messages=messages, temperature=0.1, purpose=purpose)
        try:
            return _extract_json_object(text)
        except json.JSONDecodeError as first_exc:
            repair = _repair_json_response(text)
            if repair is not None:
                _log_json_repair_attempt(
                    purpose=purpose,
                    model=settings.model,
                    provider=settings.provider,
                    stage="initial",
                    error=str(first_exc),
                    repair=repair,
                )
                return repair.repaired_json
            retry_text = _chat_text(
                settings=settings,
                messages=_json_retry_messages(messages, text),
                temperature=0.0,
                purpose=f"{purpose}_json_retry",
            )
            try:
                return _extract_json_object(retry_text)
            except json.JSONDecodeError as retry_exc:
                retry_repair = _repair_json_response(retry_text)
                if retry_repair is not None:
                    _log_json_repair_attempt(
                        purpose=purpose,
                        model=settings.model,
                        provider=settings.provider,
                        stage="retry",
                        error=str(retry_exc),
                        repair=retry_repair,
                    )
                    return retry_repair.repaired_json
                _log_json_repair_failure(
                    purpose=purpose,
                    model=settings.model,
                    provider=settings.provider,
                    initial_text=text,
                    retry_text=retry_text,
                    initial_error=str(first_exc),
                    retry_error=str(retry_exc),
                )
                raise
    client = OpenAI()
    response = client.chat.completions.create(
        model=settings.model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


def _json_retry_messages(
    original_messages: list[dict[str, str]],
    invalid_response: str,
) -> list[dict[str, str]]:
    retry_prompt = (
        "The previous response was not valid JSON. Re-run the original task and return exactly "
        "one valid JSON object, with no markdown fences, no prose before or after it, and no "
        "comments. Preserve the schema requested by the original system message. If the original "
        "task asked for a full prompt field, include that field as a JSON string."
    )
    payload = {
        "original_messages": original_messages,
        "invalid_response": invalid_response[:4000],
    }
    return [
        {"role": "system", "content": retry_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _repair_json_response(text: str) -> JsonRepairResult | None:
    for method, candidate in _json_repair_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return JsonRepairResult(
                repaired_json=json.dumps(parsed, ensure_ascii=False),
                method=method,
                original_text=text,
                repaired_text=candidate,
            )
    return None


def _json_repair_candidates(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    stripped = _strip_markdown_json_fence(text).strip()
    if stripped and stripped != text:
        candidates.append(("strip_markdown_fence", stripped))

    object_candidate = _best_effort_json_object_substring(stripped or text)
    if object_candidate:
        candidates.append(("object_substring", object_candidate))
        escaped = _escape_json_string_control_chars(object_candidate)
        if escaped != object_candidate:
            candidates.append(("object_substring_escape_control_chars", escaped))
        completed = _complete_truncated_json_object(escaped)
        if completed != escaped:
            candidates.append(("complete_truncated_object", completed))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for method, candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append((method, candidate))
    return unique


def _strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _best_effort_json_object_substring(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    end = _last_balanced_json_object_end(text, start)
    if end is not None:
        return text[start:end]
    return text[start:].strip()


def _last_balanced_json_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    last_end: int | None = None
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if in_string and char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                last_end = index + 1
    return last_end


def _complete_truncated_json_object(text: str) -> str:
    candidate = text.strip()
    pieces = list(candidate)
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in candidate:
        if escaped:
            escaped = False
            continue
        if in_string and char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
            continue
        if char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
            continue
    if in_string:
        pieces.append('"')
    while stack:
        pieces.append(stack.pop())
    return "".join(pieces)


def _log_json_repair_attempt(
    *,
    purpose: str,
    model: str,
    provider: str,
    stage: str,
    error: str,
    repair: JsonRepairResult,
) -> None:
    _append_json_repair_log(
        {
            "event": "json_repair_success",
            "purpose": purpose,
            "model": model,
            "provider": provider,
            "stage": stage,
            "error": error,
            "method": repair.method,
            "original_text": repair.original_text,
            "repaired_text": repair.repaired_text,
            "repaired_json": repair.repaired_json,
        }
    )


def _log_json_repair_failure(
    *,
    purpose: str,
    model: str,
    provider: str,
    initial_text: str,
    retry_text: str,
    initial_error: str,
    retry_error: str,
) -> None:
    _append_json_repair_log(
        {
            "event": "json_repair_failure",
            "purpose": purpose,
            "model": model,
            "provider": provider,
            "initial_error": initial_error,
            "retry_error": retry_error,
            "initial_text": initial_text,
            "retry_text": retry_text,
        }
    )


def _append_json_repair_log(payload: dict[str, Any]) -> None:
    log_path = PROJECT_ROOT / "logs" / "json_repair_attempts.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": _utc_timestamp(), **payload}, ensure_ascii=False) + "\n")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chat_text(
    settings: LLMSettings,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    tools: dict[str, Any] | None = None,
    tool_choice: dict[str, Any] | str | None = None,
    purpose: str = "chat",
    expected_system_prompt_hash: str | None = None,
    expected_prompt_version: str | None = None,
) -> str:
    if settings.backend == "company_api":
        return _company_chat_text(
            settings=settings,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            purpose=purpose,
            expected_system_prompt_hash=expected_system_prompt_hash,
            expected_prompt_version=expected_prompt_version,
        )
    client = OpenAI()
    request: dict[str, Any] = {
        "model": settings.model,
        "messages": _openai_compatible_messages(messages),
        "temperature": temperature,
    }
    if tools:
        request["tools"] = list(tools.values())
        request["tool_choice"] = tool_choice or "auto"
    response = client.chat.completions.create(**request)
    return _openai_message_to_text(response.choices[0].message)


def _openai_message_to_text(message: Any) -> str:
    content = getattr(message, "content", None)
    tool_call_content = render_tool_calls_as_function_wrappers(getattr(message, "tool_calls", None))
    pieces = [
        piece
        for piece in (content, tool_call_content)
        if isinstance(piece, str) and piece.strip()
    ]
    return "\n".join(pieces)


def _openai_compatible_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "")
        if role == "tool":
            normalized.append({"role": "user", "content": f"[tool output]\n{content}"})
            continue
        if role in {"system", "user", "assistant", "tool"}:
            normalized_message = {"role": role, "content": content}
            normalized.append(normalized_message)
    return normalized


def _company_chat_text(
    settings: LLMSettings,
    messages: list[dict[str, str]],
    temperature: float,
    tools: dict[str, Any] | None = None,
    tool_choice: dict[str, Any] | str | None = None,
    purpose: str = "chat",
    expected_system_prompt_hash: str | None = None,
    expected_prompt_version: str | None = None,
) -> str:
    return generate_with_company_demo(
        messages=_openai_compatible_messages(messages),
        model=settings.model,
        provider=settings.provider,
        url=settings.base_url,
        tools=tools,
        tool_choice=tool_choice,
        max_completion_tokens=settings.max_completion_tokens,
        temperature=temperature,
        purpose=purpose,
        expected_system_prompt_hash=expected_system_prompt_hash or settings.expected_system_prompt_hash,
        expected_prompt_version=expected_prompt_version or settings.expected_prompt_version,
    )


def _preflight_company_rerun_request(
    settings: LLMSettings,
    messages: list[dict[str, str]],
    tools: dict[str, Any] | None,
    tool_choice: dict[str, Any] | str | None,
    temperature: float,
    purpose: str,
    expected_system_prompt_hash: str | None,
    expected_prompt_version: str | None,
) -> None:
    if settings.backend != "company_api":
        return
    preflight_company_request(
        messages=_openai_compatible_messages(messages),
        model=settings.model,
        provider=settings.provider,
        url=settings.base_url,
        tools=tools,
        tool_choice=tool_choice,
        max_completion_tokens=settings.max_completion_tokens,
        temperature=temperature,
        purpose=purpose,
        expected_system_prompt_hash=expected_system_prompt_hash or settings.expected_system_prompt_hash,
        expected_prompt_version=expected_prompt_version or settings.expected_prompt_version,
    )


def _extract_response_content(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("content"), str):
            return data["content"]
        if isinstance(data.get("message"), dict) and isinstance(data["message"].get("content"), str):
            return data["message"]["content"]
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if isinstance(first.get("message"), dict) and isinstance(first["message"].get("content"), str):
                    return first["message"]["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
    raise RuntimeError(f"Could not extract assistant content from response: {data}")


def _extract_json_object(text: str) -> str:
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        sanitized_text = _escape_json_string_control_chars(text)
        if sanitized_text != text:
            try:
                json.loads(sanitized_text)
                return sanitized_text
            except json.JSONDecodeError:
                pass
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                sanitized_candidate = _escape_json_string_control_chars(text[index:])
                if sanitized_candidate == text[index:]:
                    continue
                try:
                    parsed, _ = decoder.raw_decode(sanitized_candidate)
                except json.JSONDecodeError:
                    continue
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        raise


def _escape_json_string_control_chars(text: str) -> str:
    pieces: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            pieces.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            pieces.append(char)
            escaped = True
            continue
        if char == '"':
            pieces.append(char)
            in_string = not in_string
            continue
        if in_string and char == "\n":
            pieces.append("\\n")
            continue
        if in_string and char == "\r":
            pieces.append("\\r")
            continue
        if in_string and char == "\t":
            pieces.append("\\t")
            continue
        pieces.append(char)
    return "".join(pieces)


def _has_llm_access(settings: LLMSettings) -> bool:
    if settings.backend == "company_api":
        return bool(settings.base_url and settings.provider and settings.model)
    return bool(os.getenv("OPENAI_API_KEY"))


def _clean_manual_feedback(manual_feedback: dict[int, str] | None) -> dict[int, str]:
    if not manual_feedback:
        return {}
    return {int(index): text.strip() for index, text in manual_feedback.items() if text and text.strip()}


def _summarize_feedback(bad_cases: list[BadCase], manual_feedback: dict[int, str]) -> str:
    pieces = [f"{case.source}:{case.turn_index}:{case.error_type}" for case in bad_cases]
    pieces.extend(f"human:{index}" for index in manual_feedback)
    return ", ".join(pieces) if pieces else "No feedback."


def _fallback_prompt_update(
    system_prompt: str, bad_cases: list[BadCase], manual_feedback: dict[int, str]
) -> str:
    additions = [
        "",
        "Additional reliability requirements:",
        "- Follow every explicit policy, constraint, and negative instruction in this system prompt.",
        "- If a user request conflicts with policy or provided facts, refuse or correct it clearly and politely.",
        "- Do not invent policy details. Ask for clarification when required information is missing.",
        "- If the user gives a vague, uncertain, partial, refused, or unrelated reply, do not advance to the next workflow step until the required condition is clearly satisfied; ask one focused clarification question or use the allowed fallback.",
    ]
    for index, feedback in manual_feedback.items():
        additions.append(f"- Address human feedback for turn {index}: {feedback}")
    for case in bad_cases:
        if case.recommendation:
            additions.append(f"- {case.recommendation}")
    return system_prompt.rstrip() + "\n" + "\n".join(dict.fromkeys(additions))
