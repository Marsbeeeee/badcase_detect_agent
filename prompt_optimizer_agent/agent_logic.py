from __future__ import annotations

import json
import os
import re
import difflib
from dataclasses import dataclass
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
DEFAULT_COMPANY_MODEL = os.getenv("COMPANY_LLM_MODEL", "H200_01_fc_9010")
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
    """Deterministic checks for explicit hard rules that LLM judges often under-count."""
    cases: list[BadCase] = []
    if _prompt_requires_concrete_payment_fallback(data.system_prompt):
        cases.extend(_local_open_ended_payment_fallback_cases(data))
    if _prompt_requires_ptp_attempt_limit_closing(data.system_prompt):
        cases.extend(_local_ptp_attempt_limit_cases(data))
    if _prompt_requires_fresh_search_for_inquiries(data.system_prompt):
        cases.extend(_local_missing_fresh_search_cases(data))
    return _dedupe_bad_cases(cases)


def _prompt_requires_concrete_payment_fallback(system_prompt: str) -> bool:
    lowered = system_prompt.lower()
    return (
        "never ask" in lowered
        and ("amount or date" in lowered or "when can you pay" in lowered or "how much can you pay" in lowered)
        and ("concrete" in lowered or "pre-defined" in lowered or "predefined" in lowered)
        and ("payment" in lowered or "pay" in lowered)
    )


def _prompt_requires_ptp_attempt_limit_closing(system_prompt: str) -> bool:
    lowered = system_prompt.lower()
    has_attempt_limit = (
        "maximum of three attempts" in lowered
        or "3-attempt" in lowered
        or "three attempts" in lowered
    )
    return has_attempt_limit and "state 4.0" in lowered and "rtp_closing" in lowered


def _prompt_requires_fresh_search_for_inquiries(system_prompt: str) -> bool:
    lowered = system_prompt.lower()
    return (
        "fresh search" in lowered
        and (
            "every customer message" in lowered
            or "every message requires" in lowered
            or "call search_promotion" in lowered
        )
        and ("promotion" in lowered or "promotions" in lowered or "discount" in lowered)
    )


def _local_missing_fresh_search_cases(data: ConversationData) -> list[BadCase]:
    cases: list[BadCase] = []
    pending_user_index: int | None = None
    pending_started_before_name = False
    search_seen_for_pending = False
    promotion_context_active = False
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
            if _is_promotion_user_message(turn.content, promotion_context_active):
                pending_user_index = index
                pending_started_before_name = not name_known
                search_seen_for_pending = False
                promotion_context_active = True
            continue

        if role == "assistant":
            if _is_search_promotion_tool_call(turn.content):
                if pending_user_index is not None:
                    search_seen_for_pending = True
                promotion_context_active = True
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
            if not _assistant_answers_promotion_or_product(turn.content):
                continue
            if search_seen_for_pending:
                pending_user_index = None
                search_seen_for_pending = False
                continue
            cases.append(
                _missing_fresh_search_bad_case(
                    data=data,
                    assistant_index=index,
                    user_index=pending_user_index,
                    name_gate_closed_index=name_gate_closed_index if pending_started_before_name else None,
                )
            )
            pending_user_index = None
            search_seen_for_pending = False
            promotion_context_active = True
            continue

        if role == "tool":
            previous_assistant_asked_name = False

    return cases


def _missing_fresh_search_bad_case(
    data: ConversationData,
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
        "Violated rule: Promotion-related customer messages require a fresh search before answering.\n\n"
        "Evidence: "
        f"User turn {user_index} asked or followed up on a promotion{name_gate_note}. "
        f"No search_promotion tool call appeared before assistant turn {assistant_index}, "
        f"which answered with product/promotion details ({_promotion_answer_excerpt(data.interactions[assistant_index].content)})."
    )
    return BadCase(
        turn_index=assistant_index,
        role="assistant",
        error_type="missing_fresh_promotion_search",
        evidence=evidence,
        recommendation=(
            "Revise the inquiry/promotion workflow so every promotion-related customer message triggers "
            "MandiriCX_Call_Center_search_promotion before any factual product or promotion answer, including "
            "follow-up confirmations and corrections."
        ),
        source="local_scan",
    )


def _is_search_promotion_tool_call(content: str) -> bool:
    return bool(
        re.search(
            r"<function-call>\s*MandiriCX_Call_Center_search_promotion\s*:",
            content,
            flags=re.IGNORECASE,
        )
        or re.search(r"<function-call>\s*search_promotion\s*:", content, flags=re.IGNORECASE)
    )


def _assistant_asks_for_customer_name(text: str) -> bool:
    lowered = _compact_text(text).lower()
    return (
        ("siapa" in lowered and "berbicara" in lowered)
        or "boleh saya tahu" in lowered and ("nama" in lowered or "siapa" in lowered)
        or "may i know" in lowered and "name" in lowered
    )


def _is_conversation_marker(text: str) -> bool:
    return _compact_text(text).strip().lower() in {"[conversation begins]", "conversation begins"}


def _is_promotion_user_message(text: str, promotion_context_active: bool) -> bool:
    lowered = _compact_text(text).lower()
    if _is_conversation_marker(lowered):
        return False
    explicit_terms = (
        "promo",
        "promosi",
        "discount",
        "diskon",
        "personal loan",
        "personel",
        "pinjaman",
        "kredit serbaguna",
        "program",
    )
    if any(term in lowered for term in explicit_terms):
        return True
    if not promotion_context_active:
        return False
    followup_terms = (
        "dua miliar",
        "2 miliar",
        "rp2 miliar",
        "satu miliar",
        "1 miliar",
        "minimum",
        "dana",
        "tenor",
        "benefit",
        "syarat",
        "bener",
        "benar",
        "bukannya",
    )
    return any(term in lowered for term in followup_terms)


def _assistant_answers_promotion_or_product(text: str) -> bool:
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
    answer_terms = (
        "limit",
        "pinjaman",
        "kredit",
        "nasabah",
        "prioritas",
        "miliar",
        "juta",
        "suku bunga",
        "tenor",
        "tanpa agunan",
        "benefit",
        "program",
        "livin",
    )
    return any(term in lowered for term in answer_terms)


def _promotion_answer_excerpt(text: str) -> str:
    compacted = _compact_text(text)
    patterns = (
        r"(?:limit|pinjaman|kredit|personal loan)[^.!?]*(?:miliar|juta|rupiah)[^.!?]*[.!?]?",
        r"(?:program|nasabah|prioritas)[^.!?]*(?:miliar|minimum|dana|bulan)[^.!?]*[.!?]?",
        r"(?:suku bunga|tenor|tanpa agunan)[^.!?]*[.!?]?",
    )
    for pattern in patterns:
        match = re.search(pattern, compacted, flags=re.IGNORECASE)
        if match:
            return _quote_turn(match.group(0), max_chars=120)
    return _quote_turn(compacted, max_chars=120)


def _local_open_ended_payment_fallback_cases(data: ConversationData) -> list[BadCase]:
    rejected_offer = _latest_rejected_full_payment_today(data, len(data.interactions))
    if rejected_offer is None:
        return []
    proposal_index, rejection_index = rejected_offer
    offender_indices = [
        index
        for index, turn in enumerate(data.interactions)
        if (
            index > rejection_index
            and turn.role.lower() == "assistant"
            and _asks_open_ended_payment_terms(turn.content)
        )
    ]
    if not offender_indices:
        return []

    first_offender = offender_indices[0]
    repeated_note = ""
    if len(offender_indices) > 1:
        repeated_note = f" The same pattern repeats at assistant turns {_format_index_list(offender_indices)}."
    evidence = (
        "Violated rule: After a concrete payment proposal is rejected, propose the next pre-defined terms; "
        "do not ask the user for payment terms.\n\n"
        "Evidence: "
        f"Turn {rejection_index} rejected the full-payment-today proposal from turn {proposal_index}. "
        f"Assistant turn {first_offender} then asked the user for payment timing "
        f"({_payment_term_question_excerpt(data.interactions[first_offender].content)}) instead of proposing a concrete fallback."
        f"{repeated_note}"
    )
    return [
        BadCase(
            turn_index=first_offender,
            role="assistant",
            error_type="open_ended_payment_terms_after_rejection",
            evidence=evidence,
            recommendation=(
                "Revise the relevant State 2.2 negotiation branch so that after the full-payment-today "
                "proposal is rejected, the assistant offers the next concrete pre-defined fallback term "
                "instead of asking the user when or how much they can pay."
            ),
            source="local_scan",
        )
    ]


def _local_ptp_attempt_limit_cases(data: ConversationData) -> list[BadCase]:
    for index, turn in enumerate(data.interactions):
        if turn.role.lower() != "assistant":
            continue
        if not _asks_open_ended_payment_terms(turn.content):
            continue
        failed_user_indices = _failed_payment_collection_user_indices(data, before_index=index)
        if len(failed_user_indices) < 3:
            continue
        counted_indices = failed_user_indices[:3]
        evidence = (
            "Violated rule: After three failed proposal-collection attempts, proceed to State 4.0 "
            "(RTP_Closing) without prompting again.\n\n"
            "Evidence: "
            f"The failed proposal-collection count reached {len(counted_indices)} before assistant turn {index} "
            f"(user turns {_format_index_list(counted_indices)} all refused payment without giving a valid date). "
            f"Assistant turn {index} still asked for a payment date "
            f"({_payment_term_question_excerpt(turn.content)}) instead of moving to RTP_Closing."
        )
        return [
            BadCase(
                turn_index=index,
                role="assistant",
                error_type="ptp_attempt_limit_exceeded",
                evidence=evidence,
                recommendation=(
                    "Revise State 2.2 so the assistant keeps an explicit failed-attempt counter and, "
                    "once the third failed clarify/collect attempt is reached, immediately transitions "
                    "to State 4.0 (RTP_Closing) without asking another payment-date or payment-amount question."
                ),
                source="local_scan",
            )
        ]
    return []


def _latest_rejected_full_payment_today(
    data: ConversationData,
    before_index: int,
) -> tuple[int, int] | None:
    latest: tuple[int, int] | None = None
    for index, turn in enumerate(data.interactions[:before_index]):
        if turn.role.lower() != "user" or not _is_payment_refusal(turn.content):
            continue
        proposal_index = _previous_assistant_index(data, index)
        if proposal_index is None:
            continue
        if _proposes_full_payment_today(data.interactions[proposal_index].content):
            latest = (proposal_index, index)
    return latest


def _previous_assistant_index(data: ConversationData, before_index: int) -> int | None:
    for index in range(before_index - 1, -1, -1):
        if data.interactions[index].role.lower() == "assistant":
            return index
    return None


def _proposes_full_payment_today(text: str) -> bool:
    lowered = text.lower()
    return (
        "today" in lowered
        and ("payment" in lowered or "pay" in lowered or "settle" in lowered)
        and ("full" in lowered or re.search(r"\b\d+(?:\.\d+)?\b", lowered) is not None)
    )


def _asks_open_ended_payment_terms(text: str) -> bool:
    lowered = _compact_text(text).lower()
    if not any(term in lowered for term in ("pay", "payment", "settle")):
        return False
    patterns = (
        r"\b(when|what date|which date)\b.{0,140}\b(pay|payment|settle|make a payment|make payment)\b",
        r"\b(pay|payment|settle|make a payment|make payment)\b.{0,140}\b(when|what date|which date)\b",
        r"\b(let me know|tell me|share|provide|give me)\b.{0,140}\b(date|amount|when|how much)\b",
        r"\bhow much\b.{0,140}\b(pay|payment|settle)\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _failed_payment_collection_user_indices(data: ConversationData, before_index: int) -> list[int]:
    failed_indices: list[int] = []
    negotiation_started = False
    for index, turn in enumerate(data.interactions[:before_index]):
        if turn.role.lower() == "assistant" and _looks_like_payment_negotiation_turn(turn.content):
            negotiation_started = True
            continue
        if not negotiation_started or turn.role.lower() != "user":
            continue
        if _is_failed_payment_collection_response(turn.content):
            failed_indices.append(index)
    return failed_indices


def _looks_like_payment_negotiation_turn(text: str) -> bool:
    lowered = text.lower()
    return (
        ("overdue" in lowered and ("loan" in lowered or "payment" in lowered))
        or "payment hasn't been made" in lowered
        or _proposes_full_payment_today(text)
        or _asks_open_ended_payment_terms(text)
    )


def _is_failed_payment_collection_response(text: str) -> bool:
    return _is_payment_refusal(text) and not _contains_concrete_payment_date(text)


def _is_payment_refusal(text: str) -> bool:
    lowered = _compact_text(text).lower()
    refusal_patterns = (
        r"\bno\b",
        r"\bnot paying\b",
        r"\bcan't\b",
        r"\bcannot\b",
        r"\bcan not\b",
        r"\bunable\b",
        r"\bwon't\b",
        r"\bwill not\b",
        r"\bdon't have\b",
        r"\bdo not have\b",
        r"\bdon't plan\b",
        r"\bdo not plan\b",
        r"\bno money\b",
        r"\bnot able\b",
    )
    return any(re.search(pattern, lowered) for pattern in refusal_patterns)


def _contains_concrete_payment_date(text: str) -> bool:
    lowered = _compact_text(text).lower()
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", lowered):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", lowered):
        return True
    if re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", lowered):
        return True
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b", lowered):
        return True
    date_terms = (
        "today",
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "this weekend",
        "next week",
        "next month",
    )
    return any(term in lowered for term in date_terms)


def _format_turn_snippets(data: ConversationData, turn_indices: list[int]) -> str:
    return "; ".join(
        f"turn {index}: {_quote_turn(data.interactions[index].content)}"
        for index in turn_indices
        if 0 <= index < len(data.interactions)
    )


def _format_index_list(indices: list[int]) -> str:
    return ", ".join(str(index) for index in indices)


def _payment_term_question_excerpt(text: str) -> str:
    compacted = _compact_text(text)
    patterns = (
        r"could you please [^?.!]*(?:payment|date|when|amount)[^?.!]*[?.!]?",
        r"please (?:let me know|provide|tell me|share|give me)[^?.!]*(?:payment|date|when|amount)[^?.!]*[?.!]?",
        r"(?:when|what date|which date)[^?.!]*(?:pay|payment|settle)[^?.!]*[?.!]?",
        r"(?:how much)[^?.!]*(?:pay|payment|settle)[^?.!]*[?.!]?",
    )
    for pattern in patterns:
        match = re.search(pattern, compacted, flags=re.IGNORECASE)
        if match:
            return _quote_turn(match.group(0), max_chars=105)
    return _quote_turn(compacted, max_chars=105)


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
    if case.source == "local_scan":
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
    if "promotion" in text and ("search_promotion" in text or "fresh search" in text or "tool" in text):
        return "missing_fresh_promotion_search"
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
    return _apply_recommendation_full_prompt(
        settings=settings,
        current_system_prompt=current_system_prompt,
        bad_case=bad_case,
        related_turn=turn,
        payload=payload,
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
        style_violation = _prompt_edit_style_violation(current_system_prompt, optimized_prompt)
        if _has_step_append_violation(current_system_prompt, optimized_prompt) or style_violation:
            optimized_prompt = _retry_step_preserving_prompt_edit(
                settings=settings,
                current_system_prompt=current_system_prompt,
                bad_case=bad_case,
                related_turn=related_turn,
                rejected_prompt=optimized_prompt,
                rejection_reason=style_violation or "appended a separate note/requirements section",
            )
        return PromptOptimization(
            optimized_prompt=optimized_prompt,
            rationale=str(parsed.get("rationale") or ""),
            applied_feedback_summary=str(parsed.get("applied_feedback_summary") or ""),
        )
    except (OpenAIError, requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return PromptOptimization(
            optimized_prompt=current_system_prompt,
            rationale=(
                f"LLM prompt edit failed: {exc}. The prompt was left unchanged because this app "
                "now requires in-place step edits rather than appended notes."
            ),
            applied_feedback_summary="No changes applied.",
        )


def rerun_conversation(
    data: ConversationData,
    optimized_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    llm_settings: LLMSettings | None = None,
    target_assistant_turn_indices: set[int] | None = None,
    required_tools_by_turn: dict[int, str] | None = None,
    expected_system_prompt_hash: str | None = None,
    expected_prompt_version: str | None = None,
) -> list[RerunTurn]:
    settings = llm_settings or LLMSettings(model=model)
    results: list[RerunTurn] = []
    replay_messages: list[dict[str, str]] = [{"role": "system", "content": optimized_prompt}]
    rerun_all = target_assistant_turn_indices is None
    if target_assistant_turn_indices is None:
        target_assistant_turn_indices = {
            index for index, turn in enumerate(data.interactions) if turn.role == "assistant"
        }
    required_tools_by_turn = required_tools_by_turn or {}

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
    for index, turn in enumerate(data.interactions):
        if turn.role == "user":
            pending_user = (index, turn.content)
            replay_messages.append(_interaction_replay_message(turn))
            continue

        if turn.role == "assistant" and pending_user is not None:
            user_index, user_message = pending_user
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
                request_messages = _with_required_tool_instruction(request_messages, required_tool)
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
                    new_response = _forced_required_tool_call(required_tool, user_message)
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
            request_messages = _with_required_tool_instruction(request_messages, required_tool)
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
    if re.search(pattern, response, flags=re.IGNORECASE):
        return True
    if required_tool.lower().endswith("search_promotion"):
        return _is_search_promotion_tool_call(response)
    return False


def _forced_required_tool_call(required_tool: str, user_message: str) -> str:
    arguments = {"query": _forced_tool_query(user_message)}
    return f"<function-call>{required_tool}:{json.dumps(arguments, ensure_ascii=False)}</function-call>"


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
        "paragraph_1 must summarize the exact previous prompt/rule change using paragraph_inputs.paragraph_1_changes. "
        "Mention concrete changed rule text or duplicated/weak edit quality when present.\n\n"
        "paragraph_2 must evaluate whether the bad case was actually improved using "
        "paragraph_inputs.paragraph_2_improvement_evidence. Do not claim the response is identical when "
        "target_turn_comparisons says only wording changed; instead say wording changed but target behavior did or did "
        "not improve. Use token/logprob evidence only when token_probability.available is true. If it is false, say "
        "there is no token-probability evidence and use behavior/meta/tool evidence.\n\n"
        "paragraph_3 must give one actionable next optimization using paragraph_inputs.paragraph_3_next_recommendation. "
        "Tie it to the observed root cause and avoid generic advice."
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
    )
    messages = [
        {"role": "system", "content": conclusion_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    _write_conclusion_dialog_log(messages)
    try:
        content = _chat_json(
            settings=settings,
            messages=messages,
            purpose="conclusion",
        )
        parsed = json.loads(content)
        return _format_conclusion_value(parsed.get("conclusion"))
    except (OpenAIError, requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return f"Conclusion generation failed: {exc}"


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
) -> dict[str, object]:
    prompt_diff = _prompt_unified_diff(before_prompt, optimized_prompt)
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
    return {
        "output_contract": {
            "format": {
                "conclusion": {
                    "paragraph_1": "exact prompt/rule change",
                    "paragraph_2": "improvement verdict with deeper evidence",
                    "paragraph_3": "actionable next optimization",
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
                "prompt_diff": prompt_diff,
                "prompt_change_quality": prompt_change_quality,
                "instruction": (
                    "State the exact changed rule. If the edit only repeats an abstract instruction "
                    "or duplicates wording, say that clearly."
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
                    "Prefer behavior, tool-call state, metadata, and residual scan evidence over prompt wording alone.",
                ],
            },
            "paragraph_3_next_recommendation": {
                "observed_root_cause_hints": _conclusion_root_cause_hints(
                    target_turn_comparisons=target_turn_comparisons,
                    prompt_change_quality=prompt_change_quality,
                    token_probability=token_probability,
                    post_rerun_bad_cases=post_rerun_bad_cases,
                    post_rerun_scan_status=post_rerun_scan_status,
                ),
                "recommendation_rules": [
                    "Recommend one concrete prompt or workflow edit.",
                    "If the current edit duplicated abstract guidance, recommend replacing it with a concrete response pattern.",
                    "If rerun behavior still misses the required action, recommend an explicit branch/action gate.",
                ],
            },
        },
        "raw_evidence": {
            "before_prompt": before_prompt,
            "optimized_prompt": optimized_prompt,
            "prompt_diff": prompt_diff,
            "source_meta": data.source_meta,
            "tools": data.tools,
            "original_interactions": [turn.model_dump(exclude_none=True) for turn in data.interactions],
            "updated_interactions": [turn.model_dump(exclude_none=True) for turn in updated_interactions],
            "rerun_evidence": rerun_evidence,
            "rerun_results": [_rerun_result_dict(result) for result in rerun_results],
            "post_rerun_bad_cases": [case.__dict__ for case in post_rerun_bad_cases],
        },
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
        return _extract_json_object(text)
    client = OpenAI()
    response = client.chat.completions.create(
        model=settings.model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


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
        messages=messages,
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
        messages=messages,
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
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        raise


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
