from __future__ import annotations

import copy
import difflib
import hashlib
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from prompt_optimizer_agent import agent_logic
from prompt_optimizer_agent.agent_logic import (
    BadCase,
    DEFAULT_MODEL,
    LLMSettings,
    PromptOptimization,
    RerunTurn,
    apply_recommendation_to_system_prompt,
    analyze_bad_cases,
    generate_experiment_conclusion,
    optimize_system_prompt,
    rerun_conversation,
    translate_interactions_to_zh,
)
from prompt_optimizer_agent.company_demo_client import (
    list_company_models,
)
from prompt_optimizer_agent.json_utils import (
    ConversationData,
    Interaction,
    function_call_wrappers_to_tool_calls,
    parse_conversation_json,
)


load_dotenv()

DEFAULT_COMPANY_URL = os.getenv("COMPANY_LLM_URL", "http://192.168.101.15:9898")
DEFAULT_COMPANY_PROVIDER = os.getenv("COMPANY_LLM_PROVIDER", "openai_api_like")
DEFAULT_COMPANY_MODEL = os.getenv("COMPANY_LLM_MODEL", "voyager-1.6-preview-run27m4a8b4-r3")
APP_BUILD = "round-history-v89"
ROUND_HISTORY_LOG = Path(__file__).parent / "logs" / "optimization_rounds.jsonl"

st.set_page_config(
    page_title="Prompt Optimizer Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="column"] .stButton > button,
    div[data-testid="column"] .stDownloadButton > button {
        min-height: 2.5rem;
        width: 100%;
        white-space: nowrap;
    }
    div[data-testid="column"] .stButton > button p,
    div[data-testid="column"] .stDownloadButton > button p {
        white-space: nowrap;
    }
    .trace-card-title {
        min-height: 2.5rem;
        display: flex;
        align-items: center;
        line-height: 2.5rem;
        font-weight: 700;
    }
    .trace-card-body-spacer {
        height: 0.15rem;
    }
    .trace-card-bottom-spacer {
        height: 0.6rem;
    }
    .section-heading {
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.25;
        margin: 1.1rem 0 0.85rem 0;
    }
    .toast-stack {
        position: fixed;
        top: 5.2rem;
        right: 1.4rem;
        z-index: 100000;
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
        width: min(22rem, calc(100vw - 2rem));
        pointer-events: none;
    }
    .app-toast {
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-left-width: 0.34rem;
        border-radius: 0.45rem;
        box-shadow: 0 0.75rem 2rem rgba(15, 23, 42, 0.16);
        color: #1f2937;
        font-size: 0.92rem;
        line-height: 1.35;
        padding: 0.85rem 0.95rem;
        background: #ffffff;
        opacity: 0;
        transform: translateY(-0.4rem);
        animation: toast-lifecycle 5600ms ease-in-out forwards;
        animation-delay: var(--toast-delay, 0ms);
    }
    .app-toast-success {
        border-left-color: #16a34a;
        background: #ecfdf3;
        color: #166534;
    }
    .app-toast-error {
        border-left-color: #dc2626;
        background: #fef2f2;
        color: #991b1b;
    }
    .app-toast-warning {
        border-left-color: #d97706;
        background: #fffbeb;
        color: #92400e;
    }
    .app-toast-info {
        border-left-color: #2563eb;
        background: #eff6ff;
        color: #1e40af;
    }
    @keyframes toast-lifecycle {
        0% { opacity: 0; transform: translateY(-0.4rem); }
        8%, 82% { opacity: 1; transform: translateY(0); }
        100% { opacity: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "raw_text": "",
        "original_raw_data": None,
        "data": None,
        "manual_feedback": {},
        "bad_cases": [],
        "bad_cases_conversation_view": None,
        "judge_backend": "current",
        "auto_post_rerun_analysis": False,
        "analysis_status": None,
        "apply_status": None,
        "apply_status_level": "info",
        "apply_audit": [],
        "round_history": [],
        "latest_scan_round_id": None,
        "trace_source_prompt_label": None,
        "optimization": None,
        "original_system_prompt_view": "",
        "current_system_prompt_view": "",
        "optimized_prompt_editor": "",
        "pending_optimized_prompt_editor": None,
        "pending_current_system_prompt_view": None,
        "pending_generate_badcase": False,
        "pending_apply_action": None,
        "last_apply_summary": None,
        "last_apply_diff": "",
        "expected_rerun_prompt_hash": None,
        "expected_rerun_prompt_version": None,
        "prompt_versions": [],
        "selected_prompt_version_index": None,
        "trace_list_version_index": None,
        "rerun_results": [],
        "rerun_interactions": None,
        "rerun_summary": None,
        "rerun_conclusion": None,
        "last_rerun_experiment": None,
        "conversation_view": "Original",
        "show_translations": False,
        "translation_cache": {},
        "translation_status": None,
        "translation_status_level": "info",
        "last_parse_error": None,
        "last_warnings": [],
        "validation_status": None,
        "last_uploaded_file_id": None,
        "company_models": [],
        "company_models_error": None,
        "bad_case_selection_error": None,
        "pending_trace_scroll_restore": False,
        "toast_notifications": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def push_toast(message: str | None, level: str = "info") -> None:
    if not message:
        return
    allowed_levels = {"success", "error", "warning", "info"}
    toast = {
        "message": str(message),
        "level": level if level in allowed_levels else "info",
    }
    st.session_state.toast_notifications = [
        *st.session_state.get("toast_notifications", []),
        toast,
    ][-4:]


def render_toasts() -> None:
    notifications = st.session_state.get("toast_notifications", [])
    if not notifications:
        return
    pieces = []
    for index, toast in enumerate(notifications):
        level = html.escape(str(toast.get("level") or "info"))
        message = html.escape(str(toast.get("message") or ""))
        delay_ms = index * 1050
        pieces.append(
            f'<div class="app-toast app-toast-{level}" '
            f'style="--toast-delay: {delay_ms}ms">{message}</div>'
        )
    st.markdown(
        f'<div class="toast-stack">{"".join(pieces)}</div>',
        unsafe_allow_html=True,
    )
    st.session_state.toast_notifications = []


def reset_translation_state() -> None:
    st.session_state.show_translations = False
    st.session_state.translation_cache = {}
    st.session_state.translation_status = None
    st.session_state.translation_status_level = "info"


def clear_legacy_anchor_hash(anchor: str) -> None:
    components.html(
        f"""
        <script>
        try {{
            const parentWindow = window.parent || window;
            if (parentWindow.location.hash === '#{anchor}') {{
                parentWindow.history.replaceState(
                    null,
                    parentWindow.document.title,
                    parentWindow.location.pathname + parentWindow.location.search
                );
            }}
        }} catch (error) {{}}
        </script>
        """,
        height=0,
        width=0,
    )


def request_trace_scroll_restore() -> None:
    st.session_state.pending_trace_scroll_restore = True


def render_trace_scroll_restore(anchor_id: str = "trace-list-top") -> None:
    if not st.session_state.get("pending_trace_scroll_restore"):
        return
    st.session_state.pending_trace_scroll_restore = False
    safe_anchor_id = json.dumps(anchor_id)
    components.html(
        f"""
        <script>
        try {{
            const parentWindow = window.parent || window;
            const parentDocument = parentWindow.document;
            const anchorId = {safe_anchor_id};
            const restore = () => {{
                const target = parentDocument.getElementById(anchorId);
                if (!target) return;
                target.scrollIntoView({{ behavior: 'auto', block: 'start', inline: 'nearest' }});
                parentWindow.scrollBy(0, -12);
            }};
            parentWindow.requestAnimationFrame(restore);
            parentWindow.setTimeout(restore, 60);
            parentWindow.setTimeout(restore, 180);
        }} catch (error) {{}}
        </script>
        """,
        height=0,
        width=0,
    )


def load_sample() -> None:
    sample_path = Path(__file__).parent / "examples" / "sample_conversation.json"
    st.session_state.raw_text = sample_path.read_text(encoding="utf-8")
    st.session_state.last_uploaded_file_id = None
    parse_current_json()


def parse_current_json(standardize_raw_text: bool = False) -> None:
    original_raw_data = parse_raw_json_snapshot(st.session_state.raw_text)
    result = parse_conversation_json(st.session_state.raw_text)
    st.session_state.last_parse_error = result.error
    st.session_state.last_warnings = result.warnings
    if result.error or result.data is None:
        st.session_state.data = None
        st.session_state.original_raw_data = None
        st.session_state.bad_cases = []
        st.session_state.bad_cases_conversation_view = None
        st.session_state.analysis_status = None
        st.session_state.apply_status = None
        st.session_state.apply_status_level = "info"
        st.session_state.apply_audit = []
        st.session_state.round_history = []
        st.session_state.latest_scan_round_id = None
        st.session_state.trace_source_prompt_label = None
        st.session_state.optimization = None
        st.session_state.rerun_results = []
        st.session_state.rerun_interactions = None
        st.session_state.rerun_summary = None
        st.session_state.rerun_conclusion = None
        st.session_state.last_rerun_experiment = None
        st.session_state.conversation_view = "Original"
        st.session_state.expected_rerun_prompt_hash = None
        st.session_state.expected_rerun_prompt_version = None
        st.session_state.prompt_versions = []
        st.session_state.selected_prompt_version_index = None
        st.session_state.trace_list_version_index = None
        st.session_state.pending_generate_badcase = False
        st.session_state.pending_apply_action = None
        st.session_state.validation_status = None
        reset_translation_state()
        return

    st.session_state.data = result.data
    st.session_state.original_raw_data = original_raw_data or result.raw_data
    st.session_state.bad_cases = []
    st.session_state.bad_cases_conversation_view = None
    st.session_state.analysis_status = None
    st.session_state.apply_status = None
    st.session_state.apply_status_level = "info"
    st.session_state.apply_audit = []
    st.session_state.round_history = []
    st.session_state.latest_scan_round_id = None
    st.session_state.trace_source_prompt_label = None
    st.session_state.optimization = None
    st.session_state.rerun_results = []
    st.session_state.rerun_interactions = None
    st.session_state.rerun_summary = None
    st.session_state.rerun_conclusion = None
    st.session_state.last_rerun_experiment = None
    st.session_state.conversation_view = "Original"
    st.session_state.last_apply_summary = None
    st.session_state.last_apply_diff = ""
    st.session_state.expected_rerun_prompt_hash = None
    st.session_state.expected_rerun_prompt_version = None
    st.session_state.prompt_versions = []
    st.session_state.selected_prompt_version_index = None
    st.session_state.trace_list_version_index = None
    st.session_state.pending_generate_badcase = False
    st.session_state.pending_apply_action = None
    reset_translation_state()
    st.session_state.original_system_prompt_view = result.data.system_prompt
    st.session_state.current_system_prompt_view = result.data.system_prompt
    st.session_state.optimized_prompt_editor = result.data.system_prompt
    st.session_state.prompt_versions = [
        {
            "label": "v0 Original",
            "prompt": result.data.system_prompt,
            "summary": "Original uploaded system prompt.",
            "diff": "",
            "trace_cases": [],
            "trace_conversation_view": None,
            "trace_source_prompt_label": None,
            "applied_trace_count": 0,
            "scan_trace_recorded": False,
            "scan_trace_cases": [],
            "scan_conversation_view": None,
            "scan_source_prompt_label": None,
        }
    ]
    st.session_state.trace_list_version_index = None
    if standardize_raw_text:
        st.session_state.raw_text = normalized_conversation_json(result.data)
        st.session_state.last_warnings = []
        st.session_state.validation_status = "JSON validated and converted to the standard schema."
    elif result.repaired_text:
        st.session_state.raw_text = result.repaired_text


def parse_raw_json_snapshot(raw_text: str):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def validate_current_json() -> None:
    parse_current_json(standardize_raw_text=True)


def clear_loaded_conversation() -> None:
    st.session_state.raw_text = ""
    st.session_state.original_raw_data = None
    st.session_state.data = None
    st.session_state.manual_feedback = {}
    st.session_state.bad_cases = []
    st.session_state.bad_cases_conversation_view = None
    st.session_state.analysis_status = None
    st.session_state.apply_status = None
    st.session_state.apply_status_level = "info"
    st.session_state.apply_audit = []
    st.session_state.round_history = []
    st.session_state.latest_scan_round_id = None
    st.session_state.trace_source_prompt_label = None
    st.session_state.optimization = None
    st.session_state.original_system_prompt_view = ""
    st.session_state.current_system_prompt_view = ""
    st.session_state.optimized_prompt_editor = ""
    st.session_state.pending_optimized_prompt_editor = None
    st.session_state.pending_current_system_prompt_view = None
    st.session_state.pending_generate_badcase = False
    st.session_state.pending_apply_action = None
    st.session_state.last_apply_summary = None
    st.session_state.last_apply_diff = ""
    st.session_state.expected_rerun_prompt_hash = None
    st.session_state.expected_rerun_prompt_version = None
    st.session_state.prompt_versions = []
    st.session_state.selected_prompt_version_index = None
    st.session_state.trace_list_version_index = None
    st.session_state.rerun_results = []
    st.session_state.rerun_interactions = None
    st.session_state.rerun_summary = None
    st.session_state.rerun_conclusion = None
    st.session_state.last_rerun_experiment = None
    st.session_state.conversation_view = "Original"
    st.session_state.expected_rerun_prompt_hash = None
    st.session_state.expected_rerun_prompt_version = None
    st.session_state.last_parse_error = None
    st.session_state.last_warnings = []
    st.session_state.validation_status = None
    st.session_state.pending_generate_badcase = False
    st.session_state.pending_apply_action = None
    reset_translation_state()


def normalized_conversation_json(data: ConversationData) -> str:
    payload = {
        "system_prompt": data.system_prompt,
        "interactions": [turn.model_dump(exclude_none=True) for turn in data.interactions],
    }
    if data.tools:
        payload["tools"] = data.tools
    if data.source_meta:
        payload["source_meta"] = data.source_meta
    return json.dumps(payload, indent=2, ensure_ascii=False)


def export_with_updated_system_prompt(data: ConversationData, updated_prompt: str) -> str:
    current_interactions = st.session_state.get("rerun_interactions") or data.interactions
    if st.session_state.get("rerun_interactions"):
        payload = {
            "system_prompt": updated_prompt,
            "interactions": [turn.model_dump(exclude_none=True) for turn in current_interactions],
        }
        if data.tools:
            payload["tools"] = data.tools
        if data.source_meta:
            payload["source_meta"] = data.source_meta
        return json.dumps(payload, ensure_ascii=False)

    raw_data = st.session_state.original_raw_data
    if raw_data is None:
        payload = {
            "system_prompt": updated_prompt,
            "interactions": [turn.model_dump(exclude_none=True) for turn in current_interactions],
        }
        if data.tools:
            payload["tools"] = data.tools
        return json.dumps(payload, ensure_ascii=False)

    exported = copy.deepcopy(raw_data)
    if replace_system_prompt_in_raw(exported, updated_prompt):
        return json.dumps(exported, ensure_ascii=False)

    payload = {
        "system_prompt": updated_prompt,
        "interactions": [turn.model_dump(exclude_none=True) for turn in data.interactions],
    }
    if data.tools:
        payload["tools"] = data.tools
    return json.dumps(payload, ensure_ascii=False)


def replace_system_prompt_in_raw(value, updated_prompt: str) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("system_prompt"), str):
            value["system_prompt"] = updated_prompt
            return True
        for key in ("dialog", "messages", "interactions", "conversation", "conversations", "chat", "turns"):
            if replace_system_prompt_in_message_list(value.get(key), updated_prompt):
                return True
        for nested in value.values():
            if isinstance(nested, (dict, list)) and replace_system_prompt_in_raw(nested, updated_prompt):
                return True
    if isinstance(value, list):
        return replace_system_prompt_in_message_list(value, updated_prompt)
    return False


def replace_system_prompt_in_message_list(value, updated_prompt: str) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, dict) and str(item.get("role", "")).lower() == "system":
            item["content"] = updated_prompt
            return True
    return False


def reset_analysis_state() -> None:
    st.session_state.bad_cases = []
    st.session_state.bad_cases_conversation_view = None
    st.session_state.analysis_status = None
    st.session_state.apply_status = None
    st.session_state.apply_status_level = "info"
    st.session_state.apply_audit = []
    st.session_state.round_history = []
    st.session_state.latest_scan_round_id = None
    st.session_state.trace_source_prompt_label = None
    st.session_state.optimization = None
    st.session_state.rerun_results = []
    st.session_state.rerun_interactions = None
    st.session_state.rerun_summary = None
    st.session_state.rerun_conclusion = None
    st.session_state.last_rerun_experiment = None
    st.session_state.conversation_view = "Original"
    st.session_state.pending_generate_badcase = False
    st.session_state.pending_apply_action = None


def split_analysis_results(cases: list[BadCase]) -> tuple[list[BadCase], list[BadCase]]:
    technical_errors = [
        case
        for case in cases
        if case.turn_index < 0 or case.error_type.endswith("_api_error")
    ]
    trace_cases = [
        case
        for case in cases
        if case not in technical_errors
    ]
    return trace_cases, technical_errors


def safe_analyze_bad_cases(
    data: ConversationData,
    manual_feedback: dict[int, str] | None,
    llm_settings: LLMSettings,
) -> list[BadCase]:
    try:
        return analyze_bad_cases(
            data=data,
            manual_feedback=manual_feedback,
            llm_settings=llm_settings,
        )
    except Exception as exc:
        return [
            BadCase(
                turn_index=-1,
                role="system",
                error_type="judge_api_error",
                evidence=str(exc),
                recommendation="AI judge failed. Retry after checking API settings or use manual bad-case marking.",
            )
        ]


def judge_llm_settings(current_settings: LLMSettings, judge_backend: str) -> LLMSettings:
    if judge_backend == "openai":
        return LLMSettings(
            backend="openai",
            model=os.getenv("PROMPT_OPTIMIZER_MODEL", DEFAULT_MODEL),
            max_completion_tokens=current_settings.max_completion_tokens,
        )
    return current_settings


def format_analysis_error_status(prefix: str, technical_errors: list[BadCase]) -> str:
    evidence = "; ".join(
        format_api_error_evidence(case.evidence)
        for case in technical_errors
        if case.evidence
    )
    return f"{prefix} AI judge could not parse or complete its response. {evidence}".strip()


def format_api_error_evidence(evidence: str) -> str:
    request_id_match = re.search(r"request_id:\s*([A-Za-z0-9-]+)", evidence)
    request_id = f" request_id={request_id_match.group(1)}." if request_id_match else ""
    if "502 Bad Gateway" in evidence:
        return f"Company API returned 502 Bad Gateway.{request_id} Retry later or refresh the company model/backend."
    if "<html" in evidence.lower():
        return f"Company API returned an HTML error response.{request_id} Retry later or check the backend."
    return evidence


def build_rerun_interactions(data: ConversationData, rerun_results: list[RerunTurn]):
    replacements = {
        result.assistant_turn_index: result
        for result in rerun_results
        if result.error is None
        and result.assistant_turn_index is not None
        and result.new_assistant_response
    }
    rebuilt: list[Interaction] = []
    for index, turn in enumerate(data.interactions):
        result = replacements.get(index)
        if result is None:
            rebuilt.append(turn)
            continue
        new_response = result.new_assistant_response
        tool_calls = function_call_wrappers_to_tool_calls(
            new_response,
            id_prefix=f"call_rerun_{index}",
        )
        update = {
            "content": new_response,
            "tool_calls": tool_calls or None,
            "tool_call_id": None,
        }
        rebuilt.append(turn.model_copy(update=update))
        for call in tool_calls:
            rebuilt.append(rerun_tool_placeholder_interaction(call))
    return rebuilt


def rerun_tool_placeholder_interaction(tool_call: dict) -> Interaction:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict):
        function = {}
    payload = {
        "status": "not_executed",
        "note": (
            "The rerun assistant requested this tool call. No local tool executor is configured, "
            "so no real tool result was generated."
        ),
        "tool_name": str(function.get("name") or ""),
        "arguments": str(function.get("arguments") or ""),
    }
    return Interaction(
        role="tool",
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=str(tool_call.get("id") or "") if isinstance(tool_call, dict) else None,
    )


def build_current_analysis_data(data: ConversationData, current_prompt: str) -> ConversationData:
    return data.model_copy(
        update={
            "system_prompt": current_prompt,
            "interactions": st.session_state.rerun_interactions or data.interactions,
        }
    )


def rerun_targets_for_cases(data: ConversationData, cases: list[BadCase]) -> set[int]:
    targets: set[int] = set()
    for case in cases:
        if 0 <= case.turn_index < len(data.interactions):
            if data.interactions[case.turn_index].role == "assistant":
                targets.add(case.turn_index)
                continue
            for index in range(case.turn_index + 1, len(data.interactions)):
                if data.interactions[index].role == "assistant":
                    targets.add(index)
                    break
    return targets


def required_tools_for_cases(data: ConversationData, cases: list[BadCase]) -> dict[int, str]:
    required: dict[int, str] = {}
    for case in cases:
        tool_name = required_tool_name_for_case(data.tools, case)
        if not tool_name:
            continue
        for target_index in rerun_targets_for_cases(data, [case]):
            required[target_index] = tool_name
    return required


def required_tool_name_for_case(tools: dict | None, case: BadCase) -> str | None:
    if not case_is_missing_required_tool_call(case):
        return None
    tool_names = available_tool_names(tools)
    if not tool_names:
        return None
    text = bad_case_text(case)
    lowered = text.lower()
    for tool_name in tool_names:
        if tool_name.lower() in lowered:
            return tool_name
    return best_matching_tool_name(tool_names, text)


def case_is_missing_required_tool_call(case: BadCase) -> bool:
    lowered = bad_case_text(case).lower()
    missing_signal = any(
        phrase in lowered
        for phrase in (
            "missing",
            "no ",
            "without",
            "skipped",
            "skip",
            "requires",
            "require",
            "must call",
            "should call",
            "before answering",
        )
    )
    tool_signal = any(
        phrase in lowered
        for phrase in (
            "tool",
            "function",
            "call",
            "fresh search",
            "search",
            "retrieval",
            "retrieve",
            "lookup",
            "look up",
        )
    )
    return missing_signal and tool_signal


def available_tool_names(tools: dict | None) -> list[str]:
    if not tools:
        return []
    names: list[str] = []
    for key, value in tools.items():
        if key:
            names.append(str(key))
        if isinstance(value, dict):
            function = value.get("function")
            if isinstance(function, dict) and function.get("name"):
                names.append(str(function["name"]))
            elif value.get("name"):
                names.append(str(value["name"]))
    return list(dict.fromkeys(names))


def best_matching_tool_name(tool_names: list[str], text: str) -> str | None:
    text_tokens = set(normalized_tool_tokens(text))
    best_name = None
    best_score = 0
    for tool_name in tool_names:
        tool_tokens = [
            token
            for token in normalized_tool_tokens(tool_name)
            if token
            not in {
                "tool",
                "call",
                "get",
                "set",
                "search",
                "lookup",
                "retrieve",
                "retrieval",
                "function",
                "api",
            }
        ]
        if not tool_tokens:
            continue
        score = len(set(tool_tokens) & text_tokens)
        lowered_tool = tool_name.lower()
        lowered_text = text.lower()
        if "search" in lowered_tool and "search" in lowered_text:
            score += 1
        if "retrieval" in lowered_tool and "retrieval" in lowered_text:
            score += 1
        if score > best_score:
            best_name = tool_name
            best_score = score
    return best_name if best_score > 0 else None


def normalized_tool_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())


def bad_case_text(case: BadCase) -> str:
    return " ".join([case.error_type, case.evidence, case.recommendation])


def rerun_with_prompt(
    data: ConversationData,
    before_prompt: str,
    optimized_prompt: str,
    applied_feedback_summary: str,
    llm_settings: LLMSettings,
    target_assistant_turn_indices: set[int] | None = None,
    required_tools_by_turn: dict[int, str] | None = None,
    expected_system_prompt_hash: str | None = None,
    expected_prompt_version: str | None = None,
) -> None:
    original_data = data.model_copy(update={"system_prompt": before_prompt})
    data.system_prompt = optimized_prompt
    expected_system_prompt_hash = expected_system_prompt_hash or prompt_fingerprint(optimized_prompt)
    expected_prompt_version = expected_prompt_version or current_prompt_version_label()
    st.session_state.expected_rerun_prompt_hash = expected_system_prompt_hash
    st.session_state.expected_rerun_prompt_version = expected_prompt_version
    st.session_state.rerun_results = rerun_conversation(
        data=data,
        optimized_prompt=optimized_prompt,
        llm_settings=llm_settings,
        target_assistant_turn_indices=target_assistant_turn_indices,
        required_tools_by_turn=required_tools_by_turn,
        expected_system_prompt_hash=expected_system_prompt_hash,
        expected_prompt_version=expected_prompt_version,
    )
    st.session_state.rerun_interactions = build_rerun_interactions(
        data,
        st.session_state.rerun_results,
    )
    updated_data = data.model_copy(
        update={
            "system_prompt": optimized_prompt,
            "interactions": st.session_state.rerun_interactions,
        }
    )
    st.session_state.rerun_summary = summarize_rerun(
        st.session_state.rerun_results,
        llm_settings,
    )
    rerun_failed = any(result.error for result in st.session_state.rerun_results)
    push_toast(
        st.session_state.rerun_summary,
        "error" if rerun_failed else "success",
    )
    st.session_state.rerun_conclusion = None
    st.session_state.last_rerun_experiment = {
        "before_prompt": before_prompt,
        "optimized_prompt": optimized_prompt,
        "applied_feedback_summary": applied_feedback_summary,
        "prompt_version": expected_prompt_version,
        "target_assistant_turn_indices": sorted(target_assistant_turn_indices or []),
    }
    st.session_state.bad_cases = []
    st.session_state.bad_cases_conversation_view = None
    st.session_state.trace_source_prompt_label = None
    st.session_state.conversation_view = "Updated"
    st.session_state.manual_feedback = {}
    for key in list(st.session_state.keys()):
        if (
            str(key).startswith("bad_case_")
            or str(key).startswith("manual_feedback_input_")
            or str(key).startswith("select_bad_case_")
        ):
            st.session_state.pop(key, None)
    if not st.session_state.get("auto_post_rerun_analysis", False):
        if rerun_failed:
            set_analysis_status("Rerun incomplete.", "error", notify=False)
        else:
            set_analysis_status("Ready to scan.", "info")
        st.session_state.rerun_conclusion = generate_experiment_conclusion(
            data=original_data,
            before_prompt=before_prompt,
            optimized_prompt=optimized_prompt,
            rerun_results=st.session_state.rerun_results,
            updated_interactions=st.session_state.rerun_interactions,
            post_rerun_bad_cases=[],
            applied_feedback_summary=applied_feedback_summary,
            llm_settings=llm_settings,
            post_rerun_scan_status="not_run",
        )
        if st.session_state.rerun_conclusion.startswith(
            ("Conclusion could not", "Conclusion was skipped", "Conclusion generation failed")
        ):
            push_toast("Conclusion skipped.", "warning")
        else:
            push_toast("Conclusion ready.", "success")
        return
    judge_settings = judge_llm_settings(llm_settings, st.session_state.judge_backend)
    analyzed_cases = safe_analyze_bad_cases(
        data=updated_data,
        manual_feedback={},
        llm_settings=judge_settings,
    )
    post_rerun_bad_cases, analysis_errors = split_analysis_results(analyzed_cases)
    if post_rerun_bad_cases:
        status = "completed"
        status_level = "warning"
        status_message = f"Updated analyzed. {len(post_rerun_bad_cases)} residual badcase(s)."
    elif analysis_errors:
        status = "failed"
        status_level = "warning"
        status_message = format_analysis_error_status("Updated generated, but", analysis_errors)
    elif judge_settings.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        status = "skipped"
        status_level = "warning"
        status_message = "Updated ready. OpenAI judge key missing."
    else:
        status = "completed"
        status_level = "success"
        status_message = "Updated analyzed. No badcases."
    set_analysis_status(status_message, status_level)
    record_scan_round(
        post_rerun_bad_cases,
        analysis_errors,
        status=status,
        status_message=status_message,
        status_level=status_level,
        judge_settings=judge_settings,
        conversation_view="Updated",
        trigger="auto_post_rerun_analysis",
    )
    st.session_state.rerun_conclusion = generate_experiment_conclusion(
        data=original_data,
        before_prompt=before_prompt,
        optimized_prompt=optimized_prompt,
        rerun_results=st.session_state.rerun_results,
        updated_interactions=st.session_state.rerun_interactions,
        post_rerun_bad_cases=post_rerun_bad_cases,
        applied_feedback_summary=applied_feedback_summary,
        llm_settings=llm_settings,
        post_rerun_scan_status="completed",
    )


def summarize_rerun(rerun_results: list[RerunTurn], llm_settings: LLMSettings) -> str:
    attempted = len([result for result in rerun_results if result.assistant_turn_index is not None])
    replaced = len(
        [
            result
            for result in rerun_results
            if result.assistant_turn_index is not None
            and result.error is None
            and result.new_assistant_response
        ]
    )
    failed = len([result for result in rerun_results if result.error])
    if failed:
        return f"Rerun failed: {failed}/{attempted}."
    return "Rerun succeeded."


def detected_company_settings(data: ConversationData | None) -> dict[str, str]:
    model_info = {}
    if data and isinstance(data.source_meta, dict) and isinstance(data.source_meta.get("model_info"), dict):
        model_info = data.source_meta["model_info"]
    return {
        "url": str(model_info.get("url") or DEFAULT_COMPANY_URL),
        "provider": str(model_info.get("provider") or DEFAULT_COMPANY_PROVIDER),
        "model": str(model_info.get("model") or DEFAULT_COMPANY_MODEL),
    }


def model_choices_for_company(data: ConversationData | None, detected: dict[str, str]) -> list[str]:
    choices: list[str] = []
    if detected["provider"] and detected["model"]:
        choices.append(f"{detected['provider']}:{detected['model']}")
    choices.extend(st.session_state.company_models)
    env_models = os.getenv("COMPANY_LLM_MODELS", "")
    choices.extend(item.strip() for item in env_models.split(",") if item.strip())
    choices.append(f"{DEFAULT_COMPANY_PROVIDER}:{DEFAULT_COMPANY_MODEL}")
    return list(dict.fromkeys(choice for choice in choices if choice))


def split_company_model(selection: str, fallback_provider: str) -> tuple[str, str]:
    if ":" in selection:
        provider, model = selection.split(":", 1)
        return provider.strip(), model.strip()
    return fallback_provider, selection.strip()


def refresh_company_models(url: str) -> None:
    try:
        st.session_state.company_models = list_company_models(url)
        st.session_state.company_models_error = None
    except Exception as exc:
        st.session_state.company_models_error = str(exc)


def is_auto_detected_bad_case(case: BadCase) -> bool:
    return case.turn_index >= 0 and case.source != "human"


def cases_by_turn(cases: list[BadCase]) -> dict[int, list[BadCase]]:
    cases_by_turn: dict[int, list[BadCase]] = {}
    for case in cases:
        if is_auto_detected_bad_case(case):
            cases_by_turn.setdefault(case.turn_index, []).append(case)
    return cases_by_turn


def current_cases_by_turn() -> dict[int, list[BadCase]]:
    return cases_by_turn(st.session_state.bad_cases)


def trace_version_conversation_view(trace_version: dict | None) -> str | None:
    if trace_version is None:
        return None
    trace_view = trace_version.get("scan_conversation_view")
    return trace_view if trace_view in ("Updated", "Original") else None


def active_trace_conversation_view(trace_version: dict | None = None) -> str | None:
    version_view = trace_version_conversation_view(trace_version)
    if version_view:
        return version_view
    current_view = st.session_state.bad_cases_conversation_view
    if st.session_state.bad_cases and current_view in ("Updated", "Original"):
        return current_view
    return None


def active_trace_cases_by_turn(
    conversation_view: str,
    trace_version: dict | None = None,
) -> dict[int, list[BadCase]]:
    trace_view = active_trace_conversation_view(trace_version)
    if trace_view != conversation_view:
        return {}
    if trace_version is not None:
        return cases_by_turn(version_trace_cases(trace_version))
    return current_cases_by_turn()


def prompt_diff(before: str, after: str, from_label: str, to_label: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=from_label,
        tofile=to_label,
        lineterm="",
    )
    return "\n".join(diff)


def ensure_prompt_versions(current_prompt: str) -> None:
    if st.session_state.prompt_versions:
        return
    original = st.session_state.original_system_prompt_view or current_prompt
    st.session_state.prompt_versions = [
        {
            "label": "v0 Original",
            "prompt": original,
            "summary": "Original uploaded system prompt.",
            "diff": "",
            "trace_cases": [],
            "trace_conversation_view": None,
            "trace_source_prompt_label": None,
            "applied_trace_count": 0,
            "scan_trace_recorded": False,
            "scan_trace_cases": [],
            "scan_conversation_view": None,
            "scan_source_prompt_label": None,
        }
    ]


def add_prompt_version(
    prompt: str,
    summary: str,
    before_prompt: str,
    trace_cases: list[BadCase] | None = None,
    applied_cases: list[BadCase] | None = None,
) -> None:
    ensure_prompt_versions(before_prompt)
    version_index = len(st.session_state.prompt_versions)
    label = f"v{version_index} Apply"
    trace_snapshot = prompt_version_trace_snapshot(trace_cases or [], applied_cases or [])
    st.session_state.prompt_versions.append(
        {
            "label": label,
            "prompt": prompt,
            "summary": summary,
            "diff": prompt_diff(before_prompt, prompt, f"v{version_index - 1}", label),
            "trace_cases": trace_snapshot,
            "trace_conversation_view": st.session_state.bad_cases_conversation_view,
            "trace_source_prompt_label": st.session_state.trace_source_prompt_label,
            "applied_trace_count": len([case for case in trace_snapshot if case.get("applied")]),
            "scan_trace_recorded": False,
            "scan_trace_cases": [],
            "scan_conversation_view": None,
            "scan_source_prompt_label": None,
        }
    )
    st.session_state.selected_prompt_version_index = version_index
    st.session_state.trace_list_version_index = None


def prompt_version_trace_snapshot(
    trace_cases: list[BadCase],
    applied_cases: list[BadCase],
) -> list[dict[str, object]]:
    applied_ids = {bad_case_trace_id(case) for case in applied_cases}
    return [
        {
            "id": bad_case_trace_id(case),
            "turn_index": case.turn_index,
            "role": case.role,
            "error_type": case.error_type,
            "source": case.source,
            "evidence": case.evidence,
            "recommendation": case.recommendation,
            "applied": bad_case_trace_id(case) in applied_ids,
        }
        for case in trace_cases
    ]


def bad_case_trace_id(case: BadCase) -> str:
    payload = {
        "turn_index": case.turn_index,
        "role": case.role,
        "error_type": case.error_type,
        "source": case.source,
        "evidence": case.evidence,
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def conversation_fingerprint() -> str | None:
    raw_text = st.session_state.get("raw_text") or ""
    if not raw_text:
        return None
    return hashlib.sha1(str(raw_text).encode("utf-8")).hexdigest()[:10]


def llm_settings_snapshot(settings: LLMSettings) -> dict[str, object]:
    return {
        "backend": settings.backend,
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url,
        "max_completion_tokens": settings.max_completion_tokens,
    }


def append_round_history_event(event: dict[str, object]) -> dict[str, object]:
    history = list(st.session_state.get("round_history") or [])
    event_type = str(event.get("type") or "event")
    type_sequence = len([item for item in history if item.get("type") == event_type]) + 1
    event = {
        **event,
        "event_id": f"{event_type}-{type_sequence:03d}",
        "sequence": len(history) + 1,
        "timestamp": utc_timestamp(),
        "conversation_hash": conversation_fingerprint(),
    }
    history.append(event)
    st.session_state.round_history = history
    persist_round_history_event(event)
    return event


def persist_round_history_event(event: dict[str, object]) -> None:
    try:
        ROUND_HISTORY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ROUND_HISTORY_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        push_toast(f"Round history log write failed: {exc}", "warning")


def record_scan_round(
    cases: list[BadCase],
    analysis_errors: list[BadCase],
    *,
    status: str,
    status_message: str,
    status_level: str,
    judge_settings: LLMSettings,
    conversation_view: str | None = None,
    trigger: str = "manual_generate_badcase",
) -> dict[str, object]:
    prompt = st.session_state.current_system_prompt_view
    prompt_label = current_prompt_version_label()
    event = append_round_history_event(
        {
            "type": "scan",
            "trigger": trigger,
            "status": status,
            "status_level": status_level,
            "status_message": status_message,
            "prompt_version_label": prompt_label,
            "prompt_hash": prompt_fingerprint(prompt),
            "prompt_len": len(prompt),
            "conversation_view": conversation_view or st.session_state.bad_cases_conversation_view,
            "badcase_count": len(cases),
            "analysis_error_count": len(analysis_errors),
            "badcases": prompt_version_trace_snapshot(cases, []),
            "analysis_errors": prompt_version_trace_snapshot(analysis_errors, []),
            "judge": llm_settings_snapshot(judge_settings),
        }
    )
    st.session_state.latest_scan_round_id = event["event_id"]
    return event


def record_review_round(
    decision: str,
    cases: list[BadCase],
    *,
    reason: str,
) -> None:
    append_round_history_event(
        {
            "type": "review",
            "decision": decision,
            "reason": reason,
            "parent_scan_round_id": st.session_state.get("latest_scan_round_id"),
            "trace_count": len(cases),
            "trace_ids": [bad_case_trace_id(case) for case in cases],
            "badcases": prompt_version_trace_snapshot(cases, []),
        }
    )


def record_apply_round(
    *,
    action: str,
    cases: list[BadCase],
    parent_scan_round_id: str | None,
    before_prompt: str,
    after_prompt: str,
    before_version: str,
    after_version: str,
    rerun_attempted: bool,
    rerun_target_turns: set[int],
    prompt_edit_retries: int,
) -> None:
    append_round_history_event(
        {
            "type": "apply",
            "action": action,
            "human_decision": "approved",
            "parent_scan_round_id": parent_scan_round_id,
            "status": st.session_state.get("apply_status_level", "info"),
            "status_message": st.session_state.get("apply_status"),
            "summary": st.session_state.get("last_apply_summary"),
            "diff": st.session_state.get("last_apply_diff"),
            "approved_trace_count": len(cases),
            "trace_ids": [bad_case_trace_id(case) for case in cases],
            "case_turns": [case.turn_index for case in cases],
            "badcases": prompt_version_trace_snapshot(cases, cases),
            "before_version": before_version,
            "after_version": after_version,
            "prompt_changed": after_prompt != before_prompt,
            "before_hash": prompt_fingerprint(before_prompt),
            "after_hash": prompt_fingerprint(after_prompt),
            "before_len": len(before_prompt),
            "after_len": len(after_prompt),
            "rerun_attempted": rerun_attempted,
            "rerun_target_turns": sorted(rerun_target_turns),
            "prompt_edit_retries": prompt_edit_retries,
        }
    )


def record_current_version_scan_trace(cases: list[BadCase]) -> None:
    ensure_prompt_versions(st.session_state.current_system_prompt_view)
    label = current_prompt_version_label()
    version = prompt_version_by_label(label)
    if version is None:
        return
    version["scan_trace_recorded"] = True
    version["scan_trace_cases"] = prompt_version_trace_snapshot(cases, [])
    version["scan_conversation_view"] = st.session_state.bad_cases_conversation_view
    version["scan_source_prompt_label"] = label


def prompt_version_by_label(label: str) -> dict | None:
    for version in st.session_state.prompt_versions:
        if version.get("label") == label:
            return version
    return None


def current_prompt_version_label() -> str:
    ensure_prompt_versions(st.session_state.current_system_prompt_view)
    if st.session_state.prompt_versions:
        return st.session_state.prompt_versions[-1]["label"]
    return "current prompt"


def sync_loaded_system_prompt(prompt: str) -> None:
    if st.session_state.data is not None:
        st.session_state.data.system_prompt = prompt


def set_apply_status(message: str, level: str = "info", notify: bool = True) -> None:
    st.session_state.apply_status = message
    st.session_state.apply_status_level = level
    if notify:
        push_toast(message, "error" if level == "error" else level)


def set_analysis_status(message: str, level: str = "info", notify: bool = True) -> None:
    st.session_state.analysis_status = message
    if notify:
        push_toast(message, level)


def run_generate_badcase_scan(data: ConversationData, llm_settings: LLMSettings) -> None:
    st.session_state.trace_list_version_index = None
    st.session_state.apply_status = None
    st.session_state.apply_status_level = "info"
    data.system_prompt = st.session_state.current_system_prompt_view
    analysis_data = build_current_analysis_data(
        data,
        st.session_state.current_system_prompt_view,
    )
    judge_settings = judge_llm_settings(llm_settings, st.session_state.judge_backend)
    analyzed_cases = safe_analyze_bad_cases(
        data=analysis_data,
        manual_feedback=st.session_state.manual_feedback,
        llm_settings=judge_settings,
    )
    st.session_state.bad_cases, analysis_errors = split_analysis_results(analyzed_cases)
    st.session_state.bad_cases_conversation_view = (
        "Updated" if st.session_state.rerun_interactions else "Original"
    )
    analyzed_view = st.session_state.bad_cases_conversation_view
    st.session_state.trace_source_prompt_label = current_prompt_version_label()
    st.session_state.conversation_view = analyzed_view
    record_current_version_scan_trace(st.session_state.bad_cases)
    clear_bad_case_selection()
    if st.session_state.bad_cases:
        status = "completed"
        status_level = "success"
        status_message = f"Scan complete. {len(st.session_state.bad_cases)} trace item(s)."
    elif analysis_errors:
        status = "failed"
        status_level = "error"
        status_message = format_analysis_error_status("Scan failed:", analysis_errors)
    elif judge_settings.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        status = "skipped"
        status_level = "warning"
        status_message = "Scan skipped. OpenAI key missing."
    else:
        status = "completed"
        status_level = "success"
        status_message = "Scan complete. No trace items."
    set_analysis_status(status_message, status_level)
    record_scan_round(
        st.session_state.bad_cases,
        analysis_errors,
        status=status,
        status_message=status_message,
        status_level=status_level,
        judge_settings=judge_settings,
    )
    generate_post_scan_conclusion_if_needed(
        data=data,
        llm_settings=llm_settings,
        post_rerun_bad_cases=st.session_state.bad_cases,
        analysis_errors=analysis_errors,
    )


def generate_post_scan_conclusion_if_needed(
    data: ConversationData,
    llm_settings: LLMSettings,
    post_rerun_bad_cases: list[BadCase],
    analysis_errors: list[BadCase],
) -> None:
    if st.session_state.bad_cases_conversation_view != "Updated":
        st.session_state.rerun_conclusion = None
        return

    experiment = st.session_state.get("last_rerun_experiment")
    if (
        not isinstance(experiment, dict)
        or not st.session_state.rerun_interactions
        or not st.session_state.rerun_results
    ):
        st.session_state.rerun_conclusion = None
        return

    if analysis_errors:
        st.session_state.rerun_conclusion = (
            "Conclusion skipped because the post-rerun badcase scan did not complete. "
            "Fix the scan error first, then run generate badcase again."
        )
        push_toast("Conclusion skipped.", "warning")
        return

    before_prompt = str(experiment.get("before_prompt") or data.system_prompt)
    optimized_prompt = str(
        experiment.get("optimized_prompt")
        or st.session_state.current_system_prompt_view
        or data.system_prompt
    )
    applied_feedback_summary = str(experiment.get("applied_feedback_summary") or "")
    original_data = data.model_copy(update={"system_prompt": before_prompt})
    conclusion = generate_experiment_conclusion(
        data=original_data,
        before_prompt=before_prompt,
        optimized_prompt=optimized_prompt,
        rerun_results=st.session_state.rerun_results,
        updated_interactions=st.session_state.rerun_interactions,
        post_rerun_bad_cases=post_rerun_bad_cases,
        applied_feedback_summary=applied_feedback_summary,
        llm_settings=llm_settings,
        post_rerun_scan_status="completed",
    )
    st.session_state.rerun_conclusion = conclusion
    if conclusion.startswith(("Conclusion could not", "Conclusion was skipped", "Conclusion generation failed")):
        push_toast("Conclusion skipped.", "warning")
    else:
        push_toast("Conclusion ready.", "success")


def run_pending_apply_action(data: ConversationData, llm_settings: LLMSettings) -> None:
    action = st.session_state.get("pending_apply_action")
    st.session_state.pending_apply_action = None
    if not isinstance(action, dict):
        return

    current_prompt = st.session_state.current_system_prompt_view
    kind = action.get("kind")
    if kind == "selected":
        case_indices = [
            int(index)
            for index in action.get("case_indices", [])
            if isinstance(index, int) or str(index).isdigit()
        ]
        apply_selected_bad_cases_to_prompt(
            data=data,
            llm_settings=llm_settings,
            case_indices=case_indices,
            current_prompt=current_prompt,
        )
        return

    if kind == "single":
        try:
            case_index = int(action.get("case_index"))
        except (TypeError, ValueError):
            return
        apply_bad_case_to_prompt(
            data=data,
            llm_settings=llm_settings,
            case_index=case_index,
            current_prompt=current_prompt,
        )


def set_apply_status_from_rerun(success_message: str) -> None:
    failed = [
        result
        for result in st.session_state.rerun_results
        if result.error
    ]
    if not failed:
        set_apply_status(success_message, "success", notify=False)
        return
    set_apply_status(
        "Prompt updated, but targeted rerun failed or was blocked before sending. "
        f"{failed[0].error}",
        "error",
        notify=False,
    )


def can_rerun_unchanged_prompt(
    target_indices: set[int],
    required_tools_by_turn: dict[int, str],
) -> bool:
    return bool(target_indices and required_tools_by_turn)


def unchanged_prompt_rerun_summary(target_count: int) -> str:
    turn_word = "turn" if target_count == 1 else "turns"
    return (
        "No new system-prompt version was needed; the existing prompt already contains the required "
        f"tool rule, so this run only regenerated the target assistant {turn_word} with the current prompt."
    )


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def record_apply_audit(
    *,
    action: str,
    case_turns: list[int],
    before_prompt: str,
    after_prompt: str,
    before_version: str,
    after_version: str,
    rerun_attempted: bool,
    rerun_target_turns: set[int],
    prompt_edit_retries: int = 0,
) -> None:
    audit = {
        "action": action,
        "case_turns": case_turns,
        "prompt_changed": after_prompt != before_prompt,
        "before_hash": prompt_fingerprint(before_prompt),
        "after_hash": prompt_fingerprint(after_prompt),
        "before_len": len(before_prompt),
        "after_len": len(after_prompt),
        "before_version": before_version,
        "after_version": after_version,
        "rerun_attempted": rerun_attempted,
        "rerun_target_turns": sorted(rerun_target_turns),
        "prompt_edit_retries": prompt_edit_retries,
    }
    st.session_state.apply_audit = [audit] + st.session_state.apply_audit[:9]


def retry_prompt_edit_if_unchanged(
    data: ConversationData,
    current_prompt: str,
    bad_case: BadCase,
    llm_settings: LLMSettings,
    optimization: PromptOptimization,
) -> tuple[PromptOptimization, int]:
    if optimization.optimized_prompt != current_prompt:
        return optimization, 0
    retry_case = BadCase(
        turn_index=bad_case.turn_index,
        role=bad_case.role,
        error_type=bad_case.error_type,
        evidence=(
            bad_case.evidence
            + "\n\nPrevious apply attempt returned the system prompt unchanged."
        ),
        recommendation=(
            "The previous apply attempt returned the system prompt unchanged. "
            "Make one concrete, minimal edit to the existing system prompt text so it explicitly "
            "addresses this bad case. Do not return the identical prompt. "
            f"Original recommendation: {bad_case.recommendation}"
        ),
        source=bad_case.source,
    )
    retry_optimization = apply_recommendation_to_system_prompt(
        data=data,
        current_system_prompt=current_prompt,
        bad_case=retry_case,
        llm_settings=llm_settings,
    )
    if retry_optimization.optimized_prompt == current_prompt:
        return optimization, 1
    return retry_optimization, 1


def undo_prompt_version() -> None:
    ensure_prompt_versions(st.session_state.current_system_prompt_view)
    if len(st.session_state.prompt_versions) <= 1:
        return
    removed = st.session_state.prompt_versions.pop()
    previous = st.session_state.prompt_versions[-1]
    st.session_state.pending_current_system_prompt_view = previous["prompt"]
    st.session_state.optimization = None
    st.session_state.last_apply_summary = f"Undid {removed['label']}; restored {previous['label']}."
    st.session_state.last_apply_diff = prompt_diff(
        removed["prompt"],
        previous["prompt"],
        removed["label"],
        previous["label"],
    )
    st.session_state.selected_prompt_version_index = len(st.session_state.prompt_versions) - 1
    st.session_state.trace_list_version_index = None


def save_manual_bad_case(
    data: ConversationData,
    llm_settings: LLMSettings,
    turn_index: int,
    role: str,
    feedback: str,
) -> None:
    feedback = feedback.strip()
    if not feedback:
        return
    st.session_state.manual_feedback[turn_index] = feedback
    recommendation_fn = getattr(agent_logic, "recommend_for_manual_bad_case", None)
    if recommendation_fn is None:
        recommendation = f"Add an explicit instruction addressing this human-reported issue: {feedback}"
    else:
        recommendation = recommendation_fn(
            data=data,
            turn_index=turn_index,
            feedback=feedback,
            llm_settings=llm_settings,
        )
    new_case = BadCase(
        turn_index=turn_index,
        role=role,
        error_type="human_flagged",
        evidence=feedback,
        recommendation=recommendation,
        source="human",
    )
    st.session_state.bad_cases = [
        case
        for case in st.session_state.bad_cases
        if not (case.source == "human" and case.turn_index == turn_index)
    ]
    st.session_state.bad_cases.append(new_case)


def delete_bad_case(case_index: int) -> None:
    if not 0 <= case_index < len(st.session_state.bad_cases):
        return
    case = st.session_state.bad_cases[case_index]
    record_review_round(
        "rejected",
        [case],
        reason="Trace item deleted during human review.",
    )
    if case.source == "human":
        st.session_state.manual_feedback.pop(case.turn_index, None)
        st.session_state[f"bad_case_{case.turn_index}"] = False
    st.session_state.bad_cases = [
        existing
        for index, existing in enumerate(st.session_state.bad_cases)
        if index != case_index
    ]


def bad_case_selection_key(case: BadCase) -> str:
    case_id = hashlib.sha1(
        json.dumps(case.__dict__, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"select_bad_case_{case_id}"


def clear_bad_case_selection() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("select_bad_case_"):
            st.session_state[key] = False


def select_all_bad_cases() -> None:
    for case in st.session_state.bad_cases:
        st.session_state[bad_case_selection_key(case)] = True


def selected_bad_case_indices() -> list[int]:
    return [
        case_index
        for case_index, case in enumerate(st.session_state.bad_cases)
        if st.session_state.get(bad_case_selection_key(case), False)
    ]


def apply_bad_case_to_prompt(
    data: ConversationData,
    llm_settings: LLMSettings,
    case_index: int,
    current_prompt: str,
) -> None:
    if not 0 <= case_index < len(st.session_state.bad_cases):
        return
    case = st.session_state.bad_cases[case_index]
    parent_scan_round_id = st.session_state.get("latest_scan_round_id")
    before_version = current_prompt_version_label()
    optimization = apply_recommendation_to_system_prompt(
        data=data,
        current_system_prompt=current_prompt,
        bad_case=case,
        llm_settings=llm_settings,
    )
    optimization, retry_count = retry_prompt_edit_if_unchanged(
        data=data,
        current_prompt=current_prompt,
        bad_case=case,
        llm_settings=llm_settings,
        optimization=optimization,
    )
    st.session_state.optimization = optimization
    st.session_state.pending_current_system_prompt_view = optimization.optimized_prompt
    st.session_state.pending_optimized_prompt_editor = optimization.optimized_prompt
    st.session_state.last_apply_summary = (
        optimization.applied_feedback_summary
        or f"Applied {case.source}:{case.turn_index}:{case.error_type}."
    )
    st.session_state.last_apply_diff = prompt_diff(
        current_prompt,
        optimization.optimized_prompt,
        "Before Apply",
        "After Apply",
    )
    target_indices = rerun_targets_for_cases(data, [case])
    required_tools_by_turn = required_tools_for_cases(data, [case])
    rerun_attempted = False
    if optimization.optimized_prompt != current_prompt:
        sync_loaded_system_prompt(optimization.optimized_prompt)
        add_prompt_version(
            optimization.optimized_prompt,
            st.session_state.last_apply_summary,
            current_prompt,
            trace_cases=st.session_state.bad_cases,
            applied_cases=[case],
        )
        push_toast("Prompt updated.", "success")
        render_toasts()
        rerun_with_prompt(
            data=data,
            before_prompt=current_prompt,
            optimized_prompt=optimization.optimized_prompt,
            applied_feedback_summary=st.session_state.last_apply_summary,
            llm_settings=llm_settings,
            target_assistant_turn_indices=target_indices,
            required_tools_by_turn=required_tools_by_turn,
        )
        rerun_attempted = True
        set_apply_status_from_rerun("Applied and rerun.")
    elif can_rerun_unchanged_prompt(target_indices, required_tools_by_turn):
        rerun_only_summary = unchanged_prompt_rerun_summary(len(target_indices))
        st.session_state.last_apply_summary = rerun_only_summary
        push_toast("No new prompt version; rerunning target turn.", "info")
        rerun_with_prompt(
            data=data,
            before_prompt=current_prompt,
            optimized_prompt=current_prompt,
            applied_feedback_summary=rerun_only_summary,
            llm_settings=llm_settings,
            target_assistant_turn_indices=target_indices,
            required_tools_by_turn=required_tools_by_turn,
            expected_prompt_version=before_version,
        )
        rerun_attempted = True
        set_apply_status_from_rerun("Conversation rerun completed with the existing prompt.")
    else:
        set_apply_status(
            "Prompt edit failed; rerun skipped. The prompt was unchanged after retry, "
            "so the trace item was kept.",
            "warning",
        )
    record_apply_audit(
        action="Apply",
        case_turns=[case.turn_index],
        before_prompt=current_prompt,
        after_prompt=optimization.optimized_prompt,
        before_version=before_version,
        after_version=current_prompt_version_label(),
        rerun_attempted=rerun_attempted,
        rerun_target_turns=target_indices,
        prompt_edit_retries=retry_count,
    )
    record_apply_round(
        action="Apply",
        cases=[case],
        parent_scan_round_id=parent_scan_round_id,
        before_prompt=current_prompt,
        after_prompt=optimization.optimized_prompt,
        before_version=before_version,
        after_version=current_prompt_version_label(),
        rerun_attempted=rerun_attempted,
        rerun_target_turns=target_indices,
        prompt_edit_retries=retry_count,
    )


def apply_selected_bad_cases_to_prompt(
    data: ConversationData,
    llm_settings: LLMSettings,
    case_indices: list[int],
    current_prompt: str,
) -> None:
    selected_cases = [
        st.session_state.bad_cases[case_index]
        for case_index in case_indices
        if 0 <= case_index < len(st.session_state.bad_cases)
    ]
    if not selected_cases:
        st.session_state.bad_case_selection_error = "Select at least one bad case first."
        return

    parent_scan_round_id = st.session_state.get("latest_scan_round_id")
    before_prompt = current_prompt
    before_version = current_prompt_version_label()
    working_prompt = current_prompt
    applied_summaries = []
    last_optimization = None
    retry_count = 0
    for case in selected_cases:
        optimization = apply_recommendation_to_system_prompt(
            data=data,
            current_system_prompt=working_prompt,
            bad_case=case,
            llm_settings=llm_settings,
        )
        optimization, case_retry_count = retry_prompt_edit_if_unchanged(
            data=data,
            current_prompt=working_prompt,
            bad_case=case,
            llm_settings=llm_settings,
            optimization=optimization,
        )
        retry_count += case_retry_count
        last_optimization = optimization
        if optimization.optimized_prompt != working_prompt:
            working_prompt = optimization.optimized_prompt
        applied_summaries.append(
            optimization.applied_feedback_summary
            or f"Applied {case.source}:{case.turn_index}:{case.error_type}."
        )

    st.session_state.bad_case_selection_error = None
    if last_optimization is not None:
        st.session_state.optimization = last_optimization
    st.session_state.pending_current_system_prompt_view = working_prompt
    st.session_state.pending_optimized_prompt_editor = working_prompt
    st.session_state.last_apply_summary = "Applied selected bad cases: " + " ".join(applied_summaries)
    st.session_state.last_apply_diff = prompt_diff(
        before_prompt,
        working_prompt,
        "Before Apply Selected",
        "After Apply Selected",
    )
    target_indices = rerun_targets_for_cases(data, selected_cases)
    required_tools_by_turn = required_tools_for_cases(data, selected_cases)
    rerun_attempted = False
    if working_prompt != before_prompt:
        sync_loaded_system_prompt(working_prompt)
        add_prompt_version(
            working_prompt,
            st.session_state.last_apply_summary,
            before_prompt,
            trace_cases=st.session_state.bad_cases,
            applied_cases=selected_cases,
        )
        push_toast("Prompt updated.", "success")
        render_toasts()
        rerun_with_prompt(
            data=data,
            before_prompt=before_prompt,
            optimized_prompt=working_prompt,
            applied_feedback_summary=st.session_state.last_apply_summary,
            llm_settings=llm_settings,
            target_assistant_turn_indices=target_indices,
            required_tools_by_turn=required_tools_by_turn,
        )
        rerun_attempted = True
        set_apply_status_from_rerun("Applied and rerun.")
    elif can_rerun_unchanged_prompt(target_indices, required_tools_by_turn):
        rerun_only_summary = unchanged_prompt_rerun_summary(len(target_indices))
        st.session_state.last_apply_summary = rerun_only_summary
        push_toast("No new prompt version; rerunning selected target turn(s).", "info")
        rerun_with_prompt(
            data=data,
            before_prompt=before_prompt,
            optimized_prompt=before_prompt,
            applied_feedback_summary=rerun_only_summary,
            llm_settings=llm_settings,
            target_assistant_turn_indices=target_indices,
            required_tools_by_turn=required_tools_by_turn,
            expected_prompt_version=before_version,
        )
        rerun_attempted = True
        set_apply_status_from_rerun("Conversation rerun completed with the existing prompt.")
    else:
        set_apply_status(
            "Prompt edit failed; rerun skipped. The prompt was unchanged after retry, "
            "so the trace items were kept.",
            "warning",
        )
    record_apply_audit(
        action="Apply selected",
        case_turns=[case.turn_index for case in selected_cases],
        before_prompt=before_prompt,
        after_prompt=working_prompt,
        before_version=before_version,
        after_version=current_prompt_version_label(),
        rerun_attempted=rerun_attempted,
        rerun_target_turns=target_indices,
        prompt_edit_retries=retry_count,
    )
    record_apply_round(
        action="Apply selected",
        cases=selected_cases,
        parent_scan_round_id=parent_scan_round_id,
        before_prompt=before_prompt,
        after_prompt=working_prompt,
        before_version=before_version,
        after_version=current_prompt_version_label(),
        rerun_attempted=rerun_attempted,
        rerun_target_turns=target_indices,
        prompt_edit_retries=retry_count,
    )
    clear_bad_case_selection()


def conversation_translation_cache_key(conversation_view: str, interactions: list[Interaction]) -> str:
    payload = [
        {
            "role": turn.role,
            "content": turn.content,
        }
        for turn in interactions
    ]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{conversation_view}:{digest}"


def render_translation_controls(
    visible_interactions: list[Interaction],
    conversation_view: str,
    llm_settings: LLMSettings,
) -> dict[int, str]:
    cache_key = conversation_translation_cache_key(conversation_view, visible_interactions)
    translation_cache = st.session_state.translation_cache
    cached = translation_cache.get(cache_key)
    has_cached_translation = isinstance(cached, dict) and bool(cached)

    translate_col, toggle_col, clear_col = st.columns([0.42, 0.29, 0.29], gap="small")
    translate_label = "Refresh Chinese translation" if has_cached_translation else "Translate to Chinese"
    if translate_col.button(translate_label, use_container_width=True):
        with st.spinner("Translating current conversation..."):
            try:
                translations = translate_interactions_to_zh(
                    visible_interactions,
                    llm_settings=llm_settings,
                )
                if translations:
                    translation_cache[cache_key] = translations
                    st.session_state.translation_cache = translation_cache
                    st.session_state.show_translations = True
                    st.session_state.translation_status = (
                        f"Translated {len(translations)} turn(s) in the {conversation_view} conversation."
                    )
                    st.session_state.translation_status_level = "success"
                else:
                    st.session_state.translation_status = "Translation returned no turn-level results."
                    st.session_state.translation_status_level = "warning"
            except Exception as exc:
                st.session_state.translation_status = f"Translation failed: {exc}"
                st.session_state.translation_status_level = "error"
        cached = translation_cache.get(cache_key)
        has_cached_translation = isinstance(cached, dict) and bool(cached)

    toggle_label = "Hide translations" if st.session_state.show_translations else "Show translations"
    if toggle_col.button(toggle_label, disabled=not has_cached_translation, use_container_width=True):
        st.session_state.show_translations = not st.session_state.show_translations
        st.rerun()

    if clear_col.button("Clear translation", disabled=not has_cached_translation, use_container_width=True):
        translation_cache.pop(cache_key, None)
        st.session_state.translation_cache = translation_cache
        st.session_state.show_translations = False
        st.session_state.translation_status = "Cleared cached translation for this conversation view."
        st.session_state.translation_status_level = "info"
        st.rerun()

    render_translation_status()
    if st.session_state.show_translations and has_cached_translation and isinstance(cached, dict):
        return {int(index): str(value) for index, value in cached.items()}
    return {}


def render_translation_status() -> None:
    message = st.session_state.get("translation_status")
    if not message:
        return
    level = st.session_state.get("translation_status_level", "info")
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def render_turn(
    data: ConversationData,
    llm_settings: LLMSettings,
    index: int,
    role: str,
    content: str,
    ai_cases: list | None = None,
    translation: str | None = None,
) -> None:
    role_label = "User" if role == "user" else "Assistant" if role == "assistant" else role.title()
    with st.container(border=True):
        header_col, action_col = st.columns([0.75, 0.25], vertical_alignment="center")
        ai_label = " | AI bad case" if ai_cases else ""
        header_col.markdown(f"**{index}. {role_label}{ai_label}**")
        marked = action_col.checkbox("Bad case", key=f"bad_case_{index}")
        render_turn_content(role, translation or content, translated=bool(translation))
        if ai_cases:
            for case in ai_cases:
                st.warning(f"Analysis: {case.evidence}")
                if case.recommendation:
                    st.caption(f"Recommendation: {case.recommendation}")
        if marked:
            feedback = st.text_area(
                "What went wrong?",
                value=st.session_state.manual_feedback.get(index, ""),
                key=f"manual_feedback_input_{index}",
                placeholder="Example: The assistant ignored the 30-day return policy.",
            )
            if st.button("Save remark", key=f"save_manual_feedback_{index}"):
                with st.spinner("Generating recommendation from your remark..."):
                    save_manual_bad_case(data, llm_settings, index, role, feedback)
                st.rerun()


def render_turn_content(role: str, content: str, translated: bool = False) -> None:
    if translated or role.lower() != "tool":
        st.markdown(content)
        return
    for label, body, language in tool_content_display_blocks(content):
        if label:
            st.caption(label)
        st.code(body, language=language)


def tool_content_display_blocks(content: str) -> list[tuple[str, str, str]]:
    parsed = parse_json_object_string(content)
    if not isinstance(parsed, dict):
        return [("Tool output", content, "text")]
    string_items = [
        (str(key), value)
        for key, value in parsed.items()
        if isinstance(value, str)
    ]
    if string_items and len(string_items) == len(parsed):
        return [
            (f"Tool output: {key}", value, tool_string_language(value))
            for key, value in string_items
        ]
    return [
        (
            "Tool output JSON",
            json.dumps(parsed, ensure_ascii=False, indent=2),
            "json",
        )
    ]


def parse_json_object_string(content: str) -> dict | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def tool_string_language(value: str) -> str:
    if re.search(r"(?m)^#{1,6}\s|^\s*[-*]\s|\|.+\|", value):
        return "markdown"
    return "text"


def render_bad_cases(
    data: ConversationData,
    llm_settings: LLMSettings,
    current_prompt: str,
    trace_version: dict | None = None,
) -> None:
    if trace_version is not None:
        render_version_trace_list(trace_version)
        return

    if not st.session_state.bad_cases:
        st.info("No trace items yet.")
        return
    ai_count = len([case for case in st.session_state.bad_cases if is_auto_detected_bad_case(case)])
    human_count = len([case for case in st.session_state.bad_cases if case.source == "human"])
    if st.session_state.bad_cases_conversation_view:
        prompt_label = st.session_state.trace_source_prompt_label or "current prompt"
        st.caption(
            f"Trace source: {st.session_state.bad_cases_conversation_view} conversation / {prompt_label}"
        )
    st.caption(f"AI-detected: {ai_count} | Human-marked: {human_count}")
    select_col, clear_col, apply_col = st.columns([0.28, 0.28, 0.44], gap="small")
    if select_col.button("Select all", use_container_width=True):
        select_all_bad_cases()
        request_trace_scroll_restore()
    if clear_col.button("Clear", use_container_width=True):
        clear_bad_case_selection()
        request_trace_scroll_restore()
    render_trace_scroll_restore()
    selected_indices = selected_bad_case_indices()
    if apply_col.button(
        f"Apply selected ({len(selected_indices)})",
        disabled=not selected_indices,
        type="primary",
        use_container_width=True,
    ):
        st.session_state.pending_apply_action = {
            "kind": "selected",
            "case_indices": selected_indices,
        }
        st.rerun()
    if st.session_state.bad_case_selection_error:
        st.warning(st.session_state.bad_case_selection_error)
    for case_index, case in enumerate(st.session_state.bad_cases):
        with st.container(border=True):
            header_col, apply_col, delete_col = st.columns(
                [0.58, 0.21, 0.21],
                vertical_alignment="center",
            )
            with header_col:
                select_item_col, title_col = st.columns([0.12, 0.88], gap="small", vertical_alignment="center")
                select_item_col.checkbox(
                    "Select",
                    key=bad_case_selection_key(case),
                    label_visibility="collapsed",
                )
                title_col.markdown(
                    f"<div class='trace-card-title'>Turn {case.turn_index}</div>",
                    unsafe_allow_html=True,
                )
            if apply_col.button("Apply", key=f"apply_bad_case_{case_index}"):
                st.session_state.pending_apply_action = {
                    "kind": "single",
                    "case_index": case_index,
                }
                st.rerun()
            if delete_col.button("Delete", key=f"delete_bad_case_{case_index}"):
                delete_bad_case(case_index)
                st.rerun()
            st.markdown("<div class='trace-card-body-spacer'></div>", unsafe_allow_html=True)
            if case.evidence:
                st.write(case.evidence)
            if case.recommendation:
                st.markdown("**Recommendation**")
                st.write(case.recommendation)
            st.markdown("<div class='trace-card-bottom-spacer'></div>", unsafe_allow_html=True)


def round_history_event_title(event: dict[str, object]) -> str:
    event_id = str(event.get("event_id") or "event")
    event_type = str(event.get("type") or "event")
    timestamp = str(event.get("timestamp") or "")[:19]
    if event_type == "scan":
        count = int(event.get("badcase_count") or 0)
        status = str(event.get("status") or "unknown")
        prompt_label = str(event.get("prompt_version_label") or "prompt")
        return f"{event_id} - scan - {count} badcase(s) - {status} - {prompt_label} - {timestamp}"
    if event_type == "apply":
        count = int(event.get("approved_trace_count") or 0)
        status = str(event.get("status") or "unknown")
        before = str(event.get("before_version") or "?")
        after = str(event.get("after_version") or "?")
        return f"{event_id} - apply - {count} approved - {status} - {before} -> {after} - {timestamp}"
    if event_type == "review":
        decision = str(event.get("decision") or "reviewed")
        count = int(event.get("trace_count") or 0)
        return f"{event_id} - review - {decision} {count} trace(s) - {timestamp}"
    return f"{event_id} - {event_type} - {timestamp}"


def round_history_trace_rows(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        rows.append(
            {
                "id": case.get("id"),
                "turn": case.get("turn_index"),
                "type": case.get("error_type"),
                "source": case.get("source"),
                "applied": bool(case.get("applied")),
            }
        )
    return rows


def render_round_history() -> None:
    st.markdown("<div class='section-heading'>Round History</div>", unsafe_allow_html=True)
    history = st.session_state.get("round_history") or []
    if not history:
        st.info("No recorded rounds yet.")
        return

    st.caption(f"Persisted log: {ROUND_HISTORY_LOG.relative_to(Path(__file__).parent)}")
    st.download_button(
        "download round history",
        data=json.dumps(history, ensure_ascii=False, indent=2),
        file_name="round_history.json",
        mime="application/json",
        use_container_width=True,
    )
    for event in reversed(history[-12:]):
        with st.expander(round_history_event_title(event), expanded=False):
            event_type = str(event.get("type") or "event")
            if event_type == "scan":
                metric_cols = st.columns(3)
                metric_cols[0].metric("Badcases", int(event.get("badcase_count") or 0))
                metric_cols[1].metric("Errors", int(event.get("analysis_error_count") or 0))
                metric_cols[2].metric("Prompt hash", str(event.get("prompt_hash") or ""))
                st.caption(
                    f"View: {event.get('conversation_view') or 'unknown'} | "
                    f"Trigger: {event.get('trigger') or 'unknown'} | "
                    f"Prompt: {event.get('prompt_version_label') or 'unknown'}"
                )
                trace_rows = round_history_trace_rows(event.get("badcases") or [])
                if trace_rows:
                    st.dataframe(trace_rows, hide_index=True, use_container_width=True)
                else:
                    st.info("No badcases recorded for this scan.")
                if event.get("analysis_errors"):
                    st.markdown("**Analysis errors**")
                    st.json(event.get("analysis_errors"))
            elif event_type == "apply":
                st.caption(
                    f"Parent scan: {event.get('parent_scan_round_id') or 'none'} | "
                    f"Prompt changed: {event.get('prompt_changed')} | "
                    f"Rerun attempted: {event.get('rerun_attempted')}"
                )
                if event.get("summary"):
                    st.write(event.get("summary"))
                trace_rows = round_history_trace_rows(event.get("badcases") or [])
                if trace_rows:
                    st.dataframe(trace_rows, hide_index=True, use_container_width=True)
                diff = str(event.get("diff") or "")
                if diff:
                    st.code(diff, language="diff")
            elif event_type == "review":
                st.caption(f"Parent scan: {event.get('parent_scan_round_id') or 'none'}")
                if event.get("reason"):
                    st.write(event.get("reason"))
                trace_rows = round_history_trace_rows(event.get("badcases") or [])
                if trace_rows:
                    st.dataframe(trace_rows, hide_index=True, use_container_width=True)
            else:
                st.json(event)


def render_version_trace_list(version: dict) -> None:
    label = str(version.get("label") or "selected version")
    trace_cases = version_trace_cases(version)
    scan_recorded = bool(version.get("scan_trace_recorded"))
    trace_view = version.get("scan_conversation_view")
    trace_prompt_label = version.get("scan_source_prompt_label")
    if trace_view or trace_prompt_label:
        st.caption(
            f"Scan source: {trace_view or 'unknown conversation'} / "
            f"{trace_prompt_label or label}"
        )
    st.caption(f"{label}: scanned trace result. Historical trace view is read-only.")
    if st.button("Show current trace", use_container_width=True):
        st.session_state.trace_list_version_index = None
        st.rerun()
    if not scan_recorded:
        st.info(f"No generate badcase scan is recorded for {label}.")
        return
    if not trace_cases:
        st.info(f"No trace items found for {label}.")
        return
    for case in trace_cases:
        with st.container(border=True):
            st.markdown(
                f"<div class='trace-card-title'>Turn {case.turn_index}</div>",
                unsafe_allow_html=True,
            )
            if case.source:
                st.caption(f"Source: {case.source}")
            st.markdown("<div class='trace-card-body-spacer'></div>", unsafe_allow_html=True)
            if case.evidence:
                st.write(case.evidence)
            if case.recommendation:
                st.markdown("**Recommendation**")
                st.write(case.recommendation)
            st.markdown("<div class='trace-card-bottom-spacer'></div>", unsafe_allow_html=True)


def version_trace_cases(version: dict) -> list[BadCase]:
    cases = []
    for item in version.get("scan_trace_cases") or []:
        if not isinstance(item, dict):
            continue
        try:
            turn_index = int(item.get("turn_index"))
        except (TypeError, ValueError):
            continue
        cases.append(
            BadCase(
                turn_index=turn_index,
                role=str(item.get("role") or "assistant"),
                error_type=str(item.get("error_type") or "badcase"),
                evidence=str(item.get("evidence") or ""),
                recommendation=str(item.get("recommendation") or ""),
                source=str(item.get("source") or "version_trace"),
            )
        )
    return cases


def selected_trace_version() -> dict | None:
    index = st.session_state.get("trace_list_version_index")
    if index is None:
        return None
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    versions = st.session_state.get("prompt_versions") or []
    if not 0 <= index < len(versions):
        return None
    return versions[index]


def render_prompt_versions() -> None:
    ensure_prompt_versions(st.session_state.current_system_prompt_view)
    st.markdown("**Prompt Versions**")
    undo_disabled = len(st.session_state.prompt_versions) <= 1
    if st.button("Undo last version", disabled=undo_disabled, use_container_width=True):
        undo_prompt_version()
        st.rerun()

    version_cols = st.columns(3)
    for index, version in enumerate(st.session_state.prompt_versions):
        with version_cols[index % 3]:
            if st.button(version["label"], key=f"select_prompt_version_{index}", use_container_width=True):
                select_prompt_version(index)
                st.rerun()

    selected_index = st.session_state.selected_prompt_version_index
    if selected_index is None:
        selected_index = len(st.session_state.prompt_versions) - 1
    if 0 <= selected_index < len(st.session_state.prompt_versions):
        selected = st.session_state.prompt_versions[selected_index]
        if selected.get("diff"):
            with st.expander(f"{selected['label']} diff", expanded=True):
                st.code(selected["diff"], language="diff")


def select_prompt_version(index: int) -> None:
    st.session_state.selected_prompt_version_index = index
    st.session_state.trace_list_version_index = index
    clear_bad_case_selection()


def main() -> None:
    init_state()
    render_toasts()
    pending_optimized_prompt = st.session_state.pop("pending_optimized_prompt_editor", None)
    if pending_optimized_prompt is not None:
        st.session_state.optimized_prompt_editor = pending_optimized_prompt
    pending_current_prompt = st.session_state.pop("pending_current_system_prompt_view", None)
    if pending_current_prompt is not None:
        st.session_state.current_system_prompt_view = pending_current_prompt

    st.title("Prompt Optimizer Agent")
    data: ConversationData | None = st.session_state.data

    with st.sidebar:
        st.caption(f"Build: {APP_BUILD}")
        st.header("Input")
        uploaded_file = st.file_uploader("Upload conversation JSON", type=["json", "txt"])
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            uploaded_hash = hashlib.sha256(uploaded_bytes).hexdigest()[:16]
            uploaded_file_id = f"{uploaded_file.name}:{uploaded_file.size}:{uploaded_hash}"
            if st.session_state.last_uploaded_file_id != uploaded_file_id:
                st.session_state.raw_text = uploaded_bytes.decode("utf-8")
                st.session_state.last_uploaded_file_id = uploaded_file_id
                parse_current_json()
                data = st.session_state.data
        elif st.session_state.last_uploaded_file_id is not None:
            st.session_state.last_uploaded_file_id = None
            clear_loaded_conversation()
            data = None

        st.button("Load sample", on_click=load_sample, use_container_width=True)
        st.text_area("Raw JSON", key="raw_text", height=260)
        st.button("Check JSON", on_click=validate_current_json, use_container_width=True)

        st.header("Model")
        backend = st.radio(
            "Backend",
            options=["company_api", "openai"],
            format_func=lambda value: "Company API" if value == "company_api" else "OpenAI",
            horizontal=True,
        )
        detected = detected_company_settings(data)
        if backend == "company_api":
            company_url = st.text_input("Company API URL", value=detected["url"])
            if st.button("Refresh company models", use_container_width=True):
                refresh_company_models(company_url)
            if st.session_state.company_models_error:
                st.warning(f"Could not load model list: {st.session_state.company_models_error}")

            choices = model_choices_for_company(data, detected)
            default_choice = f"{detected['provider']}:{detected['model']}"
            default_index = choices.index(default_choice) if default_choice in choices else 0
            selected_model = st.selectbox(
                "Company model",
                options=choices,
                index=default_index,
                accept_new_options=True,
            )
            company_provider, model = split_company_model(selected_model, detected["provider"])
            company_provider = st.text_input("Provider", value=company_provider)
            max_completion_tokens = st.number_input(
                "Max completion tokens",
                min_value=64,
                max_value=32768,
                value=4096,
                step=256,
            )
            llm_settings = LLMSettings(
                backend="company_api",
                model=model,
                provider=company_provider,
                base_url=company_url,
                max_completion_tokens=int(max_completion_tokens),
            )
            if data and data.tools:
                st.caption(f"Loaded {len(data.tools)} tool definitions from JSON.")
        else:
            model = st.text_input("OpenAI model", value=os.getenv("PROMPT_OPTIMIZER_MODEL", DEFAULT_MODEL))
            llm_settings = LLMSettings(
                backend="openai",
                model=model,
            )

        api_ready = bool(os.getenv("OPENAI_API_KEY"))
        judge_backend = st.selectbox(
            "Badcase judge",
            options=["current", "openai"],
            format_func=lambda value: "Current backend" if value == "current" else "OpenAI",
            help="Use Current backend to judge with the selected Company API/OpenAI backend. Use OpenAI only when your OpenAI account has quota.",
        )
        st.session_state.judge_backend = judge_backend
        if backend == "openai":
            st.caption("OpenAI API key: configured" if api_ready else "OpenAI API key: not configured")
        elif judge_backend == "openai":
            st.caption("OpenAI judge key: configured" if api_ready else "OpenAI judge key: not configured")
        st.checkbox(
            "Auto post-rerun analysis",
            key="auto_post_rerun_analysis",
            help=(
                "When enabled, Apply also runs a full updated-conversation judge and conclusion "
                "after targeted rerun. Leave it off for the faster iterative workflow; click "
                "generate badcase when you want the next analysis round."
            ),
        )
        st.caption(f"Build: {APP_BUILD}")

    if st.session_state.last_parse_error:
        st.error(st.session_state.last_parse_error)
    if st.session_state.validation_status:
        st.success(st.session_state.validation_status)
    for warning in st.session_state.last_warnings:
        st.warning(warning)

    data = st.session_state.data
    if data is None:
        st.info("Upload or paste a JSON conversation to begin.")
        return
    if not st.session_state.current_system_prompt_view:
        st.session_state.current_system_prompt_view = data.system_prompt
    if st.session_state.pending_apply_action:
        with st.spinner("Applying and rerunning..."):
            run_pending_apply_action(
                build_current_analysis_data(data, st.session_state.current_system_prompt_view),
                llm_settings,
            )
        st.rerun()
    if st.session_state.pending_generate_badcase:
        st.session_state.pending_generate_badcase = False
        with st.spinner("Scanning bad cases..."):
            run_generate_badcase_scan(data, llm_settings)
        st.rerun()

    left_col, right_col = st.columns([0.43, 0.57], gap="large")
    with left_col:
        st.subheader("System Prompt")
        if not st.session_state.original_system_prompt_view:
            st.session_state.original_system_prompt_view = data.system_prompt
        st.text_area(
            "Original system prompt",
            value=st.session_state.original_system_prompt_view,
            height=180,
            disabled=True,
        )
        current_prompt = st.text_area(
            "Working system prompt",
            height=310,
            key="current_system_prompt_view",
        )
        if current_prompt != data.system_prompt:
            data.system_prompt = current_prompt
            st.session_state.optimization = None
            st.session_state.rerun_results = []
            st.session_state.rerun_interactions = None
            st.session_state.rerun_summary = None
            st.session_state.rerun_conclusion = None
            st.session_state.last_rerun_experiment = None

        render_prompt_versions()

        export_payload = export_with_updated_system_prompt(
            data,
            st.session_state.current_system_prompt_view,
        )

        download_col, analyze_col = st.columns([1, 1], gap="small")
        download_col.download_button(
            "download json",
            data=export_payload,
            file_name="optimized_conversation.json",
            mime="application/json",
            use_container_width=True,
        )
        analyze_clicked = analyze_col.button("generate badcase", type="primary", use_container_width=True)
        if analyze_clicked:
            st.session_state.pending_generate_badcase = True
            st.rerun()

        clear_legacy_anchor_hash("trace-list")
        st.markdown("<div id='trace-list-top'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>Trace List</div>", unsafe_allow_html=True)
        trace_version = selected_trace_version()
        render_bad_cases(
            build_current_analysis_data(data, st.session_state.current_system_prompt_view),
            llm_settings,
            st.session_state.current_system_prompt_view,
            trace_version=trace_version,
        )
        render_round_history()

    with right_col:
        conversation_view = "Original"
        visible_interactions = data.interactions
        if st.session_state.rerun_interactions:
            trace_view = active_trace_conversation_view(trace_version)
            if trace_view:
                st.session_state.conversation_view = trace_view
            if st.session_state.get("conversation_view") not in ("Updated", "Original"):
                st.session_state.conversation_view = "Updated"
            conversation_view = st.radio(
                "Conversation version",
                options=["Updated", "Original"],
                horizontal=True,
                label_visibility="collapsed",
                key="conversation_view",
                disabled=bool(trace_view),
            )
            visible_interactions = (
                st.session_state.rerun_interactions
                if conversation_view == "Updated"
                else data.interactions
            )
        if conversation_view == "Updated" and st.session_state.rerun_conclusion:
            st.subheader("Conclusion")
            with st.container(border=True):
                st.markdown(st.session_state.rerun_conclusion)

        st.subheader(f"{conversation_view} Conversation")
        if conversation_view == "Updated":
            st.caption("Showing assistant replies generated with the approved optimized prompt.")
        turn_ai_cases = active_trace_cases_by_turn(conversation_view, trace_version)
        translation_map = render_translation_controls(
            visible_interactions,
            conversation_view,
            llm_settings,
        )
        for index, turn in enumerate(visible_interactions):
            render_turn(
                data,
                llm_settings,
                index,
                turn.role,
                turn.content,
                turn_ai_cases.get(index),
                translation=translation_map.get(index),
            )

    render_toasts()

if __name__ == "__main__":
    main()
