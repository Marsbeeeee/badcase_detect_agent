import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import prompt_optimizer_agent.company_demo_client as company_demo_client
import prompt_optimizer_agent.agent_logic as agent_logic
from prompt_optimizer_agent.agent_logic import (
    AUTO_RERUN_CONTEXT_WINDOW,
    BadCase,
    LLMSettings,
    RerunTurn,
    analyze_bad_cases,
    apply_recommendation_to_system_prompt,
    _build_conclusion_payload,
    _normalize_ai_bad_case_turn_index,
    _parse_judge_bad_cases,
    _experiment_diagnostic_evidence,
    _format_conclusion_value,
    _local_prompt_rule_bad_cases,
    _prompt_edit_style_violation,
    _rerun_request_messages,
    _write_conclusion_dialog_log,
    rerun_conversation,
    translate_interactions_to_zh,
)
from prompt_optimizer_agent.company_demo_client import _prompt_hash
from prompt_optimizer_agent.json_utils import ConversationData, Interaction, parse_conversation_json


def test_parse_judge_bad_case_uses_explicit_turn_index() -> None:
    data = ConversationData(
        system_prompt="Follow flow.",
        interactions=[
            Interaction(role="user", content="hello"),
            Interaction(role="assistant", content="bad answer with account points"),
        ],
    )
    content = json.dumps(
        {
            "bad_cases": [
                {
                    "turn_index": 1,
                    "role": "assistant",
                    "error_type": "bad",
                    "evidence": "bad account points",
                    "recommendation": "fix",
                    "turn_content_excerpt": "bad answer",
                    "violated_prompt_excerpt": "Do not invent account points.",
                    "severity": "medium",
                    "confidence": 0.9,
                    "hard_violation": True,
                }
            ]
        }
    )

    cases = _parse_judge_bad_cases(data, content)

    assert len(cases) == 1
    assert cases[0].turn_index == 1
    assert "Violated rule: Do not invent account points." in cases[0].evidence


def test_parse_judge_bad_case_filters_non_hard_or_low_confidence_items() -> None:
    data = ConversationData(
        system_prompt="Follow flow.",
        interactions=[
            Interaction(role="user", content="hello"),
            Interaction(role="assistant", content="acceptable but brief"),
            Interaction(role="assistant", content="maybe bad"),
        ],
    )
    content = json.dumps(
        {
            "bad_cases": [
                {
                    "turn_index": 1,
                    "role": "assistant",
                    "error_type": "style",
                    "evidence": "Could be more polite.",
                    "recommendation": "Be warmer.",
                    "turn_content_excerpt": "acceptable",
                    "violated_prompt_excerpt": "",
                    "severity": "low",
                    "confidence": 0.9,
                    "hard_violation": False,
                },
                {
                    "turn_index": 2,
                    "role": "assistant",
                    "error_type": "uncertain",
                    "evidence": "Might have skipped a step.",
                    "recommendation": "Clarify.",
                    "turn_content_excerpt": "maybe",
                    "violated_prompt_excerpt": "Ask for clarification before advancing.",
                    "severity": "medium",
                    "confidence": 0.4,
                    "hard_violation": True,
                },
            ]
        }
    )

    assert _parse_judge_bad_cases(data, content) == []


def test_tool_wrapper_bad_case_remaps_to_natural_assistant_reply() -> None:
    turns = [
        Interaction(role="user" if index % 2 == 0 else "assistant", content=f"filler {index}")
        for index in range(14)
    ]
    turns.extend(
        [
            Interaction(role="user", content="abc123abc"),
            Interaction(
                role="assistant",
                content='<function-call>search_account_from_name:{"account_name": "abc123abc"}</function-call>',
            ),
            Interaction(role="user", content="<function-response>bukan anggota</function-response>"),
            Interaction(
                role="assistant",
                content=(
                    "Akun Anda tidak terdaftar sebagai anggota dan tidak memiliki poin. "
                    "Apakah Anda ingin mendaftar sebagai anggota?"
                ),
            ),
        ]
    )
    data = ConversationData(system_prompt="Follow flow.", interactions=turns)
    item = {
        "turn_index": 15,
        "role": "assistant",
        "error_type": "contradiction",
        "evidence": "The assistant claims the user's account is not registered as a member and has no points.",
        "recommendation": "Verify the user's account status before offering membership or points.",
    }

    assert _normalize_ai_bad_case_turn_index(data, item) == 17


def test_local_scan_flags_open_ended_payment_terms_after_rejected_today_offer() -> None:
    data = _ptp_attempt_limit_conversation()

    cases = _local_prompt_rule_bad_cases(data)

    case = next(
        item
        for item in cases
        if item.error_type == "open_ended_payment_terms_after_rejection"
    )
    assert case.turn_index == 7
    assert "Turn 6 rejected the full-payment-today proposal from turn 5" in case.evidence
    assert "Assistant turn 7 then asked the user for payment timing" in case.evidence
    assert "repeats at assistant turns 7, 9" in case.evidence
    assert "concrete fallback" in case.evidence
    assert len(case.evidence) < 700


def test_local_scan_flags_ptp_attempt_limit_with_counted_evidence() -> None:
    data = _ptp_attempt_limit_conversation()

    cases = _local_prompt_rule_bad_cases(data)

    case = next(
        item
        for item in cases
        if item.error_type == "ptp_attempt_limit_exceeded"
    )
    assert case.turn_index == 9
    assert "failed proposal-collection count reached 3" in case.evidence
    assert "user turns 4, 6, 8" in case.evidence
    assert "State 4.0 (RTP_Closing)" in case.evidence
    assert len(case.evidence) < 650


def test_local_scan_flags_missing_fresh_promotion_search() -> None:
    data = ConversationData(
        system_prompt=(
            "Customer Name Gate: You must know the customer's name before processing any request. "
            "If they stated a request earlier, immediately call the required tool for that request "
            "once they provide their name.\n"
            "Inquiry Handling: Promotions or discounts -> MandiriCX_Call_Center_search_promotion. "
            "Every message requires a fresh search, even if you already know the answer from earlier results."
        ),
        interactions=[
            Interaction(role="user", content="promo personal loan dua miliar itu apa"),
            Interaction(role="assistant", content="<function-call>language_detection:{\"language\":\"Bahasa\"}</function-call>"),
            Interaction(role="tool", content="{'status': 'ok'}"),
            Interaction(role="assistant", content="Boleh saya tahu dengan siapa saya berbicara?"),
            Interaction(role="user", content="dengan ibu almah"),
            Interaction(role="assistant", content="Mohon tunggu, saya carikan informasinya."),
            Interaction(role="assistant", content="Limit pinjaman tersedia hingga tiga ratus lima puluh juta rupiah."),
            Interaction(role="user", content="bukannya dua miliar ya"),
            Interaction(role="assistant", content="Untuk nasabah prioritas, limitnya tujuh ratus lima puluh juta rupiah."),
            Interaction(role="user", content="promosi nikmati kemudahan personal loan dua miliar"),
            Interaction(
                role="assistant",
                content=(
                    "<function-call>MandiriCX_Call_Center_search_promotion:"
                    "{\"query\":\"promo personal loan dua miliar\"}</function-call>"
                ),
            ),
            Interaction(role="tool", content="{\"promotions\":\"Personal loan hingga Rp2 Miliar.\"}"),
            Interaction(role="assistant", content="Program ini menawarkan personal loan hingga dua miliar rupiah."),
        ],
    )

    cases = [
        case
        for case in _local_prompt_rule_bad_cases(data)
        if case.error_type == "missing_fresh_promotion_search"
    ]

    assert [case.turn_index for case in cases] == [6, 8]
    assert "User turn 0 asked or followed up on a promotion after the name gate closed at user turn 4" in cases[0].evidence
    assert "No search_promotion tool call appeared before assistant turn 6" in cases[0].evidence
    assert "User turn 7 asked or followed up on a promotion" in cases[1].evidence


def test_analyze_bad_cases_dedupes_ai_and_local_same_turn_tool_failure() -> None:
    old_chat_json = agent_logic._chat_json

    def fake_chat_json(settings, messages, purpose="json"):
        return json.dumps(
            {
                "bad_cases": [
                    {
                        "turn_index": 1,
                        "role": "assistant",
                        "error_type": "missing_tool_call",
                        "evidence": (
                            "The assistant answered a promotion question without a fresh "
                            "search_promotion tool call. This duplicates the local scan."
                        ),
                        "recommendation": "Call search_promotion before answering.",
                        "turn_content_excerpt": "Program ini untuk nasabah prioritas.",
                        "violated_prompt_excerpt": "Every promotion message requires fresh search.",
                        "severity": "high",
                        "confidence": 0.95,
                        "hard_violation": True,
                    }
                ]
            }
        )

    agent_logic._chat_json = fake_chat_json
    try:
        data = ConversationData(
            system_prompt=(
                "For promotion inquiries, every customer message requires a fresh search. "
                "You must call search_promotion before answering any promotion, discount, or banking-service facts."
            ),
            interactions=[
                Interaction(role="user", content="Ada promo personal loan?"),
                Interaction(
                    role="assistant",
                    content="Program personal loan ini untuk nasabah prioritas dengan dana minimum satu miliar.",
                ),
            ],
        )
        cases = analyze_bad_cases(
            data=data,
            llm_settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
        )
    finally:
        agent_logic._chat_json = old_chat_json

    assert len(cases) == 1
    assert cases[0].turn_index == 1
    assert cases[0].error_type == "missing_fresh_promotion_search"
    assert cases[0].source == "local_scan"


def test_translate_interactions_to_zh_parses_turn_indexed_response() -> None:
    old_chat_json = agent_logic._chat_json
    captured_payload = {}

    def fake_chat_json(settings, messages, purpose="json"):
        captured_payload["purpose"] = purpose
        captured_payload["payload"] = json.loads(messages[1]["content"])
        return json.dumps(
            {
                "translations": [
                    {"turn_index": 0, "translation": "你好"},
                    {"turn_index": 1, "translation": "助手回复"},
                ]
            },
            ensure_ascii=False,
        )

    agent_logic._chat_json = fake_chat_json
    try:
        translations = translate_interactions_to_zh(
            [
                Interaction(role="user", content="hello"),
                Interaction(role="assistant", content="assistant reply"),
            ],
            llm_settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
        )
    finally:
        agent_logic._chat_json = old_chat_json

    assert captured_payload["purpose"] == "translation"
    assert captured_payload["payload"]["target_language"] == "Simplified Chinese"
    assert translations == {0: "你好", 1: "助手回复"}


def _ptp_attempt_limit_conversation() -> ConversationData:
    system_prompt = (
        "5.3 Prohibited Negotiation Actions:\n"
        "- NEVER ask the user to suggest an amount or date. You must always lead with proposals.\n"
        "- Under NO circumstances should you ask open-ended questions about payment terms. NEVER ask "
        "\"how much can you pay?\" or \"when can you pay?\". Your role is to ALWAYS propose concrete, "
        "specific terms. If one proposal is rejected, you must move to the next logical, pre-defined proposal.\n\n"
        "2.2 Clarify & Collect the User's Proposal\n"
        "You have a maximum of three attempts to clarify and collect the complete proposal.\n"
        "If the 3-attempt limit exceeded, do not prompt the user again and must immediately proceed to "
        "State 4.0 (RTP_Closing)."
    )
    return ConversationData(
        system_prompt=system_prompt,
        interactions=[
            Interaction(role="user", content="[Conversation Begins]"),
            Interaction(role="assistant", content="Hello! Am I speaking with John Tan?"),
            Interaction(role="user", content="yes it is"),
            Interaction(
                role="assistant",
                content=(
                    "You have 5 overdue loans with Grab Finance. May I know why the payment hasn't "
                    "been made yet?"
                ),
            ),
            Interaction(
                role="user",
                content="I'm not paying. I don't have the money and I don't plan to pay anytime soon.",
            ),
            Interaction(
                role="assistant",
                content=(
                    "I understand. Would you be able to make a full payment of 750 ringgit and "
                    "41 cents today to get this resolved?"
                ),
            ),
            Interaction(role="user", content="No, I told you already. I can't."),
            Interaction(
                role="assistant",
                content=(
                    "Could you please let me know when you might be able to make a payment?"
                ),
            ),
            Interaction(
                role="user",
                content="Still no. I just can't make any payments right now.",
            ),
            Interaction(
                role="assistant",
                content=(
                    "Could you please provide a specific date when you might be able to make a payment?"
                ),
            ),
        ],
    )


def test_rerun_conversation_blocks_on_preflight_prompt_mismatch() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        old_audit_path = company_demo_client.DEFAULT_AUDIT_PATH
        old_preflight_path = company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH
        company_demo_client.DEFAULT_AUDIT_PATH = Path(temp_dir) / "request.jsonl"
        company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH = Path(temp_dir) / "preflight.log"
        try:
            data = ConversationData(
                system_prompt="new prompt",
                interactions=[
                    Interaction(role="user", content="hello"),
                    Interaction(role="assistant", content="old answer"),
                ],
            )
            results = rerun_conversation(
                data=data,
                optimized_prompt="new prompt",
                llm_settings=LLMSettings(
                    backend="company_api",
                    model="m",
                    provider="p",
                    base_url="http://example.test",
                ),
                target_assistant_turn_indices={1},
                expected_system_prompt_hash=_prompt_hash("different prompt"),
                expected_prompt_version="v2 Apply",
            )

            assert len(results) == 1
            assert results[0].error is not None
            assert "blocked before send" in results[0].error
            assert "match=False" in company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH.read_text()
            assert not company_demo_client.DEFAULT_AUDIT_PATH.exists()
        finally:
            company_demo_client.DEFAULT_AUDIT_PATH = old_audit_path
            company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH = old_preflight_path


def test_openai_chat_text_passes_tools_and_renders_tool_call() -> None:
    old_openai = agent_logic.OpenAI
    captured_request = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name="MandiriCX_Call_Center_search_promotion",
                                        arguments='{"query":"personal loan"}',
                                    )
                                )
                            ],
                        )
                    )
                ]
            )

    class FakeClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    agent_logic.OpenAI = FakeClient
    try:
        content = agent_logic._chat_text(
            settings=LLMSettings(backend="openai", model="test-model"),
            messages=[
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Any promo?"},
            ],
            tools={
                "MandiriCX_Call_Center_search_promotion": {
                    "type": "function",
                    "function": {
                        "name": "MandiriCX_Call_Center_search_promotion",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            },
            purpose="targeted_rerun",
        )
    finally:
        agent_logic.OpenAI = old_openai

    assert captured_request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "MandiriCX_Call_Center_search_promotion",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert captured_request["tool_choice"] == "auto"
    assert content == '<function-call>MandiriCX_Call_Center_search_promotion:{"query":"personal loan"}</function-call>'


def test_rerun_conversation_passes_normalized_upload_tools() -> None:
    parsed = parse_conversation_json(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "Use tools."},
                    {"role": "user", "content": "Any promo?"},
                    {"role": "assistant", "content": "Old natural answer."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "MandiriCX_Call_Center_search_promotion",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )
    assert parsed.data is not None
    old_preflight = agent_logic._preflight_company_rerun_request
    old_chat_text = agent_logic._chat_text
    captured = {}

    def fake_preflight(**kwargs):
        captured["preflight_tools"] = kwargs["tools"]

    def fake_chat_text(**kwargs):
        captured["chat_tools"] = kwargs["tools"]
        return '<function-call>MandiriCX_Call_Center_search_promotion:{"query":"promo"}</function-call>'

    agent_logic._preflight_company_rerun_request = fake_preflight
    agent_logic._chat_text = fake_chat_text
    try:
        results = rerun_conversation(
            data=parsed.data,
            optimized_prompt=parsed.data.system_prompt,
            llm_settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
            target_assistant_turn_indices={1},
        )
    finally:
        agent_logic._preflight_company_rerun_request = old_preflight
        agent_logic._chat_text = old_chat_text

    assert captured["preflight_tools"] == parsed.data.tools
    assert captured["chat_tools"] == parsed.data.tools
    assert results[0].new_assistant_response.startswith("<function-call>")


def test_rerun_conversation_forces_required_tool_when_model_answers_text() -> None:
    data = ConversationData(
        system_prompt="Use tools.",
        interactions=[
            Interaction(role="user", content="promo personal loan"),
            Interaction(role="assistant", content="old natural answer"),
        ],
        tools={
            "MandiriCX_Call_Center_search_promotion": {
                "type": "function",
                "function": {
                    "name": "MandiriCX_Call_Center_search_promotion",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        },
    )
    old_preflight = agent_logic._preflight_company_rerun_request
    old_chat_text = agent_logic._chat_text
    captured = {}

    def fake_preflight(**kwargs):
        captured["tool_choice"] = kwargs["tool_choice"]
        captured["messages"] = kwargs["messages"]

    def fake_chat_text(**kwargs):
        captured["chat_tool_choice"] = kwargs["tool_choice"]
        return "Program ini punya limit khusus untuk nasabah prioritas."

    agent_logic._preflight_company_rerun_request = fake_preflight
    agent_logic._chat_text = fake_chat_text
    try:
        results = rerun_conversation(
            data=data,
            optimized_prompt=data.system_prompt,
            llm_settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
            target_assistant_turn_indices={1},
            required_tools_by_turn={1: "MandiriCX_Call_Center_search_promotion"},
        )
    finally:
        agent_logic._preflight_company_rerun_request = old_preflight
        agent_logic._chat_text = old_chat_text

    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "MandiriCX_Call_Center_search_promotion"},
    }
    assert captured["chat_tool_choice"] == captured["tool_choice"]
    assert captured["messages"][-1]["role"] == "system"
    assert "MUST call" in captured["messages"][-1]["content"]
    assert results[0].new_assistant_response == (
        '<function-call>MandiriCX_Call_Center_search_promotion:{"query": "promo personal loan"}</function-call>'
    )
    assert results[0].response_diagnostics["forced_required_tool_call"]["tool_name"] == (
        "MandiriCX_Call_Center_search_promotion"
    )


def test_targeted_rerun_handles_consecutive_assistant_turn_with_required_tool() -> None:
    data = ConversationData(
        system_prompt="Use tools.",
        interactions=[
            Interaction(role="user", content="Saya mau tanya promo personal loan."),
            Interaction(role="assistant", content="Boleh saya tahu nama Ibu?"),
            Interaction(role="user", content="Almah"),
            Interaction(role="assistant", content="Baik, Ibu Almah. Mohon tunggu sebentar."),
            Interaction(role="assistant", content="Old factual promotion answer with Pinjaman Serbaguna limit."),
        ],
        tools={
            "MandiriCX_Call_Center_search_promotion": {
                "type": "function",
                "function": {
                    "name": "MandiriCX_Call_Center_search_promotion",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        },
    )
    old_preflight = agent_logic._preflight_company_rerun_request
    old_chat_text = agent_logic._chat_text
    captured = {}

    def fake_preflight(**kwargs):
        captured["messages"] = kwargs["messages"]
        captured["tool_choice"] = kwargs["tool_choice"]

    def fake_chat_text(**kwargs):
        return "Old-style natural answer."

    agent_logic._preflight_company_rerun_request = fake_preflight
    agent_logic._chat_text = fake_chat_text
    try:
        results = rerun_conversation(
            data=data,
            optimized_prompt=data.system_prompt,
            llm_settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
            target_assistant_turn_indices={4},
            required_tools_by_turn={4: "MandiriCX_Call_Center_search_promotion"},
        )
    finally:
        agent_logic._preflight_company_rerun_request = old_preflight
        agent_logic._chat_text = old_chat_text

    assert len(results) == 1
    assert results[0].user_turn_index == 2
    assert results[0].assistant_turn_index == 4
    assert results[0].old_assistant_response == "Old factual promotion answer with Pinjaman Serbaguna limit."
    assert "promo personal loan" in results[0].new_assistant_response
    assert "Pinjaman Serbaguna" in results[0].new_assistant_response
    assert captured["messages"][1:5] == [
        {"role": "user", "content": "Saya mau tanya promo personal loan."},
        {"role": "assistant", "content": "Boleh saya tahu nama Ibu?"},
        {"role": "user", "content": "Almah"},
        {"role": "assistant", "content": "Baik, Ibu Almah. Mohon tunggu sebentar."},
    ]
    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "MandiriCX_Call_Center_search_promotion"},
    }


def test_prompt_edit_uses_exact_replace_patch() -> None:
    old_chat_json = agent_logic._chat_json
    calls = []

    def fake_chat_json(settings, messages, purpose="json"):
        calls.append(purpose)
        return json.dumps(
            {
                "replace": {
                    "old": "Step 2: Ask for a phone number.",
                    "new": "Step 2: Ask for a phone number or account name.",
                },
                "rationale": "Added account-name fallback.",
                "applied_feedback_summary": "Patched Step 2.",
            }
        )

    agent_logic._chat_json = fake_chat_json
    try:
        data = ConversationData(
            system_prompt="Step 1: Greet.\nStep 2: Ask for a phone number.",
            interactions=[Interaction(role="assistant", content="bad")],
        )
        result = apply_recommendation_to_system_prompt(
            data=data,
            current_system_prompt=data.system_prompt,
            bad_case=BadCase(
                turn_index=0,
                role="assistant",
                error_type="missing_alternative",
                evidence="Only asks for phone.",
                recommendation="Allow account name as fallback.",
            ),
            llm_settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
        )

        assert result.optimized_prompt == "Step 1: Greet.\nStep 2: Ask for a phone number or account name."
        assert calls == ["prompt_edit_patch"]
    finally:
        agent_logic._chat_json = old_chat_json


def test_prompt_edit_falls_back_to_full_prompt_when_patch_cannot_apply() -> None:
    old_chat_json = agent_logic._chat_json
    calls = []

    def fake_chat_json(settings, messages, purpose="json"):
        calls.append(purpose)
        if purpose == "prompt_edit_patch":
            return json.dumps({"replace": {"old": "missing text", "new": "replacement"}})
        return json.dumps(
            {
                "optimized_prompt": "Step 1: Greet.\nStep 2: Ask for a phone number or account name.",
                "rationale": "Full fallback.",
                "applied_feedback_summary": "Updated Step 2.",
            }
        )

    agent_logic._chat_json = fake_chat_json
    try:
        data = ConversationData(
            system_prompt="Step 1: Greet.\nStep 2: Ask for a phone number.",
            interactions=[Interaction(role="assistant", content="bad")],
        )
        result = apply_recommendation_to_system_prompt(
            data=data,
            current_system_prompt=data.system_prompt,
            bad_case=BadCase(
                turn_index=0,
                role="assistant",
                error_type="missing_alternative",
                evidence="Only asks for phone.",
                recommendation="Allow account name as fallback.",
            ),
            llm_settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
        )

        assert result.optimized_prompt.endswith("account name.")
        assert calls == ["prompt_edit_patch", "prompt_edit"]
    finally:
        agent_logic._chat_json = old_chat_json


def test_company_chat_json_retries_non_json_response() -> None:
    old_chat_text = agent_logic._chat_text
    calls = []

    def fake_chat_text(**kwargs):
        calls.append(kwargs["purpose"])
        if kwargs["purpose"] == "prompt_edit":
            return "I updated the prompt, but forgot JSON."
        assert kwargs["purpose"] == "prompt_edit_json_retry"
        return json.dumps(
            {
                "optimized_prompt": "Step 1: Greet.\nStep 2: Close.",
                "rationale": "Retry returned valid JSON.",
                "applied_feedback_summary": "Updated Step 2.",
            }
        )

    agent_logic._chat_text = fake_chat_text
    try:
        content = agent_logic._chat_json(
            settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
            messages=[
                {"role": "system", "content": "Return JSON only with key optimized_prompt."},
                {"role": "user", "content": "{}"},
            ],
            purpose="prompt_edit",
        )
    finally:
        agent_logic._chat_text = old_chat_text

    parsed = json.loads(content)
    assert parsed["optimized_prompt"].endswith("Close.")
    assert calls == ["prompt_edit", "prompt_edit_json_retry"]


def test_extract_json_object_escapes_raw_newlines_inside_strings() -> None:
    content = '{"optimized_prompt": "Step 1: Greet.\nStep 2: Close.", "rationale": "ok"}'

    extracted = agent_logic._extract_json_object(content)

    parsed = json.loads(extracted)
    assert parsed["optimized_prompt"] == "Step 1: Greet.\nStep 2: Close."


def test_repair_json_response_completes_truncated_string_and_preserves_fields() -> None:
    content = (
        '```json\n{"optimized_prompt":"line one\nline two",'
        '"rationale":"patched the branch",'
        '"applied_feedback_summary":"changed'
    )

    repair = agent_logic._repair_json_response(content)

    assert repair is not None
    parsed = json.loads(repair.repaired_json)
    assert parsed["optimized_prompt"] == "line one\nline two"
    assert parsed["rationale"] == "patched the branch"
    assert parsed["applied_feedback_summary"] == "changed"
    assert repair.original_text == content
    assert repair.method == "complete_truncated_object"


def test_json_repair_log_preserves_original_and_repaired_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent_logic, "PROJECT_ROOT", tmp_path)
    repair = agent_logic.JsonRepairResult(
        repaired_json='{"ok": true}',
        method="unit_test",
        original_text='model said {"ok": true',
        repaired_text='{"ok": true}',
    )

    agent_logic._log_json_repair_attempt(
        purpose="prompt_edit",
        model="test-model",
        provider="test-provider",
        stage="initial",
        error="bad json",
        repair=repair,
    )

    log_path = tmp_path / "logs" / "json_repair_attempts.jsonl"
    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["event"] == "json_repair_success"
    assert payload["original_text"] == 'model said {"ok": true'
    assert payload["repaired_text"] == '{"ok": true}'
    assert payload["repaired_json"] == '{"ok": true}'


def test_prompt_edit_retries_as_patch_when_full_json_is_malformed() -> None:
    old_chat_json = agent_logic._chat_json
    calls = []

    def fake_chat_json(settings, messages, purpose="json"):
        calls.append(purpose)
        if purpose == "prompt_edit_patch":
            return json.dumps({"replace": {"old": "missing text", "new": "replacement"}})
        if purpose == "prompt_edit":
            return '{"optimized_prompt": "Step 1: Greet.\nStep 2: Ask for account name.'
        if purpose == "prompt_edit_json_patch_retry":
            return json.dumps(
                {
                    "replace": {
                        "old": "Step 2: Ask for a phone number.",
                        "new": "Step 2: Ask for a phone number or account name.",
                    },
                    "rationale": "Retried as small patch.",
                    "applied_feedback_summary": "Patched Step 2.",
                }
            )
        raise AssertionError(f"unexpected purpose {purpose}")

    agent_logic._chat_json = fake_chat_json
    try:
        data = ConversationData(
            system_prompt="Step 1: Greet.\nStep 2: Ask for a phone number.",
            interactions=[Interaction(role="assistant", content="bad")],
        )
        result = apply_recommendation_to_system_prompt(
            data=data,
            current_system_prompt=data.system_prompt,
            bad_case=BadCase(
                turn_index=0,
                role="assistant",
                error_type="missing_alternative",
                evidence="Only asks for phone.",
                recommendation="Allow account name as fallback.",
            ),
            llm_settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
        )
    finally:
        agent_logic._chat_json = old_chat_json

    assert result.optimized_prompt == "Step 1: Greet.\nStep 2: Ask for a phone number or account name."
    assert calls == ["prompt_edit_patch", "prompt_edit", "prompt_edit_json_patch_retry"]


def test_prompt_edit_retries_overlong_branch_paragraph() -> None:
    old_chat_json = agent_logic._chat_json
    calls = []
    current_prompt = (
        "Step 1: Greet.\n"
        "Step 3: Confirm Arrival Time\n"
        "Ask for the time Julia plans to arrive."
    )
    long_branch = (
        "Step 1: Greet.\n"
        "Step 3: Confirm Arrival Time\n"
        "Ask for the time Julia plans to arrive. If Julia provides the arrival time, confirm this "
        "time along with the event date. If Julia refuses to give the time, ask what Julia is "
        "concerned about. No matter what Julia replies, end the conversation. If Julia is not sure "
        "when to come, tell her she can call back as soon as she decides. Alternatively, offer to "
        "follow up with her later to confirm attendance and arrival time. If Julia is not sure "
        "whether to attend, gently persuade her by highlighting the event benefits or offering to "
        "follow up later. If Julia is unsure, gently persuade her by highlighting the event benefits "
        "or offering to follow up later."
    )
    structured_prompt = (
        "Step 1: Greet.\n"
        "Step 3: Confirm Arrival Time\n"
        "Ask for the time Julia plans to arrive.\n"
        "3.1 If Julia provides an arrival time, confirm it with the event date, call record_time, then end politely.\n"
        "3.2 If Julia refuses to give a time, ask what she is concerned about, then end politely after her reply.\n"
        "3.3 If Julia is unsure when to arrive, offer a later follow-up or preferred contact method."
    )

    def fake_chat_json(settings, messages, purpose="json"):
        calls.append(purpose)
        if purpose == "prompt_edit_patch":
            return json.dumps({"replace": {"old": "missing text", "new": "replacement"}})
        if purpose == "prompt_edit":
            return json.dumps({"optimized_prompt": long_branch})
        if purpose == "prompt_edit_retry":
            return json.dumps({"optimized_prompt": structured_prompt})
        raise AssertionError(f"unexpected purpose {purpose}")

    agent_logic._chat_json = fake_chat_json
    try:
        data = ConversationData(
            system_prompt=current_prompt,
            interactions=[
                Interaction(role="user", content="Saya belum yakin."),
                Interaction(role="assistant", content="What time will you arrive?"),
            ],
        )
        result = apply_recommendation_to_system_prompt(
            data=data,
            current_system_prompt=current_prompt,
            bad_case=BadCase(
                turn_index=1,
                role="assistant",
                error_type="advanced_on_uncertain_reply",
                evidence="Assistant asked arrival time even though user was unsure.",
                recommendation="Add an explicit branch for uncertain attendance or arrival time.",
            ),
            llm_settings=LLMSettings(
                backend="company_api",
                model="m",
                provider="p",
                base_url="http://example.test",
            ),
        )

        assert result.optimized_prompt == structured_prompt
        assert calls == ["prompt_edit_patch", "prompt_edit", "prompt_edit_retry"]
    finally:
        agent_logic._chat_json = old_chat_json


def test_prompt_edit_style_violation_detects_long_repeated_branch() -> None:
    original_prompt = "Step 3: Confirm Arrival Time\nAsk for the time Julia plans to arrive."
    optimized_prompt = (
        original_prompt
        + "\nIf Julia is not sure when to come, offer to follow up later to confirm attendance "
        "and arrival time. If Julia is not sure whether to attend, offer to follow up later to "
        "confirm attendance and arrival time. If Julia is unsure, offer to follow up later to "
        "confirm attendance and arrival time. Alternatively, offer to follow up later to confirm "
        "attendance and arrival time, or ask for the preferred method of contact."
    )

    assert _prompt_edit_style_violation(original_prompt, optimized_prompt) is not None


def test_rerun_request_messages_uses_recent_window() -> None:
    replay_messages = [{"role": "system", "content": "prompt"}] + [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"m{index}"}
        for index in range(8)
    ]

    request_messages = _rerun_request_messages(replay_messages, 3)

    assert request_messages == [
        {"role": "system", "content": "prompt"},
        {"role": "assistant", "content": "m5"},
        {"role": "user", "content": "m6"},
        {"role": "assistant", "content": "m7"},
    ]


def test_rerun_request_messages_auto_keeps_short_context_full() -> None:
    replay_messages = [{"role": "system", "content": "prompt"}] + [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"m{index}"}
        for index in range(4)
    ]

    assert _rerun_request_messages(replay_messages, AUTO_RERUN_CONTEXT_WINDOW) == replay_messages


def test_rerun_request_messages_auto_trims_long_context() -> None:
    replay_messages = [{"role": "system", "content": "prompt"}] + [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 3000}
        for index in range(12)
    ]

    request_messages = _rerun_request_messages(replay_messages, AUTO_RERUN_CONTEXT_WINDOW)

    assert request_messages[0] == {"role": "system", "content": "prompt"}
    assert request_messages[1:] == replay_messages[-6:]


def test_format_conclusion_value_normalizes_list_to_markdown() -> None:
    conclusion = _format_conclusion_value(["Cause.", "Fix.", "Residual issue."])

    assert conclusion == "Cause.\n\nFix.\n\nResidual issue."


def test_format_conclusion_value_normalizes_structured_sections() -> None:
    conclusion = _format_conclusion_value(
        {
            "modifications": ["Added an unsure-user branch."],
            "improvement_evidence": {
                "verdict": "部分改善",
                "evidence": "The new reply acknowledges uncertainty.",
            },
            "next_steps": "Run generate badcase to scan residual issues.",
        }
    )

    paragraphs = conclusion.split("\n\n")
    assert len(paragraphs) == 3
    assert "Added an unsure-user branch." in paragraphs[0]
    assert "verdict: 部分改善" in paragraphs[1]
    assert "Run generate badcase" in conclusion


def test_format_conclusion_value_accepts_numbered_paragraph_fields() -> None:
    conclusion = _format_conclusion_value(
        {
            "paragraph_1": "Changed Step 2.2.",
            "paragraph_2": "Wording changed, but behavior did not improve.",
            "paragraph_3": "Replace abstract persuasion with concrete benefits.",
        }
    )

    assert conclusion == (
        "Changed Step 2.2.\n\n"
        "Wording changed, but behavior did not improve.\n\n"
        "Replace abstract persuasion with concrete benefits."
    )


def test_format_conclusion_value_strips_headings_and_bullets() -> None:
    conclusion = _format_conclusion_value(
        "### 1. Previous change\n"
        "- Added a fallback branch.\n\n"
        "### 2. Improvement evidence\n"
        "- The rerun changed the target behavior.\n\n"
        "### 3. Next step\n"
        "- Run generate badcase again."
    )

    assert conclusion == (
        "Added a fallback branch.\n\n"
        "The rerun changed the target behavior.\n\n"
        "Run generate badcase again."
    )


def test_format_conclusion_value_splits_two_paragraph_output() -> None:
    conclusion = _format_conclusion_value(
        "The prompt changed Step 2.2. The rerun changed wording but not behavior.\n\n"
        "Replace the abstract instruction with concrete event benefits."
    )

    paragraphs = conclusion.split("\n\n")
    assert len(paragraphs) == 3
    assert paragraphs[0] == "The prompt changed Step 2.2."
    assert paragraphs[1] == "The rerun changed wording but not behavior."
    assert paragraphs[2] == "Replace the abstract instruction with concrete event benefits."


def test_experiment_diagnostic_evidence_exposes_non_surface_signals() -> None:
    data = ConversationData(
        system_prompt="Follow flow.",
        interactions=[Interaction(role="user", content="Hi")],
        source_meta={"model_info": {"provider": "openai_api_like", "model": "m"}},
        tools={"search_account": {"type": "function"}},
    )
    diagnostics = _experiment_diagnostic_evidence(
        data=data,
        rerun_results=[
            RerunTurn(
                user_turn_index=0,
                assistant_turn_index=1,
                user_message="abc",
                old_assistant_response='<function-call>search_account:{}</function-call>',
                new_assistant_response="I can help with another lookup method.",
            )
        ],
        post_rerun_bad_cases=[
            BadCase(
                turn_index=1,
                role="assistant",
                error_type="missing_fallback",
                evidence="Still no fallback.",
                recommendation="Add fallback.",
            )
        ],
        post_rerun_scan_status="completed",
    )

    assert diagnostics["token_probability"]["available"] is False
    assert diagnostics["metadata"]["tools_available"] is True
    assert diagnostics["metadata"]["tool_names"] == ["search_account"]
    assert diagnostics["behavior_change"]["old_response_contains_function_call"] is True
    assert diagnostics["behavior_change"]["new_response_contains_function_call"] is False
    assert diagnostics["post_rerun_scan"]["residual_badcase_count"] == 1


def test_experiment_diagnostic_evidence_uses_available_logprobs() -> None:
    data = ConversationData(
        system_prompt="Follow flow.",
        interactions=[Interaction(role="user", content="Hi")],
    )
    diagnostics = _experiment_diagnostic_evidence(
        data=data,
        rerun_results=[
            RerunTurn(
                user_turn_index=0,
                assistant_turn_index=1,
                user_message="Hi",
                old_assistant_response="old",
                new_assistant_response="new",
                response_diagnostics={
                    "logprobs": {
                        "available": True,
                        "avg_logprob": -0.3,
                        "min_logprob": -1.2,
                        "low_confidence_tokens": [{"token": "new", "logprob": -1.2}],
                    }
                },
            )
        ],
        post_rerun_bad_cases=[],
        post_rerun_scan_status="completed",
    )

    assert diagnostics["token_probability"]["available"] is True
    assert diagnostics["token_probability"]["avg_logprob_by_turn"] == [-0.3]
    assert diagnostics["token_probability"]["low_confidence_tokens"][0]["token"] == "new"


def test_experiment_diagnostic_evidence_tolerates_legacy_rerun_results() -> None:
    data = ConversationData(
        system_prompt="Follow flow.",
        interactions=[Interaction(role="user", content="Hi")],
    )
    legacy_result = SimpleNamespace(
        user_turn_index=0,
        assistant_turn_index=1,
        user_message="Hi",
        old_assistant_response="old",
        new_assistant_response="new",
        error=None,
    )

    diagnostics = _experiment_diagnostic_evidence(
        data=data,
        rerun_results=[legacy_result],
        post_rerun_bad_cases=[],
        post_rerun_scan_status="completed",
    )

    assert diagnostics["behavior_change"]["changed_turn_count"] == 1
    assert diagnostics["token_probability"]["available"] is False


def test_build_conclusion_payload_structures_paragraph_evidence() -> None:
    before_prompt = "Step 2.2 If Julia is unsure, persuade her to attend."
    optimized_prompt = (
        "Step 2.2 If Julia is unsure, persuade her to attend. "
        "Persuade Julia to attend the event. Persuade Julia to attend the event."
    )
    payload = _build_conclusion_payload(
        data=ConversationData(
            system_prompt=before_prompt,
            interactions=[Interaction(role="user", content="Unsure.")],
        ),
        before_prompt=before_prompt,
        optimized_prompt=optimized_prompt,
        rerun_results=[
            RerunTurn(
                user_turn_index=0,
                assistant_turn_index=1,
                user_message="Unsure.",
                old_assistant_response="Please call us when you decide.",
                new_assistant_response="Please contact us again once you decide.",
                response_diagnostics={
                    "logprobs": {
                        "requested": True,
                        "available": False,
                        "reason": "No logprobs.",
                    }
                },
            )
        ],
        updated_interactions=[Interaction(role="user", content="Unsure.")],
        post_rerun_bad_cases=[],
        applied_feedback_summary="Repeated the persuasion rule.",
        post_rerun_scan_status="not_run",
    )

    paragraph_inputs = payload["paragraph_inputs"]
    change_quality = paragraph_inputs["paragraph_1_changes"]["prompt_change_quality"]
    comparison = paragraph_inputs["paragraph_2_improvement_evidence"]["target_turn_comparisons"][0]

    assert payload["output_contract"]["format"]["conclusion"]["paragraph_1"]
    assert change_quality["has_repeated_added_instruction"] is True
    assert comparison["exact_text_changed"] is True
    assert comparison["approximate_similarity"] >= 0.5
    assert any(
        "token-probability" in hint
        for hint in paragraph_inputs["paragraph_3_next_recommendation"]["observed_root_cause_hints"]
    )


def test_build_conclusion_payload_marks_rerun_only_when_prompt_unchanged() -> None:
    prompt = "Every promotion message requires search_promotion before answering."
    payload = _build_conclusion_payload(
        data=ConversationData(
            system_prompt=prompt,
            interactions=[Interaction(role="user", content="Any promo?")],
        ),
        before_prompt=prompt,
        optimized_prompt=prompt,
        rerun_results=[
            RerunTurn(
                user_turn_index=0,
                assistant_turn_index=1,
                user_message="Any promo?",
                old_assistant_response="Here are promotion details.",
                new_assistant_response='<function-call>search_promotion:{"query":"Any promo?"}</function-call>',
            )
        ],
        updated_interactions=[Interaction(role="user", content="Any promo?")],
        post_rerun_bad_cases=[],
        applied_feedback_summary="No new system-prompt version was needed; reran the target assistant turn.",
        post_rerun_scan_status="not_run",
    )

    changes = payload["paragraph_inputs"]["paragraph_1_changes"]

    assert changes["prompt_changed"] is False
    assert changes["prompt_diff"] == ""
    assert "conversation-only" in changes["instruction"]


def test_write_conclusion_dialog_log_exports_compress_dialog() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "conclusion_dialog.jsonl"

        _write_conclusion_dialog_log(
            [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": '{"before_prompt": "old"}'},
            ],
            path=output_path,
        )

        record = json.loads(output_path.read_text(encoding="utf-8"))

    assert record["type"] == "compress"
    assert record["tools"] == {}
    assert [turn["role"] for turn in record["dialog"]] == ["system", "user", "assistant"]
    assert record["dialog"][0]["content"] == "Return JSON only."
    assert record["dialog"][1]["content"] == '{"before_prompt": "old"}'
    assert "<|loss_start|>" in record["dialog"][2]["content"]
    assert "<|loss_end|>" in record["dialog"][2]["content"]
