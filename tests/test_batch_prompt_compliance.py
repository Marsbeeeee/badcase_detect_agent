from prompt_optimizer_agent.json_utils import ConversationData, Interaction
from tools.batch_prompt_compliance import (
    apply_prompt_cases_with_llm,
    build_verification_failures,
    classify_prompt_edit_failure,
    failure_category_counts,
    format_failure_category_counts_zh,
    required_tool_for_case,
)


def conversation_with_transfer_tool() -> ConversationData:
    return ConversationData(
        system_prompt="Test prompt.",
        interactions=[Interaction(role="user", content="hello")],
        tools={
            "transfer_to_human": {
                "type": "function",
                "function": {"name": "transfer_to_human"},
            }
        },
    )


def test_required_tool_for_case_ignores_exact_message_escalation() -> None:
    case = {
        "turn_index": 5,
        "error_type": "escalation_action_not_followed",
        "evidence": (
            "When escalation is triggered, deliver the following message exactly "
            "and stop negotiation."
        ),
        "recommendation": (
            "Revise the escalation branch so it includes the required spoken text, "
            "hotline ending, and call transfer_to_human."
        ),
    }

    assert required_tool_for_case(conversation_with_transfer_tool(), case) is None


def test_required_tool_for_case_still_infers_missing_tool_case() -> None:
    case = {
        "turn_index": 3,
        "error_type": "missing_required_tool_call",
        "evidence": "Assistant answered without the required function call.",
        "recommendation": "The assistant must call transfer_to_human before answering.",
    }

    assert required_tool_for_case(conversation_with_transfer_tool(), case) == "transfer_to_human"


def test_classify_prompt_edit_failure_marks_backend_json_failure() -> None:
    case = {
        "turn_index": 9,
        "error_type": "late_payment_proposal_not_rtp_closing",
        "evidence": "Assistant negotiated after a late date.",
        "recommendation": "Clarify RTP_Closing and do not negotiate.",
    }

    category = classify_prompt_edit_failure(
        data=conversation_with_transfer_tool(),
        case=case,
        rationale="LLM prompt edit failed: Unterminated string starting at line 2.",
    )

    assert category == "backend_failed_to_patch"


def test_build_verification_failures_marks_patch_applied_but_failed() -> None:
    applied_cases = [
        {
            "id": "case-a",
            "turn_index": 9,
            "error_type": "late_payment_proposal_not_rtp_closing",
            "fix_type": "prompt_edit_targeted_rerun",
            "repair_class": "fixable_by_prompt_clarification",
        }
    ]
    residual_cases = [
        {
            "id": "residual-a",
            "turn_index": 9,
            "error_type": "late_payment_proposal_not_rtp_closing",
            "evidence": "Assistant still negotiated after rerun.",
        }
    ]

    failures = build_verification_failures(applied_cases, residual_cases)

    assert failures == [
        {
            "case_id": "case-a",
            "turn_index": 9,
            "error_type": "late_payment_proposal_not_rtp_closing",
            "fix_type": "prompt_edit_targeted_rerun",
            "repair_class": "fixable_by_prompt_clarification",
            "failure_category": "patch_applied_but_failed_verification",
            "reason": (
                "A prompt change and/or targeted rerun was applied, but the residual scan "
                "still found the same rule violation."
            ),
            "residual_case_id": "residual-a",
            "residual_turn_index": 9,
            "residual_evidence": "Assistant still negotiated after rerun.",
            "next_experiment": (
                "Strengthen the RTP_Closing rule into deterministic forbidden/required outputs, "
                "then rerun the target turn."
            ),
        }
    ]
    assert failure_category_counts([{"verification_failures": failures}]) == {
        "patch_applied_but_failed_verification": 1
    }


def test_format_failure_category_counts_zh_explains_counts() -> None:
    assert format_failure_category_counts_zh(
        {
            "backend_failed_to_patch": 3,
            "patch_applied_but_failed_verification": 1,
        }
    ) == "后端未生成可用 prompt patch=3 个，已改动但 residual scan 未通过=1 个"


def test_apply_prompt_cases_matches_app_selected_rerun(monkeypatch) -> None:
    data = ConversationData(
        system_prompt="Prompt v0",
        interactions=[
            Interaction(role="user", content="hello"),
            Interaction(role="assistant", content="hi"),
            Interaction(role="user", content="next month"),
            Interaction(role="assistant", content="when exactly?"),
            Interaction(role="user", content="already paid"),
            Interaction(role="assistant", content="what date?"),
        ],
        tools={},
    )
    cases = [
        {
            "id": "case-a",
            "turn_index": 3,
            "role": "assistant",
            "error_type": "late_payment_proposal_not_rtp_closing",
            "evidence": "late date",
            "recommendation": "go to closing",
        },
        {
            "id": "case-b",
            "turn_index": 5,
            "role": "assistant",
            "error_type": "escalation_action_not_followed",
            "evidence": "already paid",
            "recommendation": "escalate",
        },
    ]
    rerun_calls = []

    class FakeOptimization:
        def __init__(self, prompt: str) -> None:
            self.optimized_prompt = prompt
            self.rationale = "changed"
            self.applied_feedback_summary = "changed"

    class FakeRerun:
        def __init__(self, turn: int) -> None:
            self.user_turn_index = turn - 1
            self.assistant_turn_index = turn
            self.user_message = ""
            self.old_assistant_response = ""
            self.new_assistant_response = f"fixed turn {turn}"
            self.error = None
            self.response_diagnostics = None

    def fake_apply_recommendation_to_system_prompt(**kwargs):
        return FakeOptimization(f'{kwargs["current_system_prompt"]} + {kwargs["bad_case"].evidence}')

    def fake_rerun_conversation(**kwargs):
        targets = set(kwargs["target_assistant_turn_indices"])
        rerun_calls.append(targets)
        return [FakeRerun(turn) for turn in sorted(targets)]

    monkeypatch.setattr(
        "tools.batch_prompt_compliance.apply_recommendation_to_system_prompt",
        fake_apply_recommendation_to_system_prompt,
    )
    monkeypatch.setattr(
        "tools.batch_prompt_compliance.rerun_conversation",
        fake_rerun_conversation,
    )
    args = type(
        "Args",
        (),
        {
            "apply_backend": "company",
            "url": None,
            "provider": None,
            "model": "test-model",
            "model_contains": None,
            "max_completion_tokens": 4096,
        },
    )()

    updated_data, applied, unsupported, meta = apply_prompt_cases_with_llm(
        data=data,
        prompt_cases=cases,
        tool_cases=[],
        args=args,
    )

    assert rerun_calls == [{3, 5}]
    assert [turn.content for turn in updated_data.interactions if turn.role == "assistant"] == [
        "hi",
        "fixed turn 3",
        "fixed turn 5",
    ]
    assert [batch["target_turns"] for batch in meta["rerun_batches"]] == [[3, 5]]
    assert meta["rerun_target_turns"] == [3, 5]
    assert len(applied) == 2
    assert unsupported == []


def test_apply_prompt_cases_retries_unchanged_prompt_like_app(monkeypatch) -> None:
    data = ConversationData(
        system_prompt="Prompt v0",
        interactions=[
            Interaction(role="user", content="next month"),
            Interaction(role="assistant", content="when exactly?"),
        ],
        tools={},
    )
    cases = [
        {
            "id": "case-a",
            "turn_index": 1,
            "role": "assistant",
            "error_type": "late_payment_proposal_not_rtp_closing",
            "evidence": "late date",
            "recommendation": "go to closing",
        }
    ]
    prompts_seen = []

    class FakeOptimization:
        def __init__(self, prompt: str) -> None:
            self.optimized_prompt = prompt
            self.rationale = "changed"
            self.applied_feedback_summary = "changed"

    class FakeRerun:
        user_turn_index = 0
        assistant_turn_index = 1
        user_message = ""
        old_assistant_response = ""
        new_assistant_response = "closing"
        error = None
        response_diagnostics = None

    def fake_apply_recommendation_to_system_prompt(**kwargs):
        prompts_seen.append(kwargs["bad_case"].recommendation)
        if len(prompts_seen) == 1:
            return FakeOptimization(kwargs["current_system_prompt"])
        return FakeOptimization(kwargs["current_system_prompt"] + " + retry edit")

    monkeypatch.setattr(
        "tools.batch_prompt_compliance.apply_recommendation_to_system_prompt",
        fake_apply_recommendation_to_system_prompt,
    )
    monkeypatch.setattr(
        "tools.batch_prompt_compliance.rerun_conversation",
        lambda **kwargs: [FakeRerun()],
    )
    args = type(
        "Args",
        (),
        {
            "apply_backend": "company",
            "url": None,
            "provider": None,
            "model": "test-model",
            "model_contains": None,
            "max_completion_tokens": 4096,
        },
    )()

    updated_data, applied, unsupported, meta = apply_prompt_cases_with_llm(
        data=data,
        prompt_cases=cases,
        tool_cases=[],
        args=args,
    )

    assert "previous apply attempt returned the system prompt unchanged" in prompts_seen[1].lower()
    assert updated_data.system_prompt == "Prompt v0 + retry edit"
    assert meta["prompt_edit_retries"] == 1
    assert applied[0]["prompt_changed"] is True
    assert unsupported == []


def test_apply_prompt_cases_marks_rejected_prompt_edit_unsupported(monkeypatch) -> None:
    data = ConversationData(
        system_prompt="Prompt v0",
        interactions=[
            Interaction(role="user", content="next month"),
            Interaction(role="assistant", content="when exactly?"),
        ],
        tools={},
    )
    cases = [
        {
            "id": "case-a",
            "turn_index": 1,
            "role": "assistant",
            "error_type": "late_payment_proposal_not_rtp_closing",
            "evidence": "late date",
            "recommendation": "go to closing",
        }
    ]
    rerun_calls = []

    class FakeOptimization:
        optimized_prompt = "Prompt v0"
        rationale = "LLM prompt edit rejected: returned a substantially shorter prompt."
        applied_feedback_summary = "No changes applied."

    monkeypatch.setattr(
        "tools.batch_prompt_compliance.apply_recommendation_to_system_prompt",
        lambda **kwargs: FakeOptimization(),
    )
    monkeypatch.setattr(
        "tools.batch_prompt_compliance.rerun_conversation",
        lambda **kwargs: rerun_calls.append(kwargs) or [],
    )
    args = type(
        "Args",
        (),
        {
            "apply_backend": "company",
            "url": None,
            "provider": None,
            "model": "test-model",
            "model_contains": None,
            "max_completion_tokens": 4096,
        },
    )()

    updated_data, applied, unsupported, meta = apply_prompt_cases_with_llm(
        data=data,
        prompt_cases=cases,
        tool_cases=[],
        args=args,
    )

    assert updated_data.system_prompt == "Prompt v0"
    assert applied == []
    assert unsupported[0]["id"] == "case-a"
    assert unsupported[0]["failure_category"] == "backend_failed_to_patch"
    assert rerun_calls == []
    assert meta["rerun_attempted"] is False
