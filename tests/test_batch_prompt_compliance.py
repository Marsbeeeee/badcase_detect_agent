from prompt_optimizer_agent.json_utils import ConversationData, Interaction
from tools.batch_prompt_compliance import (
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
