from prompt_optimizer_agent.json_utils import ConversationData, Interaction
from tools.batch_prompt_compliance import required_tool_for_case


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
