from prompt_optimizer_agent.json_utils import ConversationData, Interaction
import prompt_optimizer_agent.rerun_logprobs as rerun_logprobs
from prompt_optimizer_agent.rerun_logprobs import (
    RerunLogprobsSettings,
    normalize_chat_response,
    parse_turn_specs,
    rerun_target_turns_with_logprobs,
)


def test_default_company_model_is_voyager() -> None:
    assert rerun_logprobs.DEFAULT_COMPANY_MODEL == "voyager-1.6-gemma4-26b-a4b-it"
    assert RerunLogprobsSettings().model == "voyager-1.6-gemma4-26b-a4b-it"


def test_parse_turn_specs_accepts_lists_and_ranges() -> None:
    assert parse_turn_specs(["9,11", "15-17"]) == {9, 11, 15, 16, 17}


def test_normalize_openai_response_extracts_logprobs_and_usage() -> None:
    response = {
        "choices": [
            {
                "message": {"content": "fixed answer"},
                "logprobs": {"content": [{"token": "fixed", "logprob": -0.1}]},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }

    normalized = normalize_chat_response(response)

    assert normalized.content == "fixed answer"
    assert normalized.usage == {"prompt_tokens": 10, "completion_tokens": 2}
    assert normalized.logprobs == {"content": [{"token": "fixed", "logprob": -0.1}]}


def test_rerun_target_turn_requests_logprobs_and_updates_conversation() -> None:
    data = ConversationData(
        system_prompt="Prompt v0",
        interactions=[
            Interaction(role="user", content="hello"),
            Interaction(role="assistant", content="hi"),
            Interaction(role="user", content="next month"),
            Interaction(role="assistant", content="when exactly?"),
        ],
        tools={},
    )
    captured_payloads = []

    def fake_backend(payload, settings):
        captured_payloads.append(payload)
        return (
            {
                "choices": [
                    {
                        "message": {"content": "RTP closing answer"},
                        "logprobs": {
                            "content": [
                                {"token": "RTP", "logprob": -0.2},
                                {"token": " closing", "logprob": -0.3},
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            },
            "direct",
        )

    result = rerun_target_turns_with_logprobs(
        data=data,
        optimized_prompt="Prompt v1",
        target_assistant_turn_indices={3},
        settings=RerunLogprobsSettings(transport="direct", top_logprobs=5),
        backend_caller=fake_backend,
    )

    assert captured_payloads[0]["logprobs"] is True
    assert captured_payloads[0]["top_logprobs"] == 5
    assert captured_payloads[0]["messages"][0] == {"role": "system", "content": "Prompt v1"}
    assert captured_payloads[0]["messages"][-1] == {"role": "user", "content": "next month"}
    assert result.updated_data.system_prompt == "Prompt v1"
    assert result.updated_data.interactions[3].content == "RTP closing answer"
    assert result.turns[0].usage == {"prompt_tokens": 20, "completion_tokens": 5}
    assert result.turns[0].response_diagnostics["logprobs"]["available"] is True
    assert result.turns[0].request_meta["request_logprobs"] is True


def test_rerun_target_turn_writes_tool_call_placeholder() -> None:
    data = ConversationData(
        system_prompt="Prompt",
        interactions=[
            Interaction(role="user", content="agent"),
            Interaction(role="assistant", content="I'll transfer."),
        ],
        tools={
            "transfer_to_human": {
                "type": "function",
                "function": {"name": "transfer_to_human"},
            }
        },
    )

    def fake_backend(payload, settings):
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "transfer_to_human",
                                        "arguments": '{"reason":"requested"}',
                                    },
                                }
                            ],
                        },
                        "logprobs": {"content": [{"token": "<tool>", "logprob": -0.4}]},
                    }
                ],
                "usage": {"completion_tokens": 1},
            },
            "direct",
        )

    result = rerun_target_turns_with_logprobs(
        data=data,
        target_assistant_turn_indices={1},
        settings=RerunLogprobsSettings(transport="direct"),
        backend_caller=fake_backend,
    )

    assert result.updated_data.interactions[1].tool_calls[0]["function"]["name"] == "transfer_to_human"
    assert "<function-call>transfer_to_human" in result.updated_data.interactions[1].content
    assert result.updated_data.interactions[2].role == "tool"
    assert result.updated_data.interactions[2].tool_call_id == "call_1"
