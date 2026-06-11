from prompt_optimizer_agent.company_demo_client import _response_logprob_diagnostics


def test_response_logprob_diagnostics_accepts_llmparty_top_level_logprobs() -> None:
    response = {
        "content": "Halo.",
        "usage": {"prompt_tokens": 42, "completion_tokens": 3},
        "logprobs": [
            {"token": "Halo", "logprob": 0.0},
            {"token": ".", "logprob": -0.6931471824645996},
        ],
    }

    diagnostics = _response_logprob_diagnostics(
        response,
        requested=True,
        top_logprobs_requested=5,
    )

    assert diagnostics["logprobs"]["requested"] is True
    assert diagnostics["logprobs"]["available"] is True
    assert diagnostics["logprobs"]["content_token_count"] == 2
    assert diagnostics["logprobs"]["avg_logprob"] == -0.3465735912322998


def test_response_logprob_diagnostics_accepts_openai_choice_logprobs() -> None:
    response = {
        "choices": [
            {
                "message": {"content": "Halo."},
                "logprobs": {
                    "content": [
                        {"token": "Halo", "logprob": 0.0},
                        {"token": ".", "logprob": -0.5},
                    ]
                },
            }
        ]
    }

    diagnostics = _response_logprob_diagnostics(
        response,
        requested=True,
        top_logprobs_requested=5,
    )

    assert diagnostics["logprobs"]["available"] is True
    assert diagnostics["logprobs"]["content_token_count"] == 2
