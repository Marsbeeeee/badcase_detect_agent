from prompt_optimizer_agent.company_demo_client import _response_logprob_diagnostics, _should_request_logprobs


def test_should_request_logprobs_for_all_company_purposes_by_default(monkeypatch) -> None:
    monkeypatch.delenv("COMPANY_LLM_REQUEST_LOGPROBS", raising=False)

    assert _should_request_logprobs("targeted_rerun") is True
    assert _should_request_logprobs("judge") is True
    assert _should_request_logprobs("prompt_edit") is True
    assert _should_request_logprobs("conclusion") is True


def test_should_request_logprobs_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("COMPANY_LLM_REQUEST_LOGPROBS", "0")

    assert _should_request_logprobs("targeted_rerun") is False
    assert _should_request_logprobs("conclusion") is False


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
    assert diagnostics["response_meta"]["usage"]["prompt_tokens"] == 42


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


def test_response_logprob_diagnostics_reports_empty_logprobs_list() -> None:
    response = {
        "content": "OK.",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0001},
        "logprobs": [],
    }

    diagnostics = _response_logprob_diagnostics(
        response,
        requested=True,
        top_logprobs_requested=5,
    )

    assert diagnostics["logprobs"]["available"] is False
    assert diagnostics["logprobs"]["reason"] == "Company API returned an empty logprobs list."
    assert diagnostics["response_meta"]["usage"]["cost"] == 0.0001
