import json
import tempfile
from pathlib import Path

import prompt_optimizer_agent.company_demo_client as company_demo_client
from prompt_optimizer_agent.company_demo_client import (
    _extract_response_content,
    _payload_prompt_refs,
    _payload_prompt_values,
    _prompt_hash,
    _response_logprob_diagnostics,
    _record_request_audit,
    preflight_company_request,
)


def test_payload_prompt_refs_extracts_judge_system_prompt_hash() -> None:
    prompt = "new working prompt"
    refs = _payload_prompt_refs(
        [
            {"role": "system", "content": "judge prompt"},
            {"role": "user", "content": json.dumps({"system_prompt": prompt})},
        ]
    )

    assert refs["system_prompt"]["hash"] == _prompt_hash(prompt)


def test_payload_prompt_refs_extracts_prompt_edit_current_prompt_hash() -> None:
    prompt = "current prompt v3"
    refs = _payload_prompt_refs(
        [
            {"role": "system", "content": "prompt editor"},
            {"role": "user", "content": json.dumps({"current_system_prompt": prompt})},
        ]
    )

    assert refs["current_system_prompt"]["hash"] == _prompt_hash(prompt)


def test_payload_prompt_values_extracts_full_prompt_text() -> None:
    prompt = "full working prompt text"
    values = _payload_prompt_values(
        [
            {"role": "system", "content": "judge prompt"},
            {"role": "user", "content": json.dumps({"system_prompt": prompt})},
        ]
    )

    assert values["system_prompt"] == prompt


def test_expected_prompt_hash_mismatch_blocks_rerun_before_send() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        old_audit_path = company_demo_client.DEFAULT_AUDIT_PATH
        old_prompt_path = company_demo_client.DEFAULT_SYSTEM_PROMPT_LOG_PATH
        old_preflight_path = company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH
        company_demo_client.DEFAULT_AUDIT_PATH = Path(temp_dir) / "request.jsonl"
        company_demo_client.DEFAULT_SYSTEM_PROMPT_LOG_PATH = Path(temp_dir) / "prompt.log"
        company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH = Path(temp_dir) / "preflight.log"
        try:
            try:
                _record_request_audit(
                    messages=[{"role": "system", "content": "actual prompt"}],
                    model="m",
                    provider="p",
                    url="http://example.test",
                    tools=None,
                    max_completion_tokens=100,
                    temperature=0.1,
                    purpose="targeted_rerun",
                    expected_system_prompt_hash=_prompt_hash("different prompt"),
                    expected_prompt_version="v2 Apply",
                )
            except RuntimeError as exc:
                assert "blocked before send" in str(exc)
            else:
                raise AssertionError("expected RuntimeError")
            assert "match=False" in company_demo_client.DEFAULT_SYSTEM_PROMPT_LOG_PATH.read_text()
        finally:
            company_demo_client.DEFAULT_AUDIT_PATH = old_audit_path
            company_demo_client.DEFAULT_SYSTEM_PROMPT_LOG_PATH = old_prompt_path
            company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH = old_preflight_path


def test_preflight_writes_separate_prompt_log_without_request_log() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        old_audit_path = company_demo_client.DEFAULT_AUDIT_PATH
        old_preflight_path = company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH
        company_demo_client.DEFAULT_AUDIT_PATH = Path(temp_dir) / "request.jsonl"
        company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH = Path(temp_dir) / "preflight.log"
        try:
            prompt = "preflight prompt"
            preflight_company_request(
                messages=[{"role": "system", "content": prompt}],
                model="m",
                provider="p",
                url="http://example.test",
                tools=None,
                max_completion_tokens=100,
                temperature=0.1,
                purpose="targeted_rerun",
                expected_system_prompt_hash=_prompt_hash(prompt),
                expected_prompt_version="v2 Apply",
            )
            assert "targeted_rerun_preflight" in company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH.read_text()
            assert prompt in company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH.read_text()
            assert not company_demo_client.DEFAULT_AUDIT_PATH.exists()
        finally:
            company_demo_client.DEFAULT_AUDIT_PATH = old_audit_path
            company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH = old_preflight_path


def test_preflight_mismatch_blocks_before_actual_request_log() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        old_audit_path = company_demo_client.DEFAULT_AUDIT_PATH
        old_preflight_path = company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH
        company_demo_client.DEFAULT_AUDIT_PATH = Path(temp_dir) / "request.jsonl"
        company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH = Path(temp_dir) / "preflight.log"
        try:
            try:
                preflight_company_request(
                    messages=[{"role": "system", "content": "actual prompt"}],
                    model="m",
                    provider="p",
                    url="http://example.test",
                    tools=None,
                    max_completion_tokens=100,
                    temperature=0.1,
                    purpose="targeted_rerun",
                    expected_system_prompt_hash=_prompt_hash("different prompt"),
                    expected_prompt_version="v2 Apply",
                )
            except RuntimeError as exc:
                assert "blocked before send" in str(exc)
            else:
                raise AssertionError("expected RuntimeError")
            assert "match=False" in company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH.read_text()
            assert not company_demo_client.DEFAULT_AUDIT_PATH.exists()
        finally:
            company_demo_client.DEFAULT_AUDIT_PATH = old_audit_path
            company_demo_client.DEFAULT_PREFLIGHT_SYSTEM_PROMPT_LOG_PATH = old_preflight_path


def test_response_logprob_diagnostics_extracts_token_stats() -> None:
    diagnostics = _response_logprob_diagnostics(
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "bad",
                                "logprob": -4.0,
                                "top_logprobs": [{"token": "good", "logprob": -0.2}],
                            },
                            {"token": " case", "logprob": -0.5},
                        ]
                    }
                }
            ]
        },
        requested=True,
        top_logprobs_requested=5,
    )

    logprobs = diagnostics["logprobs"]
    assert logprobs["available"] is True
    assert logprobs["content_token_count"] == 2
    assert logprobs["min_logprob"] == -4.0
    assert logprobs["avg_logprob"] == -2.25
    assert logprobs["low_confidence_tokens"][0]["token"] == "bad"


def test_extract_response_content_renders_tool_calls() -> None:
    content = _extract_response_content(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "MandiriCX_Call_Center_search_promotion",
                                    "arguments": "{\"query\":\"personal loan\"}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )

    assert (
        content
        == '<function-call>MandiriCX_Call_Center_search_promotion:{"query":"personal loan"}</function-call>'
    )
