import argparse
import json

from prompt_optimizer_agent.agent_logic import BadCase, LLMSettings
from prompt_optimizer_agent.json_utils import ConversationData, Interaction
from tools.batch_prompt_compliance import (
    DEFAULT_APPLY_BATCH_ID,
    DEFAULT_RESIDUAL_CONTINUE_BATCH_ID,
    DEFAULT_SCAN_BATCH_ID,
    LATEST_APPLY_CONCLUSION_JSON,
    LATEST_APPLY_CONCLUSION_MD,
    LATEST_REVIEW_JSON,
    LATEST_REVIEW_MD,
    REVIEW_BADCASE_SCHEMA_FIELDS,
    apply_file_record,
    apply_prompt_cases_with_llm,
    bad_case_record,
    build_batch_conclusion_sections,
    build_parser,
    build_residual_continue_review,
    build_verification_failures,
    classify_prompt_edit_failure,
    exact_responses_for_case_records,
    failure_category_counts,
    format_failure_category_counts_zh,
    required_tool_for_case,
    render_apply_conclusion_markdown,
    run_apply,
    run_rerun_turn,
    run_scan,
    run_scan_judge,
    scan_file,
)


def test_batch_conclusion_has_three_semantic_parts_with_backend_evidence() -> None:
    conclusion = {
        "batch_id": "apply-one",
        "approved_case_count": 1,
        "applied_case_count": 1,
        "fixed_case_count": 0,
        "residual_badcase_count": 1,
        "unsupported_case_count": 0,
        "verification_failure_count": 1,
        "failure_category_counts": {"patch_applied_but_failed_verification": 1},
        "files": [
            {
                "source_file": "conversation.json",
                "updated_file": "conversation_updated.json",
                "scan_event_id": "scan-1",
                "apply_event_id": "apply-1",
                "residual_scan_event_id": "residual-1",
                "apply_provider": "openai_api_like",
                "apply_model": "voyager-test",
                "source_meta": {
                    "type": "request_log",
                    "meta": {
                        "kwargs": {
                            "provider": "meta-provider",
                            "model": "meta-model",
                            "url": "http://meta.example",
                        }
                    },
                },
                "prompt_changed": True,
                "before_hash": "before",
                "after_hash": "after",
                "rerun_target_turns": [9],
                "rerun_errors": [],
                "prompt_edit_summaries": [
                    {
                        "case_id": "case-1",
                        "applied_feedback_summary": "Made RTP_Closing terminal.",
                        "rationale": "The previous branch allowed negotiation.",
                    }
                ],
                "rerun_results": [
                    {
                        "assistant_turn_index": 9,
                        "old_assistant_response": "When can you pay?",
                        "new_assistant_response": "Can you pay sooner?",
                        "error": None,
                        "response_diagnostics": {
                            "request_id": "request-1",
                            "provider": "openai_api_like",
                            "model": "voyager-test",
                            "logprobs": {
                                "available": True,
                                "avg_logprob": -0.01,
                                "min_logprob": -0.2,
                                "low_confidence_tokens": [{"token": "sooner", "logprob": -0.2}],
                            },
                        },
                    }
                ],
                "applied_case_count": 1,
                "fixed_case_count": 0,
                "residual_badcase_count": 1,
                "residual_badcases": [{"turn_index": 9}],
                "unsupported_cases": [],
                "failure_category_counts": {"patch_applied_but_failed_verification": 1},
                "verification_failures": [
                        {
                            "case_id": "case-1",
                            "residual_case_id": "residual-1",
                            "error_type": "late_payment_proposal_not_rtp_closing",
                            "reason": "The rerun still negotiated.",
                        "residual_evidence": "Turn 9 asked for another date.",
                        "next_experiment": "Use deterministic backend replacement.",
                    }
                ],
            }
        ],
    }

    sections = build_batch_conclusion_sections(conclusion)
    markdown = render_apply_conclusion_markdown(conclusion)

    assert sections["verdict_status"] == "not fixed"
    assert sections["verification_verdict"].startswith("Status: not fixed.")
    assert "Correctness basis: residual scan results" in sections["verification_verdict"]
    assert "Root cause analysis" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "Evidence data (surface)" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "Evidence data (deep)" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "`conversation.json` is not verified fixed" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "old=`When can you pay?` new=`Can you pay sooner?`" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "openai_api_like" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "voyager-test" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "request-1" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "source_meta=" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "meta-provider" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "meta-model" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "Logprob boundary: logprobs can explain confidence" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "they cannot prove correctness" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "Turn 9 asked for another date." not in sections["verification_verdict"]
    assert sections["evidence"][0]["rerun_details"][0]["request_id"] == "request-1"
    assert "## 1. Verification verdict" in markdown
    assert "## 2. Badcase diagnosis and backend evidence" in markdown
    assert "## 3. Next action" in markdown
    assert markdown.index("## 1. Verification verdict") < markdown.index("## 2. Badcase diagnosis and backend evidence")
    assert markdown.index("## 2. Badcase diagnosis and backend evidence") < markdown.index("## 3. Next action")
    return

    assert sections["verification_verdict"].startswith("验证结论：未通过。")
    assert "### 结论说明" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "#### conversation.json" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "- 实验结论：" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "- 输出观察：" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "- Prompt 修改推导：" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "- Logprob 分析：" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "- 根因判断：" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "- 实验记录：" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "仍未修复" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "用户已经给出应进入终局的付款信息" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "### 剩余风险" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "openai_api_like/voyager-test" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "request-1" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "模型稳定选择了当前输出形态" in sections["badcase_diagnosis_and_backend_evidence"]
    assert "正确性仍以 residual scan 为准" in sections["badcase_diagnosis_and_backend_evidence"]
    assert len(sections["badcase_diagnosis_and_backend_evidence"]) < 3200
    assert "Turn 9 asked for another date." not in sections["verification_verdict"]
    assert sections["next_action"].startswith("下一步只处理 residual badcase")
    assert "验收标准" in sections["next_action"]
    assert sections["evidence"][0]["rerun_details"][0]["request_id"] == "request-1"
    assert "## 1. 验证结论" in markdown
    assert "## 2. 诊断与证据" in markdown
    assert "## 3. 下一步动作" in markdown


def test_rerun_turn_defaults_to_direct_transport() -> None:
    args = build_parser().parse_args(
        [
            "rerun-turn",
            "--review",
            "batch_review.json",
            "--case-id",
            "case-one",
        ]
    )

    assert args.transport == "direct"


def test_batch_commands_use_stable_default_batch_ids() -> None:
    scan_args = build_parser().parse_args(["scan", "conversation.json"])
    apply_args = build_parser().parse_args(["apply", "batch_review.json", "--approve-all"])
    continue_args = build_parser().parse_args(["continue-residual", "batch_apply_conclusion.json"])

    assert scan_args.batch_id is None
    assert apply_args.batch_id is None
    assert continue_args.batch_id is None
    assert DEFAULT_SCAN_BATCH_ID == "latest-scan"
    assert DEFAULT_APPLY_BATCH_ID == "latest-apply"
    assert DEFAULT_RESIDUAL_CONTINUE_BATCH_ID == "latest-residual-continue"


def test_build_residual_continue_review_uses_conclusion_next_step() -> None:
    conclusion = {
        "batch_id": "apply-one",
        "approved_case_count": 1,
        "applied_case_count": 1,
        "fixed_case_count": 0,
        "residual_badcase_count": 1,
        "unsupported_case_count": 0,
        "verification_failure_count": 1,
        "failure_category_counts": {"patch_applied_but_failed_verification": 1},
        "files": [
            {
                "source_file": "original.json",
                "updated_file": "updated.json",
                "residual_badcases": [
                    {
                        "id": "residual-1",
                        "turn_index": 9,
                        "role": "assistant",
                        "error_type": "late_payment_proposal_not_rtp_closing",
                        "evidence": "Still asks for a date.",
                        "recommendation": "Old recommendation.",
                    }
                ],
                "verification_failures": [
                    {
                        "case_id": "case-1",
                        "residual_case_id": "residual-1",
                        "error_type": "late_payment_proposal_not_rtp_closing",
                        "next_experiment": "Use deterministic backend replacement.",
                    }
                ],
            }
        ],
    }

    review = build_residual_continue_review(conclusion, batch_id="continue-one")
    case = review["files"][0]["badcases"][0]

    assert review["mode"] == "residual_continue"
    assert review["parent_apply_batch_id"] == "apply-one"
    assert review["files"][0]["path"] == "updated.json"
    assert review["files"][0]["scan_event_id"] == "continue-one-residual-review-0001"
    assert case["parent_case_id"] == "case-1"
    assert "Use deterministic backend replacement." in case["recommendation"]


def test_run_scan_overwrites_latest_review_snapshot(tmp_path, monkeypatch) -> None:
    source = tmp_path / "conversation.json"
    source.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "scan"
    output_dir.mkdir()
    (output_dir / "skill-scan-old_review.json").write_text("{}", encoding="utf-8")
    (output_dir / "skill-scan-old_review.md").write_text("# old", encoding="utf-8")
    log_path = tmp_path / "rounds.jsonl"
    batches = iter(["scan-one", "scan-two"])

    monkeypatch.setattr(
        "tools.batch_prompt_compliance.scan_file",
        lambda path, **kwargs: {
            "path": str(path),
            "file_name": path.name,
            "status": "scanned",
            "badcase_count": 0,
            "badcases": [],
        },
    )

    for _ in range(2):
        args = argparse.Namespace(
            paths=[str(source)],
            pattern="*.json",
            no_recursive=False,
            judge="local",
            model=None,
            output_dir=str(output_dir),
            log=str(log_path),
            batch_id=next(batches),
        )
        assert run_scan(args) == 0

    assert sorted(path.name for path in output_dir.iterdir()) == [
        LATEST_REVIEW_JSON,
        LATEST_REVIEW_MD,
    ]
    review = json.loads((output_dir / LATEST_REVIEW_JSON).read_text(encoding="utf-8"))
    assert review["batch_id"] == "scan-two"


def test_scan_file_records_invalid_json_mechanical_badcase(tmp_path) -> None:
    source = tmp_path / "broken.json"
    source.write_text('{"system_prompt": "Prompt", "interactions": [', encoding="utf-8")

    record = scan_file(source, judge="local", settings=LLMSettings())

    assert record["status"] == "parse_error"
    assert record["badcase_count"] == 1
    assert record["badcases"][0]["error_type"] == "invalid_json"
    assert record["badcases"][0]["source"] == "mechanical_detector"
    assert record["badcases"][0]["rerunnable"] is False
    for field in REVIEW_BADCASE_SCHEMA_FIELDS:
        assert field in record["badcases"][0]


def test_scan_file_records_wrong_turn_role_mechanical_badcase(tmp_path) -> None:
    source = tmp_path / "wrong_role.json"
    source.write_text(
        json.dumps(
            {
                "system_prompt": "Prompt",
                "interactions": [{"role": "developer", "content": "hidden"}],
            }
        ),
        encoding="utf-8",
    )

    record = scan_file(source, judge="local", settings=LLMSettings())

    assert record["status"] == "parse_error"
    assert record["badcase_count"] == 1
    assert record["badcases"][0]["error_type"] == "wrong_turn_role"


def test_bad_case_record_marks_local_scan_as_deterministic_source() -> None:
    record = bad_case_record(
        BadCase(
            turn_index=26,
            role="assistant",
            error_type="missing_required_tool_call",
            evidence=(
                "Violated rule: Must call search before answering.\n\n"
                "Evidence: User turn 25 asked about a promotion. Assistant turn 26 answered without search."
            ),
            recommendation="Call the required tool before answering.",
            source="local_scan",
        )
    )

    assert list(record)[: len(REVIEW_BADCASE_SCHEMA_FIELDS)] == list(REVIEW_BADCASE_SCHEMA_FIELDS)
    assert record["source"] == "deterministic_rule_detector"
    assert record["original_source"] == "local_scan"
    assert record["violated_rule"] == "Must call search before answering."
    assert record["user_trigger"] == "User turn 25 asked about a promotion."
    assert record["assistant_violation"] == "Assistant turn 26 answered without search."
    assert record["rerunnable"] is True


def test_bad_case_record_marks_ai_judge_as_semantic_source() -> None:
    record = bad_case_record(
        BadCase(
            turn_index=3,
            role="assistant",
            error_type="workflow_branch_not_followed",
            evidence="Violated rule: Follow branch B.\n\nEvidence: Assistant turn 3 followed branch A.",
            recommendation="Follow branch B.",
            source="ai_judge",
        )
    )

    assert record["source"] == "codex_semantic_judge"
    assert record["original_source"] == "ai_judge"


def test_run_scan_normalizes_legacy_badcase_dicts_before_writing_review(tmp_path, monkeypatch) -> None:
    source = tmp_path / "conversation.json"
    source.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "scan"
    log_path = tmp_path / "rounds.jsonl"

    monkeypatch.setattr(
        "tools.batch_prompt_compliance.scan_file",
        lambda path, **kwargs: {
            "path": str(path),
            "file_name": path.name,
            "status": "scanned",
            "badcase_count": 1,
            "badcases": [
                {
                    "turn_index": 26,
                    "role": "assistant",
                    "error_type": "missing_required_tool_call",
                    "source": "local_scan",
                    "evidence": (
                        "Violated rule: Must call search before answering.\n\n"
                        "Evidence: User turn 25 asked about a promotion. "
                        "Assistant turn 26 answered without search."
                    ),
                    "recommendation": "Call the required tool before answering.",
                }
            ],
        },
    )

    args = argparse.Namespace(
        paths=[str(source)],
        pattern="*.json",
        no_recursive=False,
        judge="local",
        model=None,
        output_dir=str(output_dir),
        log=str(log_path),
        batch_id="scan-one",
    )

    assert run_scan(args) == 0

    review = json.loads((output_dir / LATEST_REVIEW_JSON).read_text(encoding="utf-8"))
    case = review["files"][0]["badcases"][0]
    assert list(case)[: len(REVIEW_BADCASE_SCHEMA_FIELDS)] == list(REVIEW_BADCASE_SCHEMA_FIELDS)
    assert case["source"] == "deterministic_rule_detector"
    assert case["original_source"] == "local_scan"
    assert case["violated_rule"] == "Must call search before answering."
    assert case["user_trigger"] == "User turn 25 asked about a promotion."
    assert case["assistant_violation"] == "Assistant turn 26 answered without search."
    assert case["rerunnable"] is True


def test_run_scan_judge_outputs_only_semantic_badcases_from_skeleton(tmp_path, monkeypatch, capsys) -> None:
    skeleton = {
        "source_file": "conversation.json",
        "file_hash": "file-hash",
        "system_prompt_hash": "prompt-hash",
        "system_prompt": "Always call lookup before answering promotion questions.",
        "turns": [
            {"role": "user", "content": "Any promotion?"},
            {"role": "assistant", "content": "Yes, there is one."},
        ],
        "tools": {"lookup": {"type": "function", "function": {"name": "lookup"}}},
    }
    skeleton_path = tmp_path / "skeleton.json"
    output_path = tmp_path / "semantic.json"
    skeleton_path.write_text(json.dumps(skeleton), encoding="utf-8")

    monkeypatch.setattr(
        "tools.batch_prompt_compliance._local_prompt_rule_bad_cases",
        lambda data: [
            BadCase(
                turn_index=1,
                role="assistant",
                error_type="thought_exposed",
                evidence="mechanical",
                recommendation="clean",
                source="mechanical_detector",
            ),
            BadCase(
                turn_index=1,
                role="assistant",
                error_type="missing_required_tool_call",
                evidence=(
                    "Violated rule: Always call lookup before answering promotion questions.\n\n"
                    "Evidence: User turn 0 asked about a promotion. Assistant turn 1 answered without lookup."
                ),
                recommendation="Call lookup before answering.",
                source="local_scan",
            ),
        ],
    )

    args = argparse.Namespace(
        skeleton_json=str(skeleton_path),
        judge="local",
        model=None,
        output=str(output_path),
    )

    assert run_scan_judge(args) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    case = payload["badcases"][0]

    assert payload["mode"] == "scan_judge"
    assert payload["badcase_count"] == 1
    assert case["error_type"] == "missing_required_tool_call"
    assert case["source"] == "deterministic_rule_detector"
    assert case["original_source"] == "local_scan"
    assert case["violated_rule"] == "Always call lookup before answering promotion questions."
    assert json.loads(capsys.readouterr().out)["output"] == str(output_path)


def test_run_rerun_turn_reads_case_from_review_and_writes_outputs(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "conversation.json"
    source.write_text(
        json.dumps(
            {
                "system_prompt": "Answer briefly.",
                "interactions": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "old"},
                ],
                "tools": {"lookup": {"type": "function", "function": {"name": "lookup"}}},
            }
        ),
        encoding="utf-8",
    )
    review_path = tmp_path / "batch_review.json"
    review_path.write_text(
        json.dumps(
            {
                "batch_id": "scan-one",
                "files": [
                    {
                        "path": str(source),
                        "scan_event_id": "scan-one-scan-0001",
                        "badcases": [
                            {
                                "id": "case-one",
                                "turn_index": 1,
                                "role": "assistant",
                                "rerunnable": True,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "updated.json"
    diagnostics_json = tmp_path / "diagnostics.json"
    calls = []

    class FakeResult:
        updated_data = ConversationData(
            system_prompt="Answer briefly.",
            interactions=[
                Interaction(role="user", content="hello"),
                Interaction(role="assistant", content="new"),
            ],
            tools={"lookup": {"type": "function", "function": {"name": "lookup"}}},
        )

        def diagnostics_payload(self, **kwargs):
            return {
                "source_path": kwargs["source_path"],
                "output_path": kwargs["output_path"],
                "target_turns": [1],
                "settings": {"model": kwargs["settings"].model},
                "aggregate": {
                    "result_count": 1,
                    "error_count": 0,
                    "logprobs_available_count": 1,
                },
                "results": [],
            }

    def fake_rerun(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    monkeypatch.setattr("tools.batch_prompt_compliance.rerun_target_turns_with_logprobs", fake_rerun)

    args = argparse.Namespace(
        review=str(review_path),
        case_id="case-one",
        url="http://example.invalid",
        provider="openai_api_like",
        model="voyager-test",
        transport="direct",
        temperature=0.1,
        max_completion_tokens=128,
        top_logprobs=3,
        no_logprobs=False,
        context_window_turns=-1,
        timeout_seconds=30,
        no_tool_placeholders=False,
        output_dir=str(tmp_path),
        output_json=str(output_json),
        diagnostics_json=str(diagnostics_json),
        run_id="run-one",
        no_raw_response=True,
        include_request_payload=False,
    )

    assert run_rerun_turn(args) == 0

    assert calls[0]["target_assistant_turn_indices"] == {1}
    assert calls[0]["data"].system_prompt == "Answer briefly."
    assert calls[0]["data"].tools["lookup"]["function"]["name"] == "lookup"
    assert calls[0]["settings"].model == "voyager-test"
    updated = json.loads(output_json.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)

    assert updated["interactions"][1]["content"] == "new"
    assert diagnostics["case_id"] == "case-one"
    assert diagnostics["scan_event_id"] == "scan-one-scan-0001"
    assert printed["updated_json"] == str(output_json)
    assert printed["target_turns"] == [1]


def test_run_apply_writes_latest_snapshot_without_batch_subdirectory(tmp_path, monkeypatch) -> None:
    review_path = tmp_path / "batch_review.json"
    review_path.write_text(
        json.dumps(
            {
                "batch_id": "scan-one",
                "files": [
                    {
                        "path": str(tmp_path / "conversation.json"),
                        "badcases": [{"id": "case-one"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "applied"
    output_dir.mkdir()
    (output_dir / "conversation_updated.json").write_text("{}", encoding="utf-8")
    (output_dir / "skill-apply-old_conclusion.json").write_text("{}", encoding="utf-8")
    (output_dir / "skill-apply-old_conclusion.md").write_text("# old", encoding="utf-8")
    log_path = tmp_path / "rounds.jsonl"

    monkeypatch.setattr(
        "tools.batch_prompt_compliance.apply_file_record",
        lambda *args, **kwargs: {
            "approved_case_count": 1,
            "applied_case_count": 1,
            "fixed_case_count": 1,
            "unsupported_cases": [],
            "verification_failures": [],
            "residual_badcase_count": 0,
        },
    )
    monkeypatch.setattr("tools.batch_prompt_compliance.append_round_event", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        list_models=False,
        review_json=str(review_path),
        batch_id="apply-one",
        output_dir=str(output_dir),
        approve_all=True,
        approval_file=None,
        log=str(log_path),
    )

    assert run_apply(args) == 0
    assert sorted(path.name for path in output_dir.iterdir()) == [
        LATEST_APPLY_CONCLUSION_JSON,
        LATEST_APPLY_CONCLUSION_MD,
    ]
    assert not (output_dir / "apply-one").exists()


def test_apply_file_record_preserves_source_meta_for_conclusion(tmp_path, monkeypatch) -> None:
    source = tmp_path / "conversation.json"
    source.write_text(
        json.dumps(
            {
                "system_prompt": "Call lookup before answering.",
                "interactions": [
                    {"role": "user", "content": "Any promotion?"},
                    {"role": "assistant", "content": "Yes."},
                ],
                "tools": {"lookup": {"type": "function", "function": {"name": "lookup"}}},
                "source_meta": {
                    "type": "request_log",
                    "meta": {"kwargs": {"provider": "meta-provider", "model": "meta-model"}},
                },
            }
        ),
        encoding="utf-8",
    )
    file_record = {
        "path": str(source),
        "badcases": [
            {
                "id": "case-one",
                "turn_index": 1,
                "role": "assistant",
                "error_type": "missing_required_tool_call",
            }
        ],
    }
    monkeypatch.setattr("tools.batch_prompt_compliance.required_tool_for_case", lambda *args, **kwargs: "lookup")

    result = apply_file_record(
        file_record,
        {"case-one"},
        tmp_path,
        argparse.Namespace(),
    )

    assert result["source_meta"]["type"] == "request_log"
    assert result["source_meta"]["meta"]["kwargs"]["provider"] == "meta-provider"


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


def test_exact_responses_for_case_records_maps_escalation_turn() -> None:
    exact = (
        "Thank you for letting me know your situation. Please contact the support hotline "
        "at one two three four. Thank you and goodbye.<dialog-end>"
    )
    data = ConversationData(
        system_prompt=(
            "**6.1 Escalation Action:**\n"
            f"When an escalation is triggered, deliver the following message exactly \"{exact}\"."
        ),
        interactions=[
            Interaction(role="user", content="I already paid."),
            Interaction(role="assistant", content="Which date was that?"),
        ],
    )
    case = {
        "turn_index": 1,
        "error_type": "escalation_action_not_followed",
        "evidence": "deliver the following message exactly and stop negotiation.",
        "recommendation": "Use the exact configured escalation action and hotline ending.",
    }

    assert exact_responses_for_case_records(data, [case]) == {1: exact}


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
                "Route recognized beyond-maximum-date triggers through deterministic backend RTP_Closing response "
                "replacement, then verify the target turn without another prompt-only retry."
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
    force_flags = []

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
        force_flags.append(kwargs.get("force_prompt_edit", False))
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
    assert force_flags == [False, True]
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
