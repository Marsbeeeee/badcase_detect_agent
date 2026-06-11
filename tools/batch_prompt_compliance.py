from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt_optimizer_agent.agent_logic import (  # noqa: E402
    BadCase,
    DEFAULT_COMPANY_MODEL,
    DEFAULT_COMPANY_PROVIDER,
    DEFAULT_COMPANY_URL,
    LLMSettings,
    _local_prompt_rule_bad_cases,
    analyze_bad_cases,
    apply_recommendation_to_system_prompt,
    extract_exact_escalation_message,
    generate_experiment_conclusion,
    rerun_conversation,
)
from prompt_optimizer_agent.company_demo_client import list_company_models  # noqa: E402
from prompt_optimizer_agent.json_utils import (  # noqa: E402
    ConversationData,
    Interaction,
    function_call_wrappers_to_tool_calls,
    parse_conversation_json,
)


SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv39",
    ".deps",
    ".pycache_tmp",
    "__pycache__",
    "dist",
    "logs",
    "outputs",
}
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "batch"
DEFAULT_ROUND_LOG = PROJECT_ROOT / "logs" / "optimization_rounds.jsonl"
LATEST_REVIEW_JSON = "batch_review.json"
LATEST_REVIEW_MD = "batch_review.md"
LATEST_APPLY_CONCLUSION_JSON = "batch_apply_conclusion.json"
LATEST_APPLY_CONCLUSION_MD = "batch_apply_conclusion.md"
REVIEW_SNAPSHOT_PATTERNS = ("*_review.json", "*_review.md")
APPLY_SNAPSHOT_PATTERNS = ("*_updated.json", "*_conclusion.json", "*_conclusion.md")


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "scan":
        return run_scan(args)
    if args.command == "apply":
        return run_apply(args)
    if args.command == "continue-residual":
        return run_continue_residual(args)
    raise SystemExit(f"Unknown command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch scan/apply prompt-compliance badcases for conversation JSON files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan one or more files/folders and write a human review report.")
    scan.add_argument("paths", nargs="+", help="JSON files or folders containing conversation JSON files.")
    scan.add_argument("--pattern", default="*.json", help="Glob pattern for folders. Default: *.json")
    scan.add_argument("--no-recursive", action="store_true", help="Do not recurse into folders.")
    scan.add_argument("--judge", choices=["local", "company", "openai"], default="local")
    scan.add_argument("--model", default=None, help="Override model for company/openai judge.")
    scan.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the latest batch_review.json/.md snapshot. Existing snapshots are overwritten.",
    )
    scan.add_argument("--log", default=str(DEFAULT_ROUND_LOG))
    scan.add_argument("--batch-id", default=None)

    apply = subparsers.add_parser("apply", help="Apply approved cases from a batch review file.")
    apply.add_argument("review_json", nargs="?", help="Path to batch_review.json produced by scan.")
    apply.add_argument("--approve-all", action="store_true", help="Apply every case in the review file.")
    apply.add_argument(
        "--approval-file",
        default=None,
        help="JSON approval file. Accepts a list of case ids or {'approved_case_ids': [...]}",
    )
    apply.add_argument(
        "--apply-backend",
        choices=["company", "openai"],
        default="company",
        help="Backend used for non-tool-call prompt edits and targeted reruns. Default: company",
    )
    apply.add_argument("--url", default=None, help="Company API base URL for non-tool-call apply.")
    apply.add_argument("--provider", default=None, help="Company API provider for non-tool-call apply.")
    apply.add_argument("--model", default=None, help="Model for non-tool-call apply.")
    apply.add_argument("--model-contains", default=None, help="Select a company model by case-insensitive substring.")
    apply.add_argument("--list-models", action="store_true", help="List company models, optionally filtered by --model-contains, then exit.")
    apply.add_argument("--max-completion-tokens", type=int, default=4096)
    apply.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR / "applied"),
        help="Directory for the latest updated files and conclusion snapshot. Existing files are overwritten.",
    )
    apply.add_argument("--log", default=str(DEFAULT_ROUND_LOG))
    apply.add_argument("--batch-id", default=None)

    cont = subparsers.add_parser(
        "continue-residual",
        help="Create a next-round review from residual badcases in an apply conclusion, optionally apply it.",
    )
    cont.add_argument("conclusion_json", help="Path to batch_apply_conclusion.json from a previous apply.")
    cont.add_argument("--approve-all", action="store_true", help="Immediately apply every residual case.")
    cont.add_argument(
        "--review-output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the next-round batch_review.json/.md snapshot.",
    )
    cont.add_argument(
        "--apply-output-dir",
        default=str(DEFAULT_OUTPUT_DIR / "applied"),
        help="Directory for apply output when --approve-all is used.",
    )
    cont.add_argument("--log", default=str(DEFAULT_ROUND_LOG))
    cont.add_argument("--batch-id", default=None)
    cont.add_argument("--apply-backend", choices=["company", "openai"], default="company")
    cont.add_argument("--url", default=None)
    cont.add_argument("--provider", default=None)
    cont.add_argument("--model", default=None)
    cont.add_argument("--model-contains", default=None)
    cont.add_argument("--list-models", action="store_true")
    cont.add_argument("--max-completion-tokens", type=int, default=4096)
    return parser


def run_scan(args: argparse.Namespace) -> int:
    files = discover_files([Path(path).expanduser() for path in args.paths], args.pattern, not args.no_recursive)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_generated_snapshots(output_dir, REVIEW_SNAPSHOT_PATTERNS)
    batch_id = args.batch_id or new_batch_id("batch-scan")
    judge_settings = build_judge_settings(args)

    records = []
    for index, path in enumerate(files, start=1):
        record = scan_file(path, judge=args.judge, settings=judge_settings)
        record["scan_event_id"] = f"{batch_id}-scan-{index:04d}"
        record["batch_id"] = batch_id
        records.append(record)
        append_round_event(Path(args.log).expanduser(), scan_round_event(record))

    review = {
        "batch_id": batch_id,
        "created_at": utc_timestamp(),
        "mode": "scan",
        "judge": args.judge,
        "input_count": len(files),
        "file_count": len(records),
        "badcase_count": sum(int(record.get("badcase_count") or 0) for record in records),
        "files": records,
        "approval_template": {
            "approved_case_ids": [],
            "notes": "Fill this list, or rerun apply with --approve-all after human review.",
        },
    }
    review_json = output_dir / LATEST_REVIEW_JSON
    review_md = output_dir / LATEST_REVIEW_MD
    review_json.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    review_md.write_text(render_review_markdown(review), encoding="utf-8")

    print(
        json.dumps(
            {
                "batch_id": batch_id,
                "file_count": review["file_count"],
                "badcase_count": review["badcase_count"],
                "review_json": str(review_json),
                "review_md": str(review_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_apply(args: argparse.Namespace) -> int:
    if args.list_models:
        print_company_model_candidates(args)
        return 0

    if not args.review_json:
        raise SystemExit("review_json is required unless --list-models is used.")

    review_path = Path(args.review_json).expanduser()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    batch_id = args.batch_id or f"{review.get('batch_id', 'batch')}-apply"
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_generated_snapshots(output_dir, APPLY_SNAPSHOT_PATTERNS)
    approved_ids = load_approved_ids(args, review)
    if not approved_ids:
        raise SystemExit("No approved case ids. Use --approve-all or provide --approval-file.")

    file_results = []
    for index, file_record in enumerate(review.get("files") or [], start=1):
        result = apply_file_record(file_record, approved_ids, output_dir, args)
        result["apply_event_id"] = f"{batch_id}-apply-{index:04d}"
        result["residual_scan_event_id"] = f"{batch_id}-residual-scan-{index:04d}"
        file_results.append(result)
        if result["approved_case_count"]:
            append_round_event(Path(args.log).expanduser(), apply_round_event(result))
            append_round_event(Path(args.log).expanduser(), residual_round_event(result))

    conclusion = {
        "batch_id": batch_id,
        "created_at": utc_timestamp(),
        "mode": "apply",
        "review_batch_id": review.get("batch_id"),
        "file_count": len(file_results),
        "approved_case_count": sum(item["approved_case_count"] for item in file_results),
        "applied_case_count": sum(item["applied_case_count"] for item in file_results),
        "fixed_case_count": sum(item.get("fixed_case_count", 0) for item in file_results),
        "unsupported_case_count": sum(len(item["unsupported_cases"]) for item in file_results),
        "verification_failure_count": sum(len(item.get("verification_failures") or []) for item in file_results),
        "residual_badcase_count": sum(item["residual_badcase_count"] for item in file_results),
        "failure_category_counts": failure_category_counts(file_results),
        "files": file_results,
    }
    conclusion["conclusion_sections"] = build_batch_conclusion_sections(conclusion)
    conclusion_json = output_dir / LATEST_APPLY_CONCLUSION_JSON
    conclusion_md = output_dir / LATEST_APPLY_CONCLUSION_MD
    conclusion_json.write_text(json.dumps(conclusion, ensure_ascii=False, indent=2), encoding="utf-8")
    conclusion_md.write_text(render_apply_conclusion_markdown(conclusion), encoding="utf-8")

    print(
        json.dumps(
            {
                "batch_id": batch_id,
                "approved_case_count": conclusion["approved_case_count"],
                "applied_case_count": conclusion["applied_case_count"],
                "fixed_case_count": conclusion["fixed_case_count"],
                "unsupported_case_count": conclusion["unsupported_case_count"],
                "verification_failure_count": conclusion["verification_failure_count"],
                "residual_badcase_count": conclusion["residual_badcase_count"],
                "failure_category_counts": conclusion["failure_category_counts"],
                "conclusion_json": str(conclusion_json),
                "conclusion_md": str(conclusion_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_continue_residual(args: argparse.Namespace) -> int:
    if args.list_models:
        print_company_model_candidates(args)
        return 0

    conclusion_path = Path(args.conclusion_json).expanduser()
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    batch_id = args.batch_id or f"{conclusion.get('batch_id', 'batch')}-residual-continue"
    output_dir = Path(args.review_output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_generated_snapshots(output_dir, REVIEW_SNAPSHOT_PATTERNS)
    review = build_residual_continue_review(conclusion, batch_id=batch_id)

    review_json = output_dir / LATEST_REVIEW_JSON
    review_md = output_dir / LATEST_REVIEW_MD
    review_json.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    review_md.write_text(render_review_markdown(review), encoding="utf-8")
    for file_record in review.get("files") or []:
        append_round_event(Path(args.log).expanduser(), scan_round_event(file_record))

    payload = {
        "batch_id": batch_id,
        "residual_case_count": review["badcase_count"],
        "review_json": str(review_json),
        "review_md": str(review_md),
    }
    if args.approve_all and review["badcase_count"]:
        apply_args = argparse.Namespace(
            list_models=False,
            review_json=str(review_json),
            approve_all=True,
            approval_file=None,
            apply_backend=args.apply_backend,
            url=args.url,
            provider=args.provider,
            model=args.model,
            model_contains=args.model_contains,
            max_completion_tokens=args.max_completion_tokens,
            output_dir=args.apply_output_dir,
            log=args.log,
            batch_id=f"{batch_id}-apply",
        )
        run_apply(apply_args)
        payload["applied"] = True
        payload["conclusion_json"] = str(Path(args.apply_output_dir).expanduser() / LATEST_APPLY_CONCLUSION_JSON)
        payload["conclusion_md"] = str(Path(args.apply_output_dir).expanduser() / LATEST_APPLY_CONCLUSION_MD)
    else:
        payload["applied"] = False
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_residual_continue_review(conclusion: dict[str, Any], *, batch_id: str) -> dict[str, Any]:
    sections = conclusion.get("conclusion_sections") or build_batch_conclusion_sections(conclusion)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(conclusion.get("files") or [], start=1):
        residual_cases = item.get("residual_badcases") or []
        if not residual_cases:
            continue
        updated_file = str(item.get("updated_file") or item.get("source_file") or "")
        failures_by_residual_id = {
            str(failure.get("residual_case_id")): failure
            for failure in item.get("verification_failures") or []
            if failure.get("residual_case_id")
        }
        continued_cases = []
        for case in residual_cases:
            case_copy = dict(case)
            failure = failures_by_residual_id.get(str(case_copy.get("id"))) or {}
            next_step = str(
                failure.get("next_experiment")
                or sections.get("next_action")
                or case_copy.get("recommendation")
                or ""
            )
            case_copy["recommendation"] = (
                "Continue from the previous residual conclusion. "
                f"Next step: {next_step} "
                "Use this as the bounded repair target; do not revisit already verified cases."
            ).strip()
            case_copy["parent_case_id"] = failure.get("case_id")
            case_copy["parent_apply_batch_id"] = conclusion.get("batch_id")
            case_copy["source"] = case_copy.get("source") or "residual_continue"
            continued_cases.append(case_copy)
        record = {
            "path": updated_file,
            "file_name": Path(updated_file).name if updated_file else str(item.get("source_file") or "unknown"),
            "file_hash": "",
            "status": "scanned",
            "warnings": [
                "Residual continuation review generated from the previous apply conclusion.",
            ],
            "parse_error": None,
            "badcase_count": len(continued_cases),
            "badcases": continued_cases,
            "analysis_error_count": 0,
            "analysis_errors": [],
            "scan_event_id": f"{batch_id}-residual-review-{index:04d}",
            "batch_id": batch_id,
            "parent_apply_batch_id": conclusion.get("batch_id"),
            "parent_conclusion_next_action": sections.get("next_action"),
        }
        records.append(record)
    return {
        "batch_id": batch_id,
        "created_at": utc_timestamp(),
        "mode": "residual_continue",
        "judge": "residual_conclusion",
        "input_count": len(records),
        "file_count": len(records),
        "badcase_count": sum(record["badcase_count"] for record in records),
        "files": records,
        "approval_template": {
            "approved_case_ids": [],
            "notes": "Approve these residual cases to continue the next repair cycle.",
        },
        "parent_apply_batch_id": conclusion.get("batch_id"),
        "parent_next_action": sections.get("next_action"),
    }


def build_judge_settings(args: argparse.Namespace) -> LLMSettings:
    if args.judge == "company":
        return LLMSettings(
            backend="company_api",
            provider=DEFAULT_COMPANY_PROVIDER,
            base_url=DEFAULT_COMPANY_URL,
            model=args.model or DEFAULT_COMPANY_MODEL,
        )
    if args.judge == "openai":
        return LLMSettings(backend="openai", model=args.model or os.getenv("PROMPT_OPTIMIZER_MODEL", "gpt-4o-mini"))
    return LLMSettings(backend="openai", model="local-scan-only")


def cleanup_generated_snapshots(output_dir: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def discover_files(paths: list[Path], pattern: str, recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            raise SystemExit(f"Input path not found: {path}")
        iterator = path.rglob(pattern) if recursive else path.glob(pattern)
        for candidate in iterator:
            if candidate.is_file() and not should_skip(candidate):
                files.append(candidate)
    return sorted(dict.fromkeys(path.resolve() for path in files))


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def scan_file(path: Path, *, judge: str, settings: LLMSettings) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    parsed = parse_conversation_json(raw)
    base = {
        "path": str(path),
        "file_name": path.name,
        "file_hash": sha1_text(raw),
        "status": "parsed" if parsed.error is None and parsed.data is not None else "parse_error",
        "warnings": parsed.warnings,
        "parse_error": parsed.error,
        "badcase_count": 0,
        "badcases": [],
    }
    if parsed.error or parsed.data is None:
        return base

    data = parsed.data
    cases = _local_prompt_rule_bad_cases(data) if judge == "local" else analyze_bad_cases(data, llm_settings=settings)
    badcases = [bad_case_record(case) for case in cases if case.turn_index >= 0]
    errors = [bad_case_record(case) for case in cases if case.turn_index < 0]
    base.update(
        {
            "status": "scanned",
            "system_prompt_hash": sha1_text(data.system_prompt),
            "system_prompt_len": len(data.system_prompt),
            "turn_count": len(data.interactions),
            "tools_count": len(data.tools or {}),
            "badcase_count": len(badcases),
            "badcases": badcases,
            "analysis_error_count": len(errors),
            "analysis_errors": errors,
        }
    )
    return base


def apply_file_record(
    file_record: dict[str, Any],
    approved_ids: set[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source = Path(str(file_record["path"]))
    raw = source.read_text(encoding="utf-8")
    parsed = parse_conversation_json(raw)
    if parsed.error or parsed.data is None:
        return apply_result_base(file_record, "parse_error", [], [], [], parsed.error or "parse failed")

    approved_cases = [
        case
        for case in file_record.get("badcases") or []
        if str(case.get("id") or "") in approved_ids
    ]
    if not approved_cases:
        return apply_result_base(file_record, "no_approved_cases", [], [], [], None)

    data = parsed.data
    tool_cases = []
    prompt_cases = []
    for case in approved_cases:
        required_tool = required_tool_for_case(data, case)
        if required_tool:
            tool_cases.append((case, required_tool))
        else:
            prompt_cases.append(case)

    if prompt_cases:
        updated_data, applied_case_records, unsupported_cases, apply_meta = apply_prompt_cases_with_llm(
            data=data,
            prompt_cases=prompt_cases,
            tool_cases=tool_cases,
            args=args,
        )
    else:
        updated_data, applied_case_records, unsupported_cases, apply_meta = apply_required_tool_cases(
            data=data,
            tool_cases=tool_cases,
        )

    residual_bad_case_objects = [
        case
        for case in _local_prompt_rule_bad_cases(updated_data)
        if case.turn_index >= 0
    ]
    residual_cases = [bad_case_record(case) for case in residual_bad_case_objects]
    verification_failures = build_verification_failures(applied_case_records, residual_cases)
    fixed_case_count = max(0, len(applied_case_records) - len(verification_failures))
    rerun_result_objects = apply_meta.pop("_rerun_result_objects", [])
    conclusion_settings = apply_meta.pop("_llm_settings", None)
    app_conclusion = build_app_style_conclusion(
        before_data=data,
        updated_data=updated_data,
        applied_case_records=applied_case_records,
        residual_bad_case_objects=residual_bad_case_objects,
        rerun_result_objects=rerun_result_objects,
        llm_settings=conclusion_settings,
    )

    output_path = output_dir / f"{source.stem}_updated.json"
    output_path.write_text(
        json.dumps(conversation_payload(updated_data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = apply_result_base(
        file_record,
        "applied" if applied_case_records else "no_supported_cases",
        approved_cases,
        applied_case_records,
        unsupported_cases,
        None,
    )
    result.update(
        {
            "updated_file": str(output_path),
            "prompt_changed": updated_data.system_prompt != data.system_prompt,
            "before_hash": sha1_text(data.system_prompt),
            "after_hash": sha1_text(updated_data.system_prompt),
            "before_version": "external Original",
            "after_version": "batch Apply" if updated_data.system_prompt != data.system_prompt else "external Original",
            "residual_badcase_count": len(residual_cases),
            "residual_badcases": residual_cases,
            "fixed_case_count": fixed_case_count,
            "verification_failures": verification_failures,
            "app_conclusion": app_conclusion,
            "failure_category_counts": failure_category_counts(
                [
                    {
                        "unsupported_cases": unsupported_cases,
                        "verification_failures": verification_failures,
                    }
                ]
            ),
            "required_tools": sorted(
                {
                    str(case.get("required_tool"))
                    for case in applied_case_records
                    if case.get("required_tool")
                }
            ),
            **apply_meta,
        }
    )
    return result


def apply_required_tool_cases(
    *,
    data: ConversationData,
    tool_cases: list[tuple[dict[str, Any], str]],
) -> tuple[ConversationData, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    replacements: dict[int, Interaction] = {}
    applied_case_records: list[dict[str, Any]] = []
    for case, required_tool in tool_cases:
        target_turn = int(case["turn_index"])
        query = query_for_case(data, case, target_turn)
        replacements[target_turn] = required_tool_interaction(target_turn, required_tool, query)
        applied_case_records.append(
            {
                **case,
                "applied": True,
                "fix_type": "conversation_only_required_tool",
                "repair_class": "missing_required_tool_call",
                "required_tool": required_tool,
            }
        )
    updated_interactions = replace_interactions_with_placeholders(data.interactions, replacements)
    return (
        data.model_copy(update={"interactions": updated_interactions}),
        applied_case_records,
        [],
        {
            "apply_mode": "required_tool_replacement",
            "rerun_attempted": False,
            "rerun_target_turns": sorted(replacements),
            "rerun_errors": [],
            "prompt_edit_summaries": [],
        },
    )


def apply_prompt_cases_with_llm(
    *,
    data: ConversationData,
    prompt_cases: list[dict[str, Any]],
    tool_cases: list[tuple[dict[str, Any], str]],
    args: argparse.Namespace,
) -> tuple[ConversationData, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    settings = build_apply_settings(args, data)
    working_prompt = data.system_prompt
    current_interactions = data.interactions
    applied_case_records: list[dict[str, Any]] = []
    unsupported_cases: list[dict[str, Any]] = []
    prompt_edit_summaries: list[dict[str, str]] = []
    rerun_results = []
    rerun_errors: list[str] = []
    rerun_target_turns: set[int] = set()
    rerun_batches: list[dict[str, Any]] = []
    total_prompt_edit_retries = 0
    applied_prompt_cases: list[dict[str, Any]] = []

    for case in prompt_cases:
        bad_case = bad_case_from_record(case)
        case_data = data.model_copy(
            update={
                "system_prompt": working_prompt,
                "interactions": current_interactions,
            }
        )
        optimization = apply_recommendation_to_system_prompt(
            data=case_data,
            current_system_prompt=working_prompt,
            bad_case=bad_case,
            llm_settings=settings,
        )
        optimization, case_retry_count = retry_prompt_edit_if_unchanged_batch(
            data=case_data,
            current_prompt=working_prompt,
            bad_case=bad_case,
            llm_settings=settings,
            optimization=optimization,
        )
        total_prompt_edit_retries += case_retry_count
        prompt_changed = optimization.optimized_prompt != working_prompt
        prompt_edit_summaries.append(
            {
                "case_id": str(case.get("id") or ""),
                "rationale": optimization.rationale,
                "applied_feedback_summary": optimization.applied_feedback_summary,
                "prompt_changed": str(prompt_changed),
                "retry_count": str(case_retry_count),
            }
        )
        if prompt_edit_backend_rejected(optimization.rationale):
            unsupported_cases.append(
                {
                    **case,
                    "failure_category": classify_prompt_edit_failure(
                        data=case_data,
                        case=case,
                        rationale=optimization.rationale,
                    ),
                    "reason": "Prompt-edit backend did not produce an acceptable in-place patch/full prompt.",
                    "rationale": optimization.rationale,
                }
            )
            continue
        if prompt_changed:
            working_prompt = optimization.optimized_prompt
        applied_prompt_cases.append(case)
        applied_case_records.append(
            {
                **case,
                "applied": True,
                "fix_type": "prompt_edit_targeted_rerun",
                "repair_class": "fixable_by_prompt_clarification",
                "applied_feedback_summary": optimization.applied_feedback_summary,
                "prompt_changed": prompt_changed,
            }
        )

    prompt_target_turns = rerun_targets_for_case_records(data, applied_prompt_cases)
    required_tools_by_turn = required_tools_for_case_records(data, tool_cases)
    required_exact_responses_by_turn = exact_responses_for_case_records(data, applied_prompt_cases)
    combined_target_turns = (
        set(prompt_target_turns)
        | set(required_tools_by_turn)
        | set(required_exact_responses_by_turn)
    )
    if combined_target_turns:
        rerun_data = data.model_copy(
            update={"system_prompt": working_prompt, "interactions": current_interactions}
        )
        rerun_results = rerun_conversation(
            data=rerun_data,
            optimized_prompt=working_prompt,
            llm_settings=settings,
            target_assistant_turn_indices=combined_target_turns,
            required_tools_by_turn=required_tools_by_turn,
            required_exact_responses_by_turn=required_exact_responses_by_turn,
        )
        rerun_errors.extend(result.error for result in rerun_results if result.error)
        rerun_target_turns.update(combined_target_turns)
        rerun_batches.append(
            {
                "case_id": "selected",
                "case_error_type": "selected_badcases",
                "target_turns": sorted(combined_target_turns),
                "required_tools_by_turn": {
                    str(turn): tool for turn, tool in sorted(required_tools_by_turn.items())
                },
                "required_exact_response_turns": sorted(required_exact_responses_by_turn),
                "result_count": len(rerun_results),
                "error_count": len([result for result in rerun_results if result.error]),
            }
        )
        current_interactions = build_rerun_interactions(rerun_data, rerun_results)

    for case, required_tool in tool_cases:
        applied_case_records.append(
            {
                **case,
                "applied": True,
                "fix_type": "llm_rerun_required_tool",
                "repair_class": "missing_required_tool_call",
                "required_tool": required_tool,
            }
        )

    updated_data = data.model_copy(
        update={
            "system_prompt": working_prompt,
            "interactions": current_interactions,
        }
    )
    return (
        updated_data,
        applied_case_records,
        unsupported_cases,
        {
            "apply_mode": "llm_prompt_edit_targeted_rerun",
            "apply_backend": settings.backend,
            "apply_model": settings.model,
            "apply_provider": settings.provider,
            "rerun_attempted": bool(rerun_results),
            "rerun_target_turns": sorted(rerun_target_turns),
            "rerun_batches": rerun_batches,
            "rerun_errors": rerun_errors,
            "rerun_results": [result.__dict__ for result in rerun_results],
            "prompt_edit_summaries": prompt_edit_summaries,
            "prompt_edit_retries": total_prompt_edit_retries,
            "_rerun_result_objects": rerun_results,
            "_llm_settings": settings,
        },
    )


def retry_prompt_edit_if_unchanged_batch(
    *,
    data: ConversationData,
    current_prompt: str,
    bad_case: BadCase,
    llm_settings: LLMSettings,
    optimization: Any,
) -> tuple[Any, int]:
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
        force_prompt_edit=True,
    )
    if retry_optimization.optimized_prompt == current_prompt:
        return optimization, 1
    return retry_optimization, 1


def apply_result_base(
    file_record: dict[str, Any],
    status: str,
    approved_cases: list[dict[str, Any]],
    applied_cases: list[dict[str, Any]],
    unsupported_cases: list[dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    return {
        "source_file": file_record.get("path"),
        "scan_event_id": file_record.get("scan_event_id"),
        "status": status,
        "error": error,
        "approved_case_count": len(approved_cases),
        "applied_case_count": len(applied_cases),
        "approved_cases": approved_cases,
        "applied_cases": applied_cases,
        "unsupported_cases": unsupported_cases,
        "residual_badcase_count": 0,
        "residual_badcases": [],
        "fixed_case_count": 0,
        "verification_failures": [],
        "failure_category_counts": {},
    }


def build_app_style_conclusion(
    *,
    before_data: ConversationData,
    updated_data: ConversationData,
    applied_case_records: list[dict[str, Any]],
    residual_bad_case_objects: list[BadCase],
    rerun_result_objects: list[Any],
    llm_settings: LLMSettings | None,
) -> str:
    if not applied_case_records:
        return "Conclusion skipped because no approved case was applied."
    applied_summary = "Applied selected bad cases: " + " ".join(
        applied.get("applied_feedback_summary")
        or f"Applied {applied.get('source')}:{applied.get('turn_index')}:{applied.get('error_type')}."
        for applied in applied_case_records
    )
    conclusion = generate_experiment_conclusion(
        data=before_data,
        before_prompt=before_data.system_prompt,
        optimized_prompt=updated_data.system_prompt,
        rerun_results=rerun_result_objects,
        updated_interactions=updated_data.interactions,
        post_rerun_bad_cases=residual_bad_case_objects,
        applied_feedback_summary=applied_summary,
        llm_settings=llm_settings,
        post_rerun_scan_status="completed",
    )
    if residual_bad_case_objects:
        residual_summary = (
            f"Residual scan found {len(residual_bad_case_objects)} remaining badcase(s); "
            "treat the apply cycle as not verified fixed."
        )
        if "successfully" in conclusion.lower() or "no further optimization is required" in conclusion.lower():
            return residual_summary + "\n\n" + conclusion
    return conclusion


def classify_prompt_edit_failure(
    *,
    data: ConversationData,
    case: dict[str, Any],
    rationale: str,
) -> str:
    if case_lacks_required_ground_truth(data, case):
        return "unsupported_by_missing_ground_truth"
    lowered = rationale.lower()
    backend_failure_signals = (
        "llm prompt edit failed",
        "llm prompt edit rejected",
        "unterminated string",
        "invalid json",
        "not valid json",
        "json contract",
        "failed to parse",
        "patch failed",
    )
    if any(signal in lowered for signal in backend_failure_signals):
        return "backend_failed_to_patch"
    return "fixable_by_prompt_clarification"


def prompt_edit_backend_rejected(rationale: str) -> bool:
    lowered = rationale.lower()
    return "llm prompt edit rejected" in lowered or "llm prompt edit failed" in lowered


def case_lacks_required_ground_truth(data: ConversationData, case: dict[str, Any]) -> bool:
    if case_requires_exact_spoken_message(case):
        return not prompt_has_exact_message_ground_truth(data.system_prompt)
    return False


def prompt_has_exact_message_ground_truth(system_prompt: str) -> bool:
    prompt = system_prompt.lower()
    if not any(phrase in prompt for phrase in ("message exactly", "say exactly", "deliver the following message exactly")):
        return False
    quote_chars = "\"'\u201c\u201d\u2018\u2019"
    escaped_quotes = re.escape(quote_chars)
    quoted_exact_message = re.search(
        rf"(?:message exactly|say exactly|deliver the following message exactly)\s*[:\uFF1A]?\s*"
        rf"[{escaped_quotes}][^{escaped_quotes}]{{20,}}[{escaped_quotes}]",
        system_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if quoted_exact_message:
        return True
    return "hotline" in prompt or "transfer" in prompt

def build_verification_failures(
    applied_cases: list[dict[str, Any]],
    residual_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    used_residual_indices: set[int] = set()
    for applied in applied_cases:
        residual_index, residual = matching_residual_case(applied, residual_cases, used_residual_indices)
        if residual is None:
            continue
        used_residual_indices.add(residual_index)
        failures.append(
            {
                "case_id": str(applied.get("id") or ""),
                "turn_index": applied.get("turn_index"),
                "error_type": applied.get("error_type"),
                "fix_type": applied.get("fix_type"),
                "repair_class": applied.get("repair_class"),
                "failure_category": "patch_applied_but_failed_verification",
                "reason": (
                    "A prompt change and/or targeted rerun was applied, but the residual scan "
                    "still found the same rule violation."
                ),
                "residual_case_id": residual.get("id"),
                "residual_turn_index": residual.get("turn_index"),
                "residual_evidence": residual.get("evidence"),
                "next_experiment": next_experiment_for_failed_case(applied),
            }
        )
    return failures


def matching_residual_case(
    applied: dict[str, Any],
    residual_cases: list[dict[str, Any]],
    used_indices: set[int],
) -> tuple[int, dict[str, Any] | None]:
    applied_error = str(applied.get("error_type") or "")
    applied_turn = applied.get("turn_index")
    for index, residual in enumerate(residual_cases):
        if index in used_indices:
            continue
        if residual.get("error_type") == applied_error and residual.get("turn_index") == applied_turn:
            return index, residual
    for index, residual in enumerate(residual_cases):
        if index in used_indices:
            continue
        if residual.get("error_type") == applied_error:
            return index, residual
    return -1, None


def next_experiment_for_failed_case(case: dict[str, Any]) -> str:
    error_type = str(case.get("error_type") or "")
    if error_type == "late_payment_proposal_not_rtp_closing":
        return (
            "Route recognized beyond-maximum-date triggers through deterministic backend RTP_Closing response "
            "replacement, then verify the target turn without another prompt-only retry."
        )
    if error_type == "escalation_action_not_followed":
        return "Extract or confirm the exact escalation message, then force deterministic spoken text plus any required action."
    return "Clarify the violated prompt rule into machine-checkable required and forbidden behavior, then rerun."


def failure_category_counts(file_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in file_results:
        for case in item.get("unsupported_cases") or []:
            category = str(case.get("failure_category") or "unknown")
            counts[category] = counts.get(category, 0) + 1
        for failure in item.get("verification_failures") or []:
            category = str(failure.get("failure_category") or "unknown")
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def load_approved_ids(args: argparse.Namespace, review: dict[str, Any]) -> set[str]:
    if args.approve_all:
        return {
            str(case.get("id"))
            for file_record in review.get("files") or []
            for case in file_record.get("badcases") or []
            if case.get("id")
        }
    if not args.approval_file:
        return set()
    payload = json.loads(Path(args.approval_file).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {str(item) for item in payload}
    if isinstance(payload, dict):
        ids = payload.get("approved_case_ids")
        if isinstance(ids, list):
            return {str(item) for item in ids}
    raise SystemExit("Approval file must be a list or {'approved_case_ids': [...]}.")


def build_apply_settings(args: argparse.Namespace, data: ConversationData) -> LLMSettings:
    model_info = {}
    if isinstance(data.source_meta, dict) and isinstance(data.source_meta.get("model_info"), dict):
        model_info = data.source_meta["model_info"]
    if args.apply_backend == "openai":
        return LLMSettings(
            backend="openai",
            model=args.model or os.getenv("PROMPT_OPTIMIZER_MODEL", "gpt-4o-mini"),
            max_completion_tokens=args.max_completion_tokens,
        )
    base_url = args.url or str(model_info.get("url") or DEFAULT_COMPANY_URL)
    provider, model = resolve_company_model_selection(
        url=base_url,
        fallback_provider=args.provider or str(model_info.get("provider") or DEFAULT_COMPANY_PROVIDER),
        fallback_model=args.model or str(model_info.get("model") or DEFAULT_COMPANY_MODEL),
        model_contains=args.model_contains,
    )
    return LLMSettings(
        backend="company_api",
        base_url=base_url,
        provider=provider,
        model=model,
        max_completion_tokens=args.max_completion_tokens,
    )


def print_company_model_candidates(args: argparse.Namespace) -> None:
    url = args.url or DEFAULT_COMPANY_URL
    models = list_company_models(url)
    filtered = filter_company_models(models, args.model_contains)
    print(
        json.dumps(
            {
                "url": url,
                "filter": args.model_contains,
                "count": len(filtered),
                "models": filtered,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def resolve_company_model_selection(
    *,
    url: str,
    fallback_provider: str,
    fallback_model: str,
    model_contains: str | None,
) -> tuple[str, str]:
    if not model_contains:
        return split_company_model(fallback_model, fallback_provider)

    matches = filter_company_models(list_company_models(url), model_contains)
    if len(matches) != 1:
        print(
            json.dumps(
                {
                    "error": "model_contains must match exactly one company model",
                    "filter": model_contains,
                    "count": len(matches),
                    "models": matches,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
    return split_company_model(matches[0], fallback_provider)


def filter_company_models(models: list[str], model_contains: str | None) -> list[str]:
    if not model_contains:
        return models
    needle = model_contains.lower()
    return [model for model in models if needle in model.lower()]


def split_company_model(selection: str, fallback_provider: str) -> tuple[str, str]:
    if ":" in selection:
        provider, model = selection.split(":", 1)
        return provider.strip(), model.strip()
    return fallback_provider, selection.strip()


def bad_case_from_record(case: dict[str, Any]) -> BadCase:
    return BadCase(
        turn_index=int(case.get("turn_index", -1)),
        role=str(case.get("role") or "assistant"),
        error_type=str(case.get("error_type") or "unknown"),
        evidence=str(case.get("evidence") or ""),
        recommendation=str(case.get("recommendation") or ""),
        source=str(case.get("source") or "batch_review"),
    )


def rerun_targets_for_case_records(data: ConversationData, cases: list[dict[str, Any]]) -> set[int]:
    targets: set[int] = set()
    for case in cases:
        try:
            turn_index = int(case.get("turn_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= turn_index < len(data.interactions):
            if data.interactions[turn_index].role == "assistant":
                targets.add(turn_index)
                continue
            for index in range(turn_index + 1, len(data.interactions)):
                if data.interactions[index].role == "assistant":
                    targets.add(index)
                    break
    return targets


def required_tools_for_case_records(
    data: ConversationData,
    tool_cases: list[tuple[dict[str, Any], str]],
) -> dict[int, str]:
    required: dict[int, str] = {}
    for case, required_tool in tool_cases:
        for target_index in rerun_targets_for_case_records(data, [case]):
            required[target_index] = required_tool
    return required


def exact_responses_for_case_records(
    data: ConversationData,
    prompt_cases: list[dict[str, Any]],
) -> dict[int, str]:
    exact_message = extract_exact_escalation_message(data.system_prompt)
    if not exact_message:
        return {}
    required: dict[int, str] = {}
    for case in prompt_cases:
        if str(case.get("error_type") or "").lower() != "escalation_action_not_followed":
            continue
        if not case_requires_exact_spoken_message(case):
            continue
        for target_index in rerun_targets_for_case_records(data, [case]):
            required[target_index] = exact_message
    return required


def build_rerun_interactions(data: ConversationData, rerun_results: list[Any]) -> list[Interaction]:
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
            id_prefix=f"call_batch_rerun_{index}",
        )
        rebuilt.append(
            turn.model_copy(
                update={
                    "content": new_response,
                    "tool_calls": tool_calls or None,
                    "tool_call_id": None,
                }
            )
        )
        for call in tool_calls:
            rebuilt.append(rerun_tool_placeholder_interaction(call))
    return rebuilt


def query_for_case(data: ConversationData, case: dict[str, Any], target_turn: int) -> str:
    evidence = str(case.get("evidence") or "")
    match = re.search(r"User turn\s+(\d+)", evidence)
    if match:
        user_index = int(match.group(1))
        if 0 <= user_index < len(data.interactions) and data.interactions[user_index].role == "user":
            return compact_query(data.interactions[user_index].content)
    for index in range(target_turn - 1, -1, -1):
        if data.interactions[index].role == "user":
            return compact_query(data.interactions[index].content)
    return "tool-required inquiry"


def compact_query(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:240]


def required_tool_for_case(data: ConversationData, case: dict[str, Any]) -> str | None:
    if case_requires_exact_spoken_message(case):
        return None
    if not case_is_missing_required_tool_call(case):
        return None
    tool_names = available_tool_names(data.tools)
    if not tool_names:
        return None
    text = case_text(case)
    lowered = text.lower()
    for tool_name in tool_names:
        if tool_name.lower() in lowered:
            return tool_name
    return best_matching_tool_name(tool_names, text)


def case_is_missing_required_tool_call(case: dict[str, Any]) -> bool:
    lowered = case_text(case).lower()
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


def case_requires_exact_spoken_message(case: dict[str, Any]) -> bool:
    lowered = case_text(case).lower()
    if "exact" not in lowered:
        return False
    if str(case.get("error_type") or "").lower() == "escalation_action_not_followed":
        return any(
            phrase in lowered
            for phrase in (
                "spoken text",
                "deliver the following message exactly",
                "required message",
                "exact configured escalation action",
                "hotline ending",
            )
        )
    return any(
        phrase in lowered
        for phrase in (
            "spoken text",
            "exact message",
            "message exactly",
            "required wording",
        )
    )


def available_tool_names(tools: dict[str, Any] | None) -> list[str]:
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
    text_tokens = set(normalized_tokens(text))
    best_name = None
    best_score = 0
    for tool_name in tool_names:
        tool_tokens = [
            token
            for token in normalized_tokens(tool_name)
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
        if "search" in tool_name.lower() and "search" in text.lower():
            score += 1
        if "retrieval" in tool_name.lower() and "retrieval" in text.lower():
            score += 1
        if score > best_score:
            best_name = tool_name
            best_score = score
    return best_name if best_score > 0 else None


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())


def case_text(case: dict[str, Any]) -> str:
    return " ".join(
        str(case.get(key) or "")
        for key in ("id", "error_type", "evidence", "recommendation")
    )


def required_tool_interaction(turn_index: int, tool_name: str, query: str) -> Interaction:
    arguments = json.dumps({"query": query}, ensure_ascii=False)
    wrapper = f"<function-call>{tool_name}:{arguments}</function-call>"
    return Interaction(
        role="assistant",
        content=wrapper,
        tool_calls=function_call_wrappers_to_tool_calls(wrapper, id_prefix=f"call_batch_{turn_index}") or None,
    )


def replace_interactions_with_placeholders(
    interactions: list[Interaction],
    replacements: dict[int, Interaction],
) -> list[Interaction]:
    updated: list[Interaction] = []
    for index, turn in enumerate(interactions):
        replacement = replacements.get(index)
        if replacement is None:
            updated.append(turn)
            continue
        updated.append(replacement)
        for call in replacement.tool_calls or []:
            updated.append(rerun_tool_placeholder_interaction(call))
    return updated


def rerun_tool_placeholder_interaction(tool_call: dict[str, Any]) -> Interaction:
    function = tool_call.get("function") if isinstance(tool_call, dict) else {}
    if not isinstance(function, dict):
        function = {}
    payload = {
        "status": "not_executed",
        "note": (
            "The batch apply requested this tool call. No local tool executor is configured, "
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


def conversation_payload(data: ConversationData) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "system_prompt": data.system_prompt,
        "interactions": [turn.model_dump(exclude_none=True) for turn in data.interactions],
    }
    if data.tools:
        payload["tools"] = data.tools
    if data.source_meta:
        payload["source_meta"] = data.source_meta
    return payload


def bad_case_record(case: BadCase) -> dict[str, Any]:
    record = {
        "turn_index": case.turn_index,
        "role": case.role,
        "error_type": case.error_type,
        "source": case.source,
        "evidence": case.evidence,
        "recommendation": case.recommendation,
    }
    record["id"] = bad_case_id(record)
    return record


def bad_case_id(record: dict[str, Any]) -> str:
    payload = {
        "turn_index": record.get("turn_index"),
        "role": record.get("role"),
        "error_type": record.get("error_type"),
        "source": record.get("source"),
        "evidence": record.get("evidence"),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def scan_round_event(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "scan",
        "event_id": record["scan_event_id"],
        "batch_id": record["batch_id"],
        "timestamp": utc_timestamp(),
        "trigger": "batch_scan",
        "status": "completed" if record.get("status") == "scanned" else "failed",
        "status_level": "success" if record.get("status") == "scanned" else "error",
        "status_message": f"Batch scan complete. {record.get('badcase_count', 0)} trace item(s).",
        "source_file": record.get("path"),
        "conversation_hash": record.get("file_hash"),
        "prompt_version_label": "external Original",
        "prompt_hash": record.get("system_prompt_hash"),
        "prompt_len": record.get("system_prompt_len"),
        "conversation_view": "Original",
        "badcase_count": record.get("badcase_count", 0),
        "analysis_error_count": record.get("analysis_error_count", 0),
        "badcases": record.get("badcases", []),
        "analysis_errors": record.get("analysis_errors", []),
        "parse_error": record.get("parse_error"),
    }


def apply_round_event(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "apply",
        "event_id": result["apply_event_id"],
        "timestamp": utc_timestamp(),
        "action": "Batch apply approved cases",
        "human_decision": "approved",
        "parent_scan_round_id": result.get("scan_event_id"),
        "source_file": result.get("source_file"),
        "updated_file": result.get("updated_file"),
        "approved_trace_count": result.get("approved_case_count", 0),
        "applied_trace_count": result.get("applied_case_count", 0),
        "fixed_trace_count": result.get("fixed_case_count", 0),
        "badcases": result.get("applied_cases", []),
        "unsupported_cases": result.get("unsupported_cases", []),
        "verification_failures": result.get("verification_failures", []),
        "failure_category_counts": result.get("failure_category_counts", {}),
        "required_tools": result.get("required_tools", []),
        "apply_mode": result.get("apply_mode"),
        "apply_backend": result.get("apply_backend"),
        "apply_model": result.get("apply_model"),
        "rerun_attempted": result.get("rerun_attempted", False),
        "rerun_target_turns": result.get("rerun_target_turns", []),
        "rerun_errors": result.get("rerun_errors", []),
        "prompt_edit_summaries": result.get("prompt_edit_summaries", []),
        "app_conclusion": result.get("app_conclusion"),
        "prompt_changed": result.get("prompt_changed", False),
        "before_hash": result.get("before_hash"),
        "after_hash": result.get("after_hash"),
        "before_version": result.get("before_version"),
        "after_version": result.get("after_version"),
        "status": result.get("status"),
        "status_message": batch_apply_status_message(result),
    }


def residual_round_event(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "scan",
        "event_id": result["residual_scan_event_id"],
        "timestamp": utc_timestamp(),
        "trigger": "batch_post_apply_residual_scan",
        "status": "completed",
        "status_level": "success" if result.get("residual_badcase_count", 0) == 0 else "warning",
        "status_message": f"Residual scan complete. {result.get('residual_badcase_count', 0)} trace item(s).",
        "source_file": result.get("updated_file"),
        "conversation_view": "Updated",
        "badcase_count": result.get("residual_badcase_count", 0),
        "analysis_error_count": 0,
        "badcases": result.get("residual_badcases", []),
        "verification_failures": result.get("verification_failures", []),
        "failure_category_counts": result.get("failure_category_counts", {}),
        "analysis_errors": [],
        "parent_apply_round_id": result.get("apply_event_id"),
    }


def append_round_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def render_review_markdown(review: dict[str, Any]) -> str:
    lines = [
        f"# Batch Badcase Review: {review['batch_id']}",
        "",
        f"- Files scanned: {review['file_count']}",
        f"- Candidate badcases: {review['badcase_count']}",
        "",
    ]
    for file_record in review.get("files") or []:
        lines.extend(
            [
                f"## {file_record.get('file_name')}",
                "",
                f"- Path: `{file_record.get('path')}`",
                f"- Scan round: `{file_record.get('scan_event_id')}`",
                f"- Status: `{file_record.get('status')}`",
                f"- Badcases: {file_record.get('badcase_count', 0)}",
                "",
            ]
        )
        if file_record.get("parse_error"):
            lines.extend([f"Parse error: {file_record['parse_error']}", ""])
            continue
        if not file_record.get("badcases"):
            lines.extend(["No candidate badcases.", ""])
            continue
        for index, case in enumerate(file_record.get("badcases") or [], start=1):
            lines.extend(
                [
                    f"### {index}. `{case['id']}` turn {case['turn_index']} `{case['error_type']}`",
                    "",
                    f"Evidence: {case['evidence']}",
                    "",
                    f"Recommendation: {case['recommendation']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Approval",
            "",
            "Use `--approve-all` after human review, or create an approval JSON:",
            "",
            "```json",
            json.dumps({"approved_case_ids": []}, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _legacy_render_apply_conclusion_markdown(conclusion: dict[str, Any]) -> str:
    return render_apply_conclusion_markdown(conclusion)

def build_batch_conclusion_sections(conclusion: dict[str, Any]) -> dict[str, Any]:
    approved = int(conclusion.get("approved_case_count") or 0)
    applied = int(conclusion.get("applied_case_count") or 0)
    fixed = int(conclusion.get("fixed_case_count") or 0)
    residual = int(conclusion.get("residual_badcase_count") or 0)
    unsupported = int(conclusion.get("unsupported_case_count") or 0)
    failures = int(conclusion.get("verification_failure_count") or 0)

    file_evidence = [_batch_file_conclusion_evidence(item) for item in conclusion.get("files") or []]
    fixed_turns = sorted(
        {
            int(case.get("turn_index"))
            for item in conclusion.get("files") or []
            for case in item.get("applied_cases") or []
            if case.get("turn_index") is not None
            and not any(
                failure.get("case_id") == case.get("id")
                for failure in item.get("verification_failures") or []
            )
        }
    )
    residual_turns = sorted(
        {
            int(case.get("turn_index"))
            for item in conclusion.get("files") or []
            for case in item.get("residual_badcases") or []
            if case.get("turn_index") is not None
        }
    )

    if residual == 0 and failures == 0 and unsupported == 0 and fixed == approved:
        verdict = (
            f"Verified fixed. All {approved} approved badcase(s) passed the post-apply residual scan. "
            f"Verified turns: {fixed_turns or 'none recorded'}. Evidence: residual_badcase_count=0, "
            f"verification_failure_count=0, unsupported_case_count=0."
        )
    elif fixed > 0:
        verdict = (
            f"Partially fixed. {fixed} of {approved} approved badcase(s) passed verification; "
            f"{residual} residual badcase(s), {failures} verification failure(s), and {unsupported} unsupported case(s) remain. "
            f"Verified turns: {fixed_turns or 'none recorded'}; residual turns: {residual_turns or 'none recorded'}."
        )
    else:
        verdict = (
            f"Not fixed. Changes were applied to {applied} of {approved} approved badcase(s), but verification confirmed "
            f"{residual} residual badcase(s), {failures} verification failure(s), and {unsupported} unsupported case(s). "
            f"Residual turns: {residual_turns or 'none recorded'}."
        )
    if file_evidence:
        surface = "\n".join(f"- {evidence['surface_summary']}" for evidence in file_evidence)
        deep = "\n".join(f"- {evidence['deep_summary']}" for evidence in file_evidence)
    else:
        surface = "- No per-file surface evidence was recorded."
        deep = "- No per-file backend evidence was recorded."
    root_cause = _build_root_cause_analysis(conclusion, file_evidence)
    diagnosis = (
        f"Failure categories: {format_failure_category_counts(conclusion.get('failure_category_counts') or {})}.\n\n"
        f"Root cause analysis:\n{root_cause}\n\n"
        f"Evidence data (surface):\n{surface}\n\n"
        f"Evidence data (deep):\n{deep}"
    )

    next_experiments = [
        next_experiment_for_failed_case(failure)
        for item in conclusion.get("files") or []
        for failure in item.get("verification_failures") or []
    ]
    if residual == 0 and failures == 0 and unsupported == 0:
        next_action = (
            "Primary action: keep the verified prompt/conversation version and stop. "
            "Acceptance criteria already met: no residual badcases, no verification failures, and no unsupported cases. "
            "Do not start another scan unless requested."
        )
    elif next_experiments:
        next_action = (
            f"Primary action: {next_experiments[0]} "
            "Why: prompt edit and model rerun completed, but the residual scan still found the same behavior, so repeating "
            "another prompt-only edit has weak evidence of benefit. Implementation: detect the verified trigger before "
            "generation, replace only the affected assistant turn with the configured terminal response, preserve unrelated "
            "turns, then run exactly one residual scan. Acceptance criteria: the target residual case disappears and adjacent "
            "valid-date/regression cases remain unchanged. Fallback: if deterministic routing cannot be implemented safely, "
            "mark the case unsupported and request the missing exact response or trigger ground truth."
        )
    elif unsupported:
        next_action = "Supply the missing ground truth for unsupported cases, then rebuild the Repair Plan before Apply."
    else:
        next_action = "Review the residual evidence and run one bounded experiment that targets the recorded failure category."

    return {
        "verification_verdict": verdict,
        "badcase_diagnosis_and_backend_evidence": diagnosis,
        "next_action": next_action,
        "evidence": file_evidence,
    }


def _build_root_cause_analysis(conclusion: dict[str, Any], file_evidence: list[dict[str, Any]]) -> str:
    failures = [
        failure
        for item in conclusion.get("files") or []
        for failure in item.get("verification_failures") or []
    ]
    if not failures:
        if int(conclusion.get("residual_badcase_count") or 0) == 0:
            return (
                "- Residual scan found no remaining badcases. The prompt/rerun changes are treated as execution evidence, "
                "while the residual scan is the correctness evidence."
            )
        return "- Residual badcases remain, but no structured verification failure was recorded; inspect the residual cases directly."

    lines = []
    evidence_by_source = {evidence.get("source_file"): evidence for evidence in file_evidence}
    for item in conclusion.get("files") or []:
        evidence = evidence_by_source.get(item.get("source_file")) or {}
        logprob_read = str(evidence.get("logprob_interpretation") or "no logprob interpretation available")
        for failure in item.get("verification_failures") or []:
            error_type = str(failure.get("error_type") or "unknown")
            turn = failure.get("turn_index")
            residual_turn = failure.get("residual_turn_index")
            case_id = failure.get("case_id") or "unknown"
            residual_id = failure.get("residual_case_id") or "unknown"
            if error_type == "late_payment_proposal_not_rtp_closing":
                cause = (
                    "The prompt edit and targeted rerun executed, but the model still routed a beyond-maximum-date "
                    "payment proposal into the negotiation/proposal-clarification path instead of terminal RTP_Closing. "
                    "That points to a branch-priority or routing failure, not a missing scan."
                )
            elif error_type == "escalation_action_not_followed":
                cause = (
                    "The model continued the conversation after an escalation trigger instead of treating the exact "
                    "escalation action as terminal, indicating the escalation branch was not dominant enough."
                )
            elif error_type == "busy_availability_check_skipped":
                cause = (
                    "The model skipped the availability gate and entered the payment sequence too early, indicating "
                    "the busy/unavailable branch priority was under-specified."
                )
            else:
                cause = "The rerun still matched the residual rule violation, so the applied change did not control the relevant branch."
            lines.append(
                f"- `{case_id}` -> `{residual_id}` turn {turn or residual_turn}: {cause} "
                f"Logprob read: {logprob_read}."
            )
    return "\n".join(lines)


def _batch_file_conclusion_evidence(item: dict[str, Any]) -> dict[str, Any]:
    reruns = item.get("rerun_results") or []
    prompt_edits = item.get("prompt_edit_summaries") or []
    residual_cases = item.get("residual_badcases") or []
    rerun_details = []
    for result in reruns:
        diagnostics = result.get("response_diagnostics") if isinstance(result, dict) else None
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        logprobs = diagnostics.get("logprobs") if isinstance(diagnostics.get("logprobs"), dict) else {}
        rerun_details.append(
            {
                "turn": result.get("assistant_turn_index"),
                "old_response": _truncate_conclusion_text(str(result.get("old_assistant_response") or ""), 350),
                "new_response": _truncate_conclusion_text(str(result.get("new_assistant_response") or ""), 500),
                "error": result.get("error"),
                "request_id": diagnostics.get("request_id"),
                "provider": diagnostics.get("provider"),
                "model": diagnostics.get("model"),
                "logprobs_available": logprobs.get("available") is True,
                "avg_logprob": logprobs.get("avg_logprob"),
                "min_logprob": logprobs.get("min_logprob"),
                "low_confidence_tokens": [
                    {
                        "token": token.get("token"),
                        "logprob": token.get("logprob"),
                    }
                    for token in (logprobs.get("low_confidence_tokens") or [])[:5]
                    if isinstance(token, dict)
                ],
            }
        )
    logprob_summary, logprob_interpretation = _summarize_logprobs(rerun_details, residual_cases)
    residual_summary = "; ".join(
        f"turn {case.get('turn_index')} {case.get('error_type')}"
        for case in residual_cases
    ) or "none"
    patch_summary = " | ".join(
        f"{edit.get('case_id')}: {_truncate_conclusion_text(str(edit.get('applied_feedback_summary') or ''), 140)}"
        for edit in prompt_edits
    ) or "none"
    request_ids = [str(detail.get("request_id")) for detail in rerun_details if detail.get("request_id")]
    failed_ids = [
        f"{failure.get('case_id')}->{failure.get('residual_case_id')}"
        for failure in item.get("verification_failures") or []
    ]
    file_name = Path(str(item.get("source_file") or "unknown")).name
    surface_summary = (
        f"{file_name}: applied={item.get('applied_case_count', 0)}, fixed={item.get('fixed_case_count', 0)}, "
        f"residual={item.get('residual_badcase_count', 0)}, unsupported={len(item.get('unsupported_cases') or [])}; "
        f"residual={residual_summary}; verification_failures={', '.join(failed_ids) or 'none'}."
    )
    deep_summary = (
        f"{file_name}: backend={item.get('apply_provider') or 'unknown'}/{item.get('apply_model') or 'unknown'}, "
        f"prompt_hash={item.get('before_hash') or '-'}->{item.get('after_hash') or '-'}, "
        f"prompt_changed={bool(item.get('prompt_changed'))}, rerun_targets={item.get('rerun_target_turns') or []}, "
        f"request_ids={', '.join(request_ids) or 'none'}, prompt_edits={patch_summary}, {logprob_summary}"
    )
    diagnosis = (
        f"{surface_summary} {deep_summary}"
    )
    return {
        "source_file": item.get("source_file"),
        "diagnosis": diagnosis,
        "surface_summary": surface_summary,
        "deep_summary": deep_summary,
        "rerun_details": rerun_details,
        "prompt_edit_summaries": prompt_edits,
        "residual_badcases": residual_cases,
        "logprob_summary": logprob_summary,
        "logprob_interpretation": logprob_interpretation,
    }


def _summarize_logprobs(rerun_details: list[dict[str, Any]], residual_cases: list[dict[str, Any]]) -> tuple[str, str]:
    logprob_available = [detail for detail in rerun_details if detail["logprobs_available"]]
    if not logprob_available:
        return (
            "logprobs=not available; do not attribute the outcome to token uncertainty.",
            "no usable logprob evidence was recorded",
        )
    residual_turns = {case.get("turn_index") for case in residual_cases}
    parts = []
    interpretations = []
    for detail in logprob_available:
        avg = detail.get("avg_logprob")
        minimum = detail.get("min_logprob")
        turn = detail.get("turn")
        if turn in residual_turns and isinstance(avg, (int, float)) and avg >= -0.2:
            interpretation = "stable wrong-branch preference"
        elif isinstance(minimum, (int, float)) and minimum <= -2.0:
            interpretation = "localized token uncertainty"
        else:
            interpretation = "confidence evidence only"
        interpretations.append(f"turn {turn}: {interpretation}")
        tokens = ", ".join(
            str(token.get("token")).strip()
            for token in detail.get("low_confidence_tokens") or []
            if token.get("token")
        ) or "none"
        parts.append(
            f"turn {turn} avg={_format_logprob_value(avg)}, min={_format_logprob_value(minimum)}, request={detail.get('request_id') or 'none'}, "
            f"read={interpretation}, low_tokens={tokens}"
        )
    return (
        "logprobs: " + "; ".join(parts)
        + ". Logprobs support confidence/routing diagnosis only; residual scan remains correctness evidence.",
        "; ".join(interpretations),
    )


def _format_logprob_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3g}"
    return str(value)


def _truncate_conclusion_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def render_apply_conclusion_markdown(conclusion: dict[str, Any]) -> str:
    sections = conclusion.get("conclusion_sections") or build_batch_conclusion_sections(conclusion)
    lines = [
        f"# Batch Apply Conclusion: {conclusion['batch_id']}",
        "",
        "## 1. Verification Verdict",
        "",
        str(sections["verification_verdict"]),
        "",
        "## 2. Badcase Diagnosis And Backend Evidence",
        "",
        str(sections["badcase_diagnosis_and_backend_evidence"]),
        "",
    ]
    for item in conclusion.get("files") or []:
        file_name = Path(str(item.get("source_file") or "unknown")).name
        lines.extend(
            [
                (
                    f"- `{file_name}`: updated=`{item.get('updated_file')}`, "
                    f"rounds=`{item.get('scan_event_id')}` / `{item.get('apply_event_id')}` / `{item.get('residual_scan_event_id')}`"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 3. Next Action",
            "",
            str(sections["next_action"]),
            "",
        ]
    )
    return "\n".join(lines)


def batch_apply_status_message(result: dict[str, Any]) -> str:
    if not result.get("applied_case_count"):
        return "Batch apply completed with no supported case changes."
    if result.get("apply_mode") == "llm_prompt_edit_targeted_rerun":
        if result.get("rerun_errors"):
            return "Batch prompt edit completed, but one or more targeted reruns failed."
        return "Batch prompt edit and targeted rerun completed."
    if result.get("apply_mode") == "required_tool_replacement":
        return "Batch conversation-only required-tool replacement completed."
    return "Batch apply completed with no supported case changes."


def batch_fix_summary(conclusion: dict[str, Any]) -> str:
    file_items = conclusion.get("files") or []
    prompt_changed = any(item.get("prompt_changed") for item in file_items)
    modes = sorted({str(item.get("apply_mode") or "none") for item in file_items})
    mode_text = ", ".join(modes)
    prompt_text = "A new prompt version was created." if prompt_changed else "No new prompt version was created."
    if conclusion["residual_badcase_count"] == 0:
        result_text = f"Verified fixed {conclusion.get('fixed_case_count', conclusion['applied_case_count'])} approved badcase(s)"
    else:
        result_text = (
            f"Verified fixed {conclusion.get('fixed_case_count', 0)} approved badcase(s); "
            f"applied changes for {conclusion['applied_case_count']} approved badcase(s); "
            f"{conclusion['residual_badcase_count']} badcase(s) remain after residual scan"
        )
    return (
        f"{result_text} across "
        f"{conclusion['file_count']} file(s). Apply mode: {mode_text}. {prompt_text}"
    )


def format_failure_category_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def format_failure_category_counts_zh(counts: dict[str, int]) -> str:
    if not counts:
        return "无"
    return "，".join(
        f"{failure_category_label_zh(key)}={value} 个"
        for key, value in sorted(counts.items())
    )


def format_failure_category_lines_zh(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- 无"]
    return [
        f"- {failure_category_label_zh(key)}（`{key}`）：{value} 个。{failure_category_description_zh(key)}"
        for key, value in sorted(counts.items())
    ]


def failure_category_label_zh(category: Any) -> str:
    labels = {
        "fixable_by_prompt_clarification": "可通过澄清 system prompt 修复",
        "unsupported_by_missing_ground_truth": "缺少必要事实，无法自动修复",
        "backend_failed_to_patch": "后端未生成可用 prompt patch",
        "backend_failed_to_autonomously_repair": "后端未能自动修复",
        "patch_applied_but_failed_verification": "已改动但 residual scan 未通过",
        "likely_model_or_context_limited": "疑似模型或上下文能力限制",
    }
    return labels.get(str(category or "unknown"), "未知失败类型")


def failure_category_description_zh(category: Any) -> str:
    descriptions = {
        "fixable_by_prompt_clarification": "规则或优先级边界不够清楚，适合用有界 prompt 澄清修复。",
        "unsupported_by_missing_ground_truth": "缺少 exact message、工具结果、日期、金额、热线或业务决策等必要信息，batch 不能自行编造。",
        "backend_failed_to_patch": "apply 后端未能产出符合 JSON/patch contract 的可用 prompt 改动。",
        "backend_failed_to_autonomously_repair": "所有适用的自动修复策略都失败，需要保留 residual badcase 供人工复核。",
        "patch_applied_but_failed_verification": "prompt edit 或 targeted rerun 已执行，但 residual scan 仍发现同类违规。",
        "likely_model_or_context_limited": "仅在多轮受控实验后，且 prompt/metadata 已足够明确时使用。",
    }
    return descriptions.get(str(category or "unknown"), "请查看记录的 residual evidence 和 failure category。")

def new_batch_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    raise SystemExit(main())
