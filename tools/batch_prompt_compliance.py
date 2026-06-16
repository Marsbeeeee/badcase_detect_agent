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


def as_file_uri(path: Path) -> str:
    return path.resolve().as_uri()
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
    review["review_json"] = str(review_json)
    review["review_md"] = str(review_md)
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
                "review_json_uri": as_file_uri(review_json),
                "review_md_uri": as_file_uri(review_md),
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
                "conclusion_json_uri": as_file_uri(conclusion_json),
                "conclusion_md_uri": as_file_uri(conclusion_md),
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
        "review_json_uri": as_file_uri(review_json),
        "review_md_uri": as_file_uri(review_md),
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
        payload["conclusion_json_uri"] = as_file_uri(
            Path(args.apply_output_dir).expanduser() / LATEST_APPLY_CONCLUSION_JSON
        )
        payload["conclusion_md_uri"] = as_file_uri(
            Path(args.apply_output_dir).expanduser() / LATEST_APPLY_CONCLUSION_MD
        )
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


def _conversation_interactions_from_file(source_path: str | None) -> list[Interaction] | None:
    if not source_path:
        return None
    try:
        raw = Path(source_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    parsed = parse_conversation_json(raw)
    if parsed.error or parsed.data is None:
        return None
    return parsed.data.interactions


def _parse_user_turn_from_evidence(evidence: str) -> int | None:
    match = re.search(r"User turn\s+(\d+)", evidence, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _compact_turn_text(text: str, limit: int = 1200) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _resolve_review_case_turn_context(
    case: dict[str, Any],
    interactions: list[Interaction] | None,
) -> tuple[int | None, int | None, str | None, str | None]:
    if interactions is None:
        return None, None, None, None
    try:
        assistant_turn = int(case.get("turn_index"))
    except (TypeError, ValueError):
        return None, None, None, None
    if not (0 <= assistant_turn < len(interactions)):
        return None, None, None, None
    if interactions[assistant_turn].role != "assistant":
        return None, None, None, None

    evidence = str(case.get("evidence") or "")
    user_turn = _parse_user_turn_from_evidence(evidence)
    if (
        user_turn is None
        or not (0 <= user_turn < len(interactions))
        or interactions[user_turn].role != "user"
    ):
        user_turn = None
        for index in range(assistant_turn - 1, -1, -1):
            if interactions[index].role == "user":
                user_turn = index
                break

    assistant_text = _compact_turn_text(str(interactions[assistant_turn].content))
    user_text = _compact_turn_text(str(interactions[user_turn].content)) if user_turn is not None else None
    return assistant_turn, user_turn, user_text, assistant_text


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


def _localize_review_evidence(evidence: str) -> str:
    if not evidence:
        return evidence

    localized = evidence
    localized = localized.replace("Evidence:", "证据：")
    localized = localized.replace("Violated rule:", "违反规则：")

    # Common local deterministic pattern for escalation checks.
    match = re.search(
        r'User turn (\d+) triggered escalation \(([^)]+)\) with "([^"]+)"\\.?\s*Assistant turn (\d+) did not execute the required escalation action \("([^"]+)"\)',
        localized,
    )
    if match:
        user_turn, rule_id, user_text, assistant_turn, action_text = match.groups()
        return (
            f"用户回合 {user_turn} 触发了升级规则（{rule_id}）：“{user_text}”。"
            f"助手回合 {assistant_turn} 未按要求执行约定的升级动作（{action_text}）。"
        )

    # Fallback partial localization for mixed-language evidence.
    localized = re.sub(r"\bUser turn (\d+)\b", r"用户回合 \1", localized)
    localized = re.sub(r"\bAssistant turn (\d+)\b", r"助手回合 \1", localized)
    localized = localized.replace("did not execute the required escalation action (", "未执行要求的升级动作（")
    localized = localized.replace("did not execute the required action (", "未执行要求的动作（")
    localized = localized.replace("did not call the required tool", "未调用要求的工具")
    localized = localized.replace('").', "）。")

    return localized


def _render_review_markdown_legacy(review: dict[str, Any]) -> str:
    interactions_cache: dict[str, list[Interaction] | None] = {}
    unavailable_turn_message = "未能定位原始轮次：case turn 不为 assistant 或索引无效"
    lines = [
        f"# 批量 Badcase Review: {review['batch_id']}",
        "",
        f"- 扫描文件数: {review['file_count']}",
        f"- 候选 badcase 数: {review['badcase_count']}",
        "",
    ]
    for file_record in review.get("files") or []:
        source_path = str(file_record.get("path") or "")
        interactions = interactions_cache.get(source_path)
        if interactions is None and source_path:
            interactions = _conversation_interactions_from_file(source_path)
            interactions_cache[source_path] = interactions
        lines.extend(
            [
                f"## {file_record.get('file_name')}",
                "",
                f"- 文件路径: `{file_record.get('path')}`",
                f"- 扫描轮次: `{file_record.get('scan_event_id')}`",
                f"- 文件状态: `{file_record.get('status')}`",
                f"- badcase 数: {file_record.get('badcase_count', 0)}",
                "",
            ]
        )
        if file_record.get("parse_error"):
            lines.extend([f"解析错误: {file_record['parse_error']}", ""])
            continue
        if not file_record.get("badcases"):
            lines.extend(["未发现可疑 badcase。", ""])
            continue
        for index, case in enumerate(file_record.get("badcases") or [], start=1):
            source_turn, user_turn, user_text, assistant_text = _resolve_review_case_turn_context(
                case=case,
                interactions=interactions,
            )
            lines.extend(
                [
                    f"### {index}. `{case['id']}`",
                    f"- turn: `{case['turn_index']}`",
                    f"- error_type: `{case['error_type']}`",
                    f"- source: `{case.get('source')}`",
                    f"- scan round: `{file_record.get('scan_event_id')}`",
                    "",
                    "#### 原始对话",
                ]
            )
            if source_turn is None:
                lines.extend([f"- {unavailable_turn_message}", ""])
            else:
                if user_turn is None:
                    lines.extend(
                        [
                            "- User turn: 无法定位到有效 user turn（已按当前策略回退）",
                        ]
                    )
                else:
                    lines.extend([f"- User turn {user_turn}: {user_text}"])
                lines.extend([f"- Assistant turn {source_turn}: {assistant_text}", ""])

            lines.extend(
                [
                    "#### 分析",
                    f"- {case['error_type']}",
                    "",
                    "#### 证据",
                    f"- {_localize_review_evidence(str(case['evidence']))}",
                    "",
                    "#### 修改建议",
                    f"- {case['recommendation']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## 审批",
            "",
            "请先人工复核，再执行 `--approve-all`，或创建审批 JSON：",
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
            "验证结论：全部修复。所有 "
            f"{approved} 个已审批 badcase 在后置残差扫描中全部通过，"
            f"已验证修复 turn：{fixed_turns or 'none recorded'}。"
            "验收依据仅看残差：residual_badcase_count=0，verification_failure_count=0，unsupported_case_count=0。"
        )
    elif fixed > 0:
        verdict = (
            "验证结论：部分修复。"
            f"本轮共修复 {fixed} / {approved} 个已审批 badcase。"
            f"残差扫描仍有 {residual} 个剩余 badcase，"
            f"verification_failure_count={failures}，unsupported_case_count={unsupported}。"
            f"已修复 turn：{fixed_turns or 'none recorded'}，残差 turn：{residual_turns or 'none recorded'}。"
        )
    else:
        verdict = (
            "验证结论：未修复。尽管进行了变更，"
            f"本轮只应用 {applied} / {approved} 个 badcase，但残差扫描仍未清零（residual_badcase_count={residual}），"
            f"且 verification_failure_count={failures}，unsupported_case_count={unsupported}。"
        )
    if file_evidence:
        surface = "\n".join(f"- {evidence['surface_summary']}" for evidence in file_evidence)
        deep = "\n".join(f"- {evidence['deep_summary']}" for evidence in file_evidence)
    else:
        surface = "- 未记录分层表层证据。"
        deep = "- 未记录后端证据。"
    root_cause = _build_root_cause_analysis(conclusion, file_evidence)
    diagnosis = (
        f"Badcase 诊断与后端证据：{format_failure_category_counts_zh(conclusion.get('failure_category_counts') or {})}\n\n"
        f"根因分析：\n{root_cause}\n\n"
        f"表层证据：\n{surface}\n\n"
        f"后端证据：\n{deep}"
    )

    next_experiments = [
        next_experiment_for_failed_case(failure)
        for item in conclusion.get("files") or []
        for failure in item.get("verification_failures") or []
    ]
    if residual == 0 and failures == 0 and unsupported == 0:
        next_action = (
            "下一步动作：保留当前已验证版本，不再新增修改，直接复核是否进入下一批任务。"
            "当前验收标准已满足：residual_badcase_count=0、verification_failure_count=0、unsupported_case_count=0。"
        )
    elif next_experiments:
        next_action = (
            f"下一步动作：{next_experiments[0]}。"
            "请先定位触发规则并执行一次受限实验：仅修正命中的分支和目标助手回合，然后执行一次 residual scan。"
            "prompt/rerun 被执行不代表成功，仍以 residual scan 结果为准。"
        )
    elif unsupported:
        next_action = (
            "下一步动作：先补齐 unsupported case 的 ground truth 或精确触发文本，"
            "再重建 Repair Plan 后重新 apply。"
        )
    else:
        next_action = "下一步动作：复核 residual 证据，按失败分类执行一次边界实验。"

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
                "- 残差扫描未发现剩余 badcase。prompt 编辑与 rerun 仅是执行过程证据，"
                "残差扫描仍是主要正确性证据。"
            )
        return "- 仍有 residual badcase，未记录结构化 verification failure，请直接复核残差用例。"

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
                    "prompt edit 与定向 rerun 已执行，但模型仍将超期支付提案引导到协商/澄清分支，"
                    "而不是终端 RTP_Closing，说明是分支优先级或路由策略问题。"
                )
            elif error_type == "escalation_action_not_followed":
                cause = (
                    "模型在 escalation 触发后未把该动作当作终局处理，继续对话，说明 escalation 分支主导性不足。"
                )
            elif error_type == "busy_availability_check_skipped":
                cause = (
                    "模型跳过了可用性检测并过早进入支付流程，说明 busy/unavailable 分支优先级不够。"
                )
            else:
                cause = "rerun 仍触发残差规则，说明应用变更未真正控制该分支。"
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
        f"# 批量 Apply 结论: {conclusion['batch_id']}",
        "",
        "## 1. 验证结论",
        "",
        str(sections["verification_verdict"]),
        "",
        "## 2. Badcase 诊断与后端证据",
        "",
        str(sections["badcase_diagnosis_and_backend_evidence"]),
        "",
    ]
    for item in conclusion.get("files") or []:
        file_name = Path(str(item.get("source_file") or "unknown")).name
        request_ids = ", ".join(
            str(detail.get("request_id"))
            for detail in (item.get("rerun_results") or [])
            if isinstance(detail, dict) and detail.get("request_id")
        ) or "none"
        lines.extend(
            [
                (
                    f"- `{file_name}`: updated=`{item.get('updated_file')}`, "
                    f"rounds=`{item.get('scan_event_id')}` / `{item.get('apply_event_id')}` / `{item.get('residual_scan_event_id')}`, "
                    f"prompt_hash={item.get('before_hash') or '-'}->{item.get('after_hash') or '-'} , "
                    f"rerun_turns={item.get('rerun_target_turns') or []}, request_ids=`{request_ids}`"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 3. 下一步动作",
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
        "fixable_by_prompt_clarification": "规则或优先级边界不够清晰，适合用受限 prompt 澄清修复。",
        "unsupported_by_missing_ground_truth": "缺少 exact message、工具结果、日期、金额、热线或业务决策等必要信息，batch 不能自行编造。",
        "backend_failed_to_patch": "apply 后端未能产出符合 JSON/patch contract 的可用 prompt 改动。",
        "backend_failed_to_autonomously_repair": "所有适用的自动修复策略都失败，需要保留 residual badcase 供人工复核。",
        "patch_applied_but_failed_verification": "prompt edit 或 targeted rerun 已执行，但 residual scan 仍发现同类违规。",
        "likely_model_or_context_limited": "仅在多轮受控实验后，且 prompt/metadata 已足够明确时使用。",
    }
    return descriptions.get(str(category or "unknown"), "请查看记录的 residual evidence 和 failure category。")


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
            "验证结论：全部修复。"
            f"所有 {approved} 个已审批 badcase 都通过了后置 residual scan，"
            f"已验证修复的 turn：{fixed_turns or '未记录'}。"
            "验收依据以 residual scan 为准：residual_badcase_count=0，"
            "verification_failure_count=0，unsupported_case_count=0。"
        )
    elif fixed > 0:
        verdict = (
            "验证结论：部分修复。"
            f"本轮修复 {fixed} / {approved} 个已审批 badcase。"
            f"后置 residual scan 仍有 {residual} 个剩余 badcase，"
            f"verification_failure_count={failures}，unsupported_case_count={unsupported}。"
            f"已修复 turn：{fixed_turns or '未记录'}；剩余 turn：{residual_turns or '未记录'}。"
        )
    else:
        verdict = (
            "验证结论：未修复。"
            f"本轮只应用了 {applied} / {approved} 个 badcase，"
            f"且后置 residual scan 未清零，residual_badcase_count={residual}，"
            f"verification_failure_count={failures}，unsupported_case_count={unsupported}。"
        )

    if file_evidence:
        surface = "\n".join(f"- {evidence['surface_summary']}" for evidence in file_evidence)
        deep = "\n".join(f"- {evidence['deep_summary']}" for evidence in file_evidence)
    else:
        surface = "- 未记录分层表层证据。"
        deep = "- 未记录后端证据。"

    root_cause = _build_root_cause_analysis(conclusion, file_evidence)
    diagnosis = (
        f"Badcase 诊断与后端证据：失败分类统计：{format_failure_category_counts_zh(conclusion.get('failure_category_counts') or {})}\n\n"
        f"根因分析：\n{root_cause}\n\n"
        f"表层证据：\n{surface}\n\n"
        f"后端证据：\n{deep}"
    )

    next_experiments = [
        next_experiment_for_failed_case(failure)
        for item in conclusion.get("files") or []
        for failure in item.get("verification_failures") or []
    ]
    if residual == 0 and failures == 0 and unsupported == 0:
        next_action = (
            "下一步动作：保留当前已验证版本，不再新增修改。"
            "当前验收标准已满足：residual_badcase_count=0，"
            "verification_failure_count=0，unsupported_case_count=0。"
        )
    elif next_experiments:
        next_action = (
            f"下一步动作：{next_experiments[0]}。"
            "请只针对剩余 badcase 定位触发规则并执行一次受限实验："
            "只修正命中的分支和目标助手回合，然后执行一次 residual scan。"
            "prompt edit 或 rerun 被执行不代表成功，仍以 residual scan 结果为准。"
        )
    elif unsupported:
        next_action = (
            "下一步动作：先补齐 unsupported case 所需的 ground truth 或精确触发文本，"
            "再重建 Repair Plan 后重新 apply。"
        )
    else:
        next_action = "下一步动作：复核 residual 证据，并按失败分类执行一次边界实验。"

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
                "- residual scan 未发现剩余 badcase。prompt edit 与 rerun 只是执行过程证据，"
                "真正的正确性依据仍是后置 residual scan 已清零。"
            )
        return "- 仍有 residual badcase，但未记录结构化 verification failure；请直接复核 residual 用例。"

    lines = []
    evidence_by_source = {evidence.get("source_file"): evidence for evidence in file_evidence}
    for item in conclusion.get("files") or []:
        evidence = evidence_by_source.get(item.get("source_file")) or {}
        logprob_read = str(evidence.get("logprob_interpretation") or "未记录 logprob 解释")
        for failure in item.get("verification_failures") or []:
            error_type = str(failure.get("error_type") or "unknown")
            turn = failure.get("turn_index") or failure.get("residual_turn_index")
            case_id = failure.get("case_id") or "unknown"
            residual_id = failure.get("residual_case_id") or "unknown"
            if error_type == "late_payment_proposal_not_rtp_closing":
                cause = (
                    "prompt edit 与 targeted rerun 已执行，但模型仍把逾期支付提案路由到协商或澄清分支，"
                    "说明 RTP_Closing 的分支优先级或路由策略仍不够强。"
                )
            elif error_type == "escalation_action_not_followed":
                cause = (
                    "模型在 escalation 触发后没有把升级动作当作终局处理，仍继续对话，"
                    "说明 escalation 分支的主导性不足。"
                )
            elif error_type == "busy_availability_check_skipped":
                cause = (
                    "模型跳过可用性检查并过早进入支付流程，说明 busy/unavailable 分支优先级不足。"
                )
            else:
                cause = "rerun 仍触发 residual 规则，说明当前改动还没有真正控制该分支。"
            lines.append(
                f"- `{case_id}` -> `{residual_id}` turn {turn}: {cause} Logprob 解读：{logprob_read}。"
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
                    {"token": token.get("token"), "logprob": token.get("logprob")}
                    for token in (logprobs.get("low_confidence_tokens") or [])[:5]
                    if isinstance(token, dict)
                ],
            }
        )

    logprob_summary, logprob_interpretation = _summarize_logprobs(rerun_details, residual_cases)
    residual_summary = "; ".join(
        f"turn {case.get('turn_index')} {case.get('error_type')}"
        for case in residual_cases
    ) or "无"
    patch_summary = " | ".join(
        f"{edit.get('case_id')}: {_truncate_conclusion_text(str(edit.get('applied_feedback_summary') or ''), 140)}"
        for edit in prompt_edits
    ) or "无"
    request_ids = [str(detail.get("request_id")) for detail in rerun_details if detail.get("request_id")]
    failed_ids = [
        f"{failure.get('case_id')}->{failure.get('residual_case_id')}"
        for failure in item.get("verification_failures") or []
    ]
    file_name = Path(str(item.get("source_file") or "unknown")).name
    surface_summary = (
        f"{file_name}：已应用 {item.get('applied_case_count', 0)} 个，"
        f"已验证修复 {item.get('fixed_case_count', 0)} 个，"
        f"剩余 {item.get('residual_badcase_count', 0)} 个，"
        f"unsupported {len(item.get('unsupported_cases') or [])} 个；"
        f"residual={residual_summary}；verification_failures={', '.join(failed_ids) or '无'}。"
    )
    deep_summary = (
        f"{file_name}：backend={item.get('apply_provider') or 'unknown'}/{item.get('apply_model') or 'unknown'}，"
        f"prompt_hash={item.get('before_hash') or '-'}->{item.get('after_hash') or '-'}，"
        f"prompt_changed={bool(item.get('prompt_changed'))}，rerun_targets={item.get('rerun_target_turns') or []}，"
        f"request_ids={', '.join(request_ids) or '无'}，prompt_edits={patch_summary}，{logprob_summary}"
    )
    diagnosis = f"{surface_summary} {deep_summary}"
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
            "logprobs=不可用；不能把结果归因于 token 置信度。",
            "未记录可用的 logprob 证据",
        )
    residual_turns = {case.get("turn_index") for case in residual_cases}
    parts = []
    interpretations = []
    for detail in logprob_available:
        avg = detail.get("avg_logprob")
        minimum = detail.get("min_logprob")
        turn = detail.get("turn")
        if turn in residual_turns and isinstance(avg, (int, float)) and avg >= -0.2:
            interpretation = "稳定偏向错误分支"
        elif isinstance(minimum, (int, float)) and minimum <= -2.0:
            interpretation = "局部 token 不确定"
        else:
            interpretation = "仅作置信度参考"
        interpretations.append(f"turn {turn}：{interpretation}")
        tokens = ", ".join(
            str(token.get("token")).strip()
            for token in detail.get("low_confidence_tokens") or []
            if token.get("token")
        ) or "无"
        parts.append(
            f"turn {turn} avg={_format_logprob_value(avg)}, min={_format_logprob_value(minimum)}, "
            f"request={detail.get('request_id') or '无'}, 解读={interpretation}, low_tokens={tokens}"
        )
    return (
        "logprobs：" + "；".join(parts)
        + "。logprobs 只能作为后端置信度或路由诊断证据，正确性仍以 residual scan 为准。",
        "；".join(interpretations),
    )


def render_apply_conclusion_markdown(conclusion: dict[str, Any]) -> str:
    sections = conclusion.get("conclusion_sections") or build_batch_conclusion_sections(conclusion)
    lines = [
        f"# 批量 Apply 结论：{conclusion['batch_id']}",
        "",
        "## 1. 验证结论",
        "",
        str(sections["verification_verdict"]),
        "",
        "## 2. Badcase 诊断与后端证据",
        "",
        str(sections["badcase_diagnosis_and_backend_evidence"]),
        "",
    ]
    for item in conclusion.get("files") or []:
        file_name = Path(str(item.get("source_file") or "unknown")).name
        request_ids = []
        for detail in item.get("rerun_results") or []:
            if not isinstance(detail, dict):
                continue
            diagnostics = detail.get("response_diagnostics")
            if isinstance(diagnostics, dict) and diagnostics.get("request_id"):
                request_ids.append(str(diagnostics.get("request_id")))
            elif detail.get("request_id"):
                request_ids.append(str(detail.get("request_id")))
        lines.append(
            f"- `{file_name}`：updated=`{item.get('updated_file')}`，"
            f"rounds=`{item.get('scan_event_id')}` / `{item.get('apply_event_id')}` / `{item.get('residual_scan_event_id')}`，"
            f"prompt_hash={item.get('before_hash') or '-'}->{item.get('after_hash') or '-'}，"
            f"rerun_turns={item.get('rerun_target_turns') or []}，request_ids=`{', '.join(request_ids) or '无'}`"
        )
    lines.extend(
        [
            "",
            "## 3. 下一步动作",
            "",
            str(sections["next_action"]),
            "",
        ]
    )
    return "\n".join(lines)


def _review_case_judgement(case: dict[str, Any]) -> str:
    error_type = str(case.get("error_type") or "unknown")
    if error_type == "escalation_action_not_followed":
        return "模型判断：这是升级分支未按要求终止对话的 badcase。用户触发升级后，助手应该立即执行固定升级动作，而不是继续解释、澄清或协商。"
    if error_type == "missing_required_tool_call":
        return "模型判断：这是要求工具未被调用的 badcase。当前回复绕过了 system prompt 中的工具约束，导致流程没有进入应有的工具动作。"
    if error_type == "late_payment_proposal_not_rtp_closing":
        return "模型判断：这是付款承诺分支路由错误的 badcase。用户已经给出应进入终局的付款信息，但助手仍把对话带回协商或澄清。"
    if error_type == "busy_availability_check_skipped":
        return "模型判断：这是可用性检查被跳过的 badcase。用户表达忙碌或不可用后，助手不应过早进入付款流程。"
    return f"模型判断：这是 `{error_type}` 类型的 prompt-compliance 候选 badcase，需要人工确认其是否违反了明确的 system prompt 规则。"


def _review_case_reason(case: dict[str, Any]) -> str:
    recommendation = str(case.get("recommendation") or "").strip()
    if recommendation:
        return f"结论思考：问题核心不是回答质量，而是对话没有服从指定分支。建议修复方向是：{recommendation}"
    return "结论思考：问题核心不是回答是否自然，而是当前助手回合与 system prompt 的流程、分支或动作要求不一致。"


def _compact_evidence_for_report(evidence: str, limit: int = 360) -> str:
    localized = _localize_review_evidence(str(evidence or ""))
    normalized = re.sub(r"\s+", " ", localized).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _render_review_markdown_legacy_v2(review: dict[str, Any]) -> str:
    interactions_cache: dict[str, list[Interaction] | None] = {}
    lines = [
        f"# 批量 Badcase Review：{review['batch_id']}",
        "",
        "## 总体判断",
        "",
        f"本轮扫描了 {review['file_count']} 个文件，发现 {review['badcase_count']} 个候选 badcase。下面每个候选先给模型判断和处理建议，证据只保留为必要核对信息。",
        "",
    ]
    for file_record in review.get("files") or []:
        source_path = str(file_record.get("path") or "")
        interactions = interactions_cache.get(source_path)
        if interactions is None and source_path:
            interactions = _conversation_interactions_from_file(source_path)
            interactions_cache[source_path] = interactions
        lines.extend(
            [
                f"## {file_record.get('file_name')}",
                "",
                f"- 文件路径：`{file_record.get('path')}`",
                f"- 扫描轮次：`{file_record.get('scan_event_id')}`",
                f"- 文件状态：`{file_record.get('status')}`",
                f"- 候选 badcase 数：{file_record.get('badcase_count', 0)}",
                "",
            ]
        )
        if file_record.get("parse_error"):
            lines.extend([f"解析错误：{file_record['parse_error']}", ""])
            continue
        badcases = file_record.get("badcases") or []
        if not badcases:
            lines.extend(["模型判断：未发现可疑 prompt-compliance badcase。", ""])
            continue
        for index, case in enumerate(badcases, start=1):
            source_turn, user_turn, user_text, assistant_text = _resolve_review_case_turn_context(
                case=case,
                interactions=interactions,
            )
            lines.extend(
                [
                    f"### {index}. `{case['id']}`",
                    "",
                    _review_case_judgement(case),
                    "",
                    _review_case_reason(case),
                    "",
                    "#### 建议动作",
                    f"- 人工确认后再 apply；如果确认该判断，优先按 recommendation 做最小 prompt edit 或 targeted rerun。",
                    f"- case 元信息：turn `{case.get('turn_index')}`，error_type `{case.get('error_type')}`，source `{case.get('source')}`。",
                    "",
                    "#### 必要证据摘要",
                    f"- {_compact_evidence_for_report(str(case.get('evidence') or ''))}",
                    "",
                    "#### 对话片段",
                ]
            )
            if source_turn is None:
                lines.extend(["- 未能定位原始轮次：case turn 不为 assistant 或索引无效。", ""])
            else:
                if user_turn is None:
                    lines.append("- User turn：未能定位到有效 user turn，已按当前策略回退。")
                else:
                    lines.append(f"- User turn {user_turn}: {user_text}")
                lines.extend([f"- Assistant turn {source_turn}: {assistant_text}", ""])

    lines.extend(
        [
            "## 审批",
            "",
            "请先人工复核上面的模型判断，再执行 `--approve-all`，或创建审批 JSON：",
            "",
            "```json",
            json.dumps({"approved_case_ids": []}, indent=2),
            "```",
            "",
        ]
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
                "old_response": _truncate_conclusion_text(str(result.get("old_assistant_response") or ""), 260),
                "new_response": _truncate_conclusion_text(str(result.get("new_assistant_response") or ""), 320),
                "error": result.get("error"),
                "request_id": diagnostics.get("request_id"),
                "provider": diagnostics.get("provider"),
                "model": diagnostics.get("model"),
                "logprobs_available": logprobs.get("available") is True,
                "avg_logprob": logprobs.get("avg_logprob"),
                "min_logprob": logprobs.get("min_logprob"),
                "low_confidence_tokens": [
                    {"token": token.get("token"), "logprob": token.get("logprob")}
                    for token in (logprobs.get("low_confidence_tokens") or [])[:5]
                    if isinstance(token, dict)
                ],
            }
        )
    logprob_summary, logprob_interpretation = _summarize_logprobs(rerun_details, residual_cases)
    residual_summary = "; ".join(
        f"turn {case.get('turn_index')} {case.get('error_type')}"
        for case in residual_cases
    ) or "无"
    patch_summary = " | ".join(
        f"{edit.get('case_id')}: {_truncate_conclusion_text(str(edit.get('applied_feedback_summary') or ''), 120)}"
        for edit in prompt_edits
    ) or "无"
    request_ids = [str(detail.get("request_id")) for detail in rerun_details if detail.get("request_id")]
    failed_ids = [
        f"{failure.get('case_id')}->{failure.get('residual_case_id')}"
        for failure in item.get("verification_failures") or []
    ]
    changed_turns = [
        f"turn {detail.get('turn')}: `{detail.get('old_response')}` -> `{detail.get('new_response')}`"
        for detail in rerun_details[:3]
        if detail.get("turn") is not None
    ]
    file_name = Path(str(item.get("source_file") or "unknown")).name
    outcome_summary = (
        f"{file_name}：应用 {item.get('applied_case_count', 0)} 个，"
        f"验证修复 {item.get('fixed_case_count', 0)} 个，剩余 {item.get('residual_badcase_count', 0)} 个，"
        f"unsupported {len(item.get('unsupported_cases') or [])} 个。"
    )
    reasoning_summary = (
        f"{file_name}：修复判断来自 residual scan 与目标回合重跑对比。"
        f"可见变化：{'; '.join(changed_turns) if changed_turns else '无目标回合重跑变化'}。"
    )
    audit_summary = (
        f"{file_name}：rounds={item.get('scan_event_id')} / {item.get('apply_event_id')} / {item.get('residual_scan_event_id')}；"
        f"backend={item.get('apply_provider') or 'unknown'}/{item.get('apply_model') or 'unknown'}；"
        f"prompt_hash={item.get('before_hash') or '-'}->{item.get('after_hash') or '-'}；"
        f"request_ids={', '.join(request_ids) or '无'}；prompt_edits={patch_summary}；"
        f"residual={residual_summary}；verification_failures={', '.join(failed_ids) or '无'}；{logprob_summary}"
    )
    return {
        "source_file": item.get("source_file"),
        "diagnosis": f"{outcome_summary} {reasoning_summary}",
        "surface_summary": outcome_summary,
        "reasoning_summary": reasoning_summary,
        "deep_summary": audit_summary,
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
            "logprobs=不可用；不能把结论归因于 token 置信度。",
            "未记录可用的 logprob 证据",
        )
    residual_turns = {case.get("turn_index") for case in residual_cases}
    parts = []
    interpretations = []
    for detail in logprob_available:
        avg = detail.get("avg_logprob")
        minimum = detail.get("min_logprob")
        turn = detail.get("turn")
        if turn in residual_turns and isinstance(avg, (int, float)) and avg >= -0.2:
            interpretation = "模型稳定偏向错误分支"
        elif isinstance(minimum, (int, float)) and minimum <= -2.0:
            interpretation = "局部 token 不确定"
        else:
            interpretation = "仅作置信度参考"
        interpretations.append(f"turn {turn}：{interpretation}")
        tokens = ", ".join(
            str(token.get("token")).strip()
            for token in detail.get("low_confidence_tokens") or []
            if token.get("token")
        ) or "无"
        parts.append(
            f"turn {turn} avg={_format_logprob_value(avg)}, min={_format_logprob_value(minimum)}, "
            f"request={detail.get('request_id') or '无'}, 解读={interpretation}, low_tokens={tokens}"
        )
    return (
        "logprobs：" + "；".join(parts) + "。logprobs 只用于解释模型置信度，正确性仍以 residual scan 为准。",
        "；".join(interpretations),
    )


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
            "验证结论：全部修复。"
            f"本轮 {approved} 个已审批 badcase 均通过后置 residual scan；"
            f"已验证修复 turn：{fixed_turns or '未记录'}。"
        )
    elif fixed > 0:
        verdict = (
            "验证结论：部分修复。"
            f"本轮修复 {fixed} / {approved} 个已审批 badcase；"
            f"剩余 {residual} 个 badcase，失败 {failures} 个，unsupported {unsupported} 个。"
            f"已修复 turn：{fixed_turns or '未记录'}；剩余 turn：{residual_turns or '未记录'}。"
        )
    else:
        verdict = (
            "验证结论：未修复。"
            f"本轮应用 {applied} / {approved} 个 badcase，但 residual scan 仍未清零；"
            f"residual_badcase_count={residual}，verification_failure_count={failures}，unsupported_case_count={unsupported}。"
        )

    reasoning = "\n".join(f"- {evidence['reasoning_summary']}" for evidence in file_evidence) or "- 未记录模型修复思考。"
    audit = "\n".join(f"- {evidence['deep_summary']}" for evidence in file_evidence) or "- 未记录后端审计证据。"
    root_cause = _build_root_cause_analysis(conclusion, file_evidence)
    diagnosis = (
        "模型结论思考：\n"
        f"{root_cause}\n\n"
        "修复判断：\n"
        f"{reasoning}\n\n"
        "必要审计证据：\n"
        f"{audit}"
    )

    next_experiments = [
        next_experiment_for_failed_case(failure)
        for item in conclusion.get("files") or []
        for failure in item.get("verification_failures") or []
    ]
    if residual == 0 and failures == 0 and unsupported == 0:
        next_action = "下一步动作：保留当前已验证版本，不再新增修改；可以进入下一批任务。"
    elif next_experiments:
        next_action = (
            f"下一步动作：{next_experiments[0]}。只针对剩余 badcase 做一次受限修复实验，"
            "完成后以一次 residual scan 作为验收标准。"
        )
    elif unsupported:
        next_action = "下一步动作：先补齐 unsupported case 的 ground truth 或精确触发文本，再重新 apply。"
    else:
        next_action = "下一步动作：复核 residual 证据，按失败分类执行一次边界实验。"

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
                "- 模型这次之所以判定已修复，是因为目标回合已经改成符合规则的响应，"
                "并且后置 residual scan 没有发现同类违规；prompt edit 和 rerun 只是过程证据，不单独代表成功。"
            )
        return "- 仍有 residual badcase，但没有结构化 verification failure；需要直接围绕 residual case 重新判断触发分支。"

    lines = []
    evidence_by_source = {evidence.get("source_file"): evidence for evidence in file_evidence}
    for item in conclusion.get("files") or []:
        evidence = evidence_by_source.get(item.get("source_file")) or {}
        logprob_read = str(evidence.get("logprob_interpretation") or "未记录 logprob 解释")
        for failure in item.get("verification_failures") or []:
            error_type = str(failure.get("error_type") or "unknown")
            turn = failure.get("turn_index") or failure.get("residual_turn_index")
            case_id = failure.get("case_id") or "unknown"
            residual_id = failure.get("residual_case_id") or "unknown"
            if error_type == "late_payment_proposal_not_rtp_closing":
                cause = "模型仍把付款承诺路由到协商/澄清分支，说明终局分支优先级还不够强。"
            elif error_type == "escalation_action_not_followed":
                cause = "模型仍没有把 escalation 当作终局动作，说明升级分支的优先级或 exact-message 约束仍不足。"
            elif error_type == "busy_availability_check_skipped":
                cause = "模型仍跳过可用性检查，说明 busy/unavailable 分支优先级仍不足。"
            else:
                cause = "rerun 仍触发 residual 规则，说明当前改动还没有真正控制该分支。"
            lines.append(f"- `{case_id}` -> `{residual_id}` turn {turn}：{cause} Logprob 解读：{logprob_read}。")
    return "\n".join(lines)


def render_apply_conclusion_markdown(conclusion: dict[str, Any]) -> str:
    sections = conclusion.get("conclusion_sections") or build_batch_conclusion_sections(conclusion)
    lines = [
        f"# 批量 Apply 结论：{conclusion['batch_id']}",
        "",
        "## 1. 验证结论",
        "",
        str(sections["verification_verdict"]),
        "",
        "## 2. 模型结论思考",
        "",
        str(sections["badcase_diagnosis_and_backend_evidence"]),
        "",
        "## 3. 下一步动作",
        "",
        str(sections["next_action"]),
        "",
        "## 附录：轮次索引",
    ]
    for item in conclusion.get("files") or []:
        file_name = Path(str(item.get("source_file") or "unknown")).name
        request_ids = []
        for detail in item.get("rerun_results") or []:
            if not isinstance(detail, dict):
                continue
            diagnostics = detail.get("response_diagnostics")
            if isinstance(diagnostics, dict) and diagnostics.get("request_id"):
                request_ids.append(str(diagnostics.get("request_id")))
            elif detail.get("request_id"):
                request_ids.append(str(detail.get("request_id")))
        lines.append(
            f"- `{file_name}`：updated=`{item.get('updated_file')}`，"
            f"rounds=`{item.get('scan_event_id')}` / `{item.get('apply_event_id')}` / `{item.get('residual_scan_event_id')}`，"
            f"request_ids=`{', '.join(request_ids) or '无'}`"
        )
    lines.append("")
    return "\n".join(lines)


def _translate_recommendation_for_report(recommendation: str, error_type: str = "") -> str:
    text = " ".join(str(recommendation or "").split())
    if not text:
        return ""
    lowered = text.lower()
    if (
        "revise the escalation branch" in lowered
        and "first assistant response after any escalation trigger" in lowered
    ):
        return (
            "强化升级分支：任何升级触发后，助手的第一句回复必须立即停止协商流程，"
            "执行配置好的升级动作；如果系统要求固定话术、转接或热线结尾，也必须完整执行。"
        )
    if "required tool" in lowered and ("missing" in lowered or "call" in lowered):
        return "补强工具调用规则：命中该分支时必须调用系统要求的工具，不能用普通文本回复替代工具动作。"
    if error_type == "escalation_action_not_followed":
        return "强化升级分支：升级触发后必须把升级动作作为终局处理，不能继续解释、澄清或协商。"
    if error_type == "late_payment_proposal_not_rtp_closing":
        return "强化付款承诺分支：命中终局条件时必须进入指定 closing 动作，不能回到协商或澄清。"
    return text


def _review_case_reason(case: dict[str, Any]) -> str:
    translated = _translate_recommendation_for_report(
        str(case.get("recommendation") or ""),
        str(case.get("error_type") or ""),
    )
    if translated:
        return f"结论思考：问题核心不是回答质量，而是对话没有服从指定分支。建议修复方向是：{translated}"
    return "结论思考：问题核心不是回答是否自然，而是当前助手回合与 system prompt 的流程、分支或动作要求不一致。"


def _localize_review_evidence(evidence: str) -> str:
    if not evidence:
        return evidence

    text = str(evidence)
    escalation_match = re.search(
        r'Violated rule:\s*"(?P<rule>.*?)"\s*'
        r'Evidence:\s*User turn (?P<user_turn>\d+) triggered escalation \((?P<trigger>[^)]+)\) with '
        r'"(?P<user_text>.*?)"\.?\s*Assistant turn (?P<assistant_turn>\d+) did not execute the required escalation action '
        r'\("(?P<assistant_text>.*?)"\)\.?',
        " ".join(text.split()),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if escalation_match:
        rule = escalation_match.group("rule")
        user_turn = escalation_match.group("user_turn")
        trigger = escalation_match.group("trigger")
        user_text = escalation_match.group("user_text")
        assistant_turn = escalation_match.group("assistant_turn")
        assistant_text = escalation_match.group("assistant_text")
        return (
            f"违反规则：\"{rule}\"。"
            f"证据：用户回合 {user_turn} 触发升级规则（{trigger}），触发文本为 \"{user_text}\"。"
            f"助手回合 {assistant_turn} 没有执行要求的升级动作，实际回复为 \"{assistant_text}\"。"
        )

    localized = text
    localized = localized.replace("Violated rule:", "违反规则：")
    localized = localized.replace("Evidence:", "证据：")
    localized = re.sub(r"\bUser turn (\d+) triggered escalation \(([^)]+)\) with\b", r"用户回合 \1 触发升级规则（\2），触发文本为", localized)
    localized = re.sub(r"\bAssistant turn (\d+) did not execute the required escalation action\b", r"助手回合 \1 没有执行要求的升级动作", localized)
    localized = re.sub(r"\bAssistant turn (\d+) did not execute the required action\b", r"助手回合 \1 没有执行要求的动作", localized)
    localized = re.sub(r"\bUser turn (\d+)\b", r"用户回合 \1", localized)
    localized = re.sub(r"\bAssistant turn (\d+)\b", r"助手回合 \1", localized)
    localized = localized.replace("did not call the required tool", "没有调用要求的工具")
    return localized


def _compact_evidence_for_report(evidence: str, limit: int = 420) -> str:
    localized = _localize_review_evidence(str(evidence or ""))
    normalized = re.sub(r"\s+", " ", localized).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _review_case_repair_actions(case: dict[str, Any]) -> list[str]:
    error_type = str(case.get("error_type") or "")
    translated = _translate_recommendation_for_report(str(case.get("recommendation") or ""), error_type)
    if error_type == "escalation_action_not_followed":
        return [
            "修复目标：把升级触发分支提升为终局优先级；一旦命中升级条件，助手第一句必须停止协商并执行固定升级动作。",
            f"Prompt edit 要点：{translated}",
            "禁止行为：不要再先道歉、解释账单状态、询问付款渠道、询问是否转人工，或继续任何协商/澄清流程。",
            "Targeted rerun 要点：重跑命中的 assistant turn，期望新回复直接输出系统要求的升级话术；如需要工具或热线结尾，也必须一起出现。",
            "验收标准：后置 residual scan 不再出现 `escalation_action_not_followed`；目标 turn 的新回复与系统要求的 exact escalation action 一致。",
        ]
    if error_type == "missing_required_tool_call":
        return [
            "修复目标：命中该分支时必须执行系统要求的工具动作，不能用文本回复绕过工具调用。",
            f"Prompt edit 要点：{translated}",
            "Targeted rerun 要点：重跑目标 assistant turn，确认输出包含必要 tool call；本地 rerun 可接受 `not_executed` placeholder tool turn。",
            "验收标准：后置 residual scan 不再出现 missing-required-tool-call 类 badcase，且工具调用名称与参数满足 prompt 约束。",
        ]
    if error_type == "late_payment_proposal_not_rtp_closing":
        return [
            "修复目标：当用户付款承诺已满足终局条件时，直接进入 RTP_Closing 或系统指定 closing 分支。",
            f"Prompt edit 要点：{translated}",
            "禁止行为：不要继续要求用户换日期、补充付款渠道，或回到协商流程。",
            "验收标准：目标 turn 重跑后进入指定 closing 响应，后置 residual scan 不再报告该分支违规。",
        ]
    if error_type == "busy_availability_check_skipped":
        return [
            "修复目标：用户表达忙碌、不可用或无法继续对话时，先执行可用性检查分支。",
            f"Prompt edit 要点：{translated}",
            "禁止行为：不要跳过 availability gate 直接进入付款、承诺或协商流程。",
            "验收标准：目标 turn 重跑后先处理可用性分支，后置 residual scan 不再报告该类违规。",
        ]
    fallback = translated or "按该 case 的 violated rule、trigger 和 assistant violation 做最小范围 prompt 澄清。"
    return [
        f"修复目标：消除 `{error_type or 'unknown'}` 对应的明确 prompt-compliance 违规。",
        f"Prompt edit 要点：{fallback}",
        "Targeted rerun 要点：只重跑命中的 assistant turn，避免改动无关对话。",
        "验收标准：后置 residual scan 不再报告同一 case 或同类 residual badcase。",
    ]


def render_review_markdown(review: dict[str, Any]) -> str:
    interactions_cache: dict[str, list[Interaction] | None] = {}
    lines = [
        f"# 批量 Badcase Review：{review['batch_id']}",
        "",
        "## 总体判断",
        "",
        f"本轮扫描了 {review['file_count']} 个文件，发现 {review['badcase_count']} 个候选 badcase。请按每条 case 的对话证据与修复建议人工复核后再继续。",
        "",
    ]
    for file_record in review.get("files") or []:
        source_path = str(file_record.get("path") or "")
        interactions = interactions_cache.get(source_path)
        if interactions is None and source_path:
            interactions = _conversation_interactions_from_file(source_path)
            interactions_cache[source_path] = interactions
        lines.extend(
            [
                f"## {file_record.get('file_name')}",
                "",
                f"- 文件路径：`{file_record.get('path')}`",
                f"- 扫描轮次：`{file_record.get('scan_event_id')}`",
                f"- 文件状态：`{file_record.get('status')}`",
                f"- 候选 badcase 数：{file_record.get('badcase_count', 0)}",
                "",
            ]
        )
        if file_record.get("parse_error"):
            lines.extend([f"解析错误：{file_record['parse_error']}", ""])
            continue
        badcases = file_record.get("badcases") or []
        if not badcases:
            lines.extend(["模型判断：未发现可疑 prompt-compliance badcase。", ""])
            continue
        for index, case in enumerate(badcases, start=1):
            source_turn, user_turn, user_text, assistant_text = _resolve_review_case_turn_context(
                case=case,
                interactions=interactions,
            )
            lines.extend(
                [
                    f"### {index}. `{case['id']}`",
                    "",
                    "#### 对话片段",
                ]
            )
            if source_turn is None:
                lines.extend(["- 未能定位原始轮次：case turn 不为 assistant 或索引无效。", ""])
            else:
                if user_turn is None:
                    lines.append("- User turn：未能定位到有效 user turn，已按当前策略回退。")
                else:
                    lines.append(f"- User turn {user_turn}: {user_text}")
                lines.extend([f"- Assistant turn {source_turn}: {assistant_text}", ""])
            lines.extend(
                [
                    "#### 必要证据摘要",
                    f"- {_compact_evidence_for_report(str(case.get('evidence') or ''))}",
                    "",
                    "#### 建议动作",
                ]
            )
            lines.extend(f"- {action}" for action in _review_case_repair_actions(case))
            lines.extend(
                [
                    "",
                    "#### 原始信息",
                    f"- turn：`{case.get('turn_index')}`",
                    f"- error_type：`{case.get('error_type')}`",
                    f"- source：`{case.get('source')}`",
                    f"- scan round：`{file_record.get('scan_event_id')}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## 审批",
            "",
            "请先人工复核上面的对话证据与建议动作，再执行 `--approve-all`，或创建审批 JSON：",
            "",
            "```json",
            json.dumps({"approved_case_ids": []}, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _build_fix_comparison_lines(conclusion: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in conclusion.get("files") or []:
        file_name = Path(str(item.get("source_file") or "unknown")).name
        prompt_changed = "已变更" if item.get("prompt_changed") else "未变更"
        applied_count = int(item.get("applied_case_count") or 0)
        fixed_count = int(item.get("fixed_case_count") or 0)
        residual_count = int(item.get("residual_badcase_count") or 0)
        rerun_results = item.get("rerun_results") or []
        rerun_deltas: list[str] = []
        for detail in rerun_results:
            if not isinstance(detail, dict):
                continue
            turn = detail.get("assistant_turn_index")
            if turn is None:
                continue
            old_response = str(detail.get("old_assistant_response") or "")
            new_response = str(detail.get("new_assistant_response") or "")
            if old_response and new_response:
                rerun_deltas.append(
                    f"turn {turn}：`{_truncate_conclusion_text(old_response, 60)} -> {_truncate_conclusion_text(new_response, 60)}`"
                )
            else:
                rerun_deltas.append(f"turn {turn}")
        if not rerun_deltas:
            rerun_deltas_text = "未观测到可比对回合"
        else:
            rerun_deltas_text = "; ".join(rerun_deltas[:3])
            if len(rerun_deltas) > 3:
                rerun_deltas_text += "；..."
        lines.append(
            f"- `{file_name}`：已尝试修复 {applied_count} 个，已通过 {fixed_count} 个，"
            f"prompt {prompt_changed}，重跑对比：{rerun_deltas_text}，残留 {residual_count} 个。"
        )
    return "\n".join(lines) if lines else "- 未产出 per-file 修复对比。"


def _build_residual_risk_lines(conclusion: dict[str, Any]) -> str:
    residual = int(conclusion.get("residual_badcase_count") or 0)
    failures = int(conclusion.get("verification_failure_count") or 0)
    unsupported = int(conclusion.get("unsupported_case_count") or 0)
    if residual == 0 and failures == 0 and unsupported == 0:
        return "- 当前无 residual badcase，主要残留风险是新对话链路中的偶发漂移。"

    lines = [
        f"- 残留 badcase：{residual} 个",
        f"- 验证失败：{failures} 个",
        f"- unsupported：{unsupported} 个",
    ]
    residual_types = []
    for item in conclusion.get("files") or []:
        for failure in item.get("verification_failures") or []:
            error_type = str(failure.get("error_type") or "")
            if error_type and error_type not in residual_types:
                residual_types.append(error_type)
    if residual_types:
        lines.append(f"- 残留风险聚焦于：{', '.join(f'`{item}`' for item in residual_types)}")
    return "\n".join(lines)


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
            "验证结论：全部通过。"
            f"本轮共审批 {approved} 个坏例，residual scan 未再暴露错误。"
            f"已验证 turn：{fixed_turns or '无'}。"
        )
    elif fixed > 0:
        verdict = (
            "验证结论：部分通过。"
            f"本轮修复 {fixed} / {approved} 个坏例。"
            f"残留 {residual} 个 badcase，失败 {failures} 个，unsupported {unsupported} 个。"
            f"已验证 turn：{fixed_turns or '无'}；残留 turn：{residual_turns or '无'}。"
        )
    else:
        verdict = (
            "验证结论：未通过。"
            f"本轮应用 {applied} / {approved} 个坏例，但 residual scan 仍未关闭。"
            f"residual_badcase_count={residual}，verification_failure_count={failures}，unsupported_case_count={unsupported}。"
        )

    reasoning = "\n".join(f"- {evidence['reasoning_summary']}" for evidence in file_evidence) or "- 当前无可读 reasoning。"
    audit = "\n".join(f"- {evidence['deep_summary']}" for evidence in file_evidence) or "- 当前无可读 deep summary。"
    root_cause = _build_root_cause_analysis(conclusion, file_evidence)
    comparison = _build_fix_comparison_lines(conclusion)
    risk = _build_residual_risk_lines(conclusion)
    diagnosis = (
        "### 模型结论思考\n"
        f"{root_cause}\n\n"
        "### 修复前后对比\n"
        f"{comparison}\n\n"
        "### 剩余风险判断\n"
        f"{risk}\n\n"
        "### 证据数据（surface）\n"
        f"{reasoning}\n\n"
        "### 证据数据（deep）\n"
        f"{audit}"
    )

    next_experiments = [
        next_experiment_for_failed_case(failure)
        for item in conclusion.get("files") or []
        for failure in item.get("verification_failures") or []
    ]
    if residual == 0 and failures == 0 and unsupported == 0:
        next_action = "下一步动作：保持当前结果观察即可，避免重复修改同一分支。"
    elif next_experiments:
        next_action = (
            f"下一步动作：{next_experiments[0]}。"
            "建议只对 residual badcase 开展 bounded 修复并在同一链路再次 residual scan。"
        )
    elif unsupported:
        next_action = "下一步动作：补齐 unsupported 的 ground truth，再按该 case 的 Repair Plan 继续。"
    else:
        next_action = "下一步动作：继续 residual 证据链验证后再进入下一轮实验。"

    return {
        "verification_verdict": verdict,
        "badcase_diagnosis_and_backend_evidence": diagnosis,
        "next_action": next_action,
        "evidence": file_evidence,
    }


def render_apply_conclusion_markdown(conclusion: dict[str, Any]) -> str:
    sections = conclusion.get("conclusion_sections") or build_batch_conclusion_sections(conclusion)
    lines = [
        f"# 批量 Apply 结论：{conclusion['batch_id']}",
        "",
        "## 1. 验证结论",
        "",
        str(sections["verification_verdict"]),
        "",
        "## 2. 模型结论思考",
        "",
        str(sections["badcase_diagnosis_and_backend_evidence"]),
        "",
        "## 3. 下一步动作",
        "",
        str(sections["next_action"]),
        "",
        "## 附录：审计信息",
    ]
    for item in conclusion.get("files") or []:
        file_name = Path(str(item.get("source_file") or "unknown")).name
        request_ids = []
        for detail in item.get("rerun_results") or []:
            if not isinstance(detail, dict):
                continue
            diagnostics = detail.get("response_diagnostics")
            if isinstance(diagnostics, dict) and diagnostics.get("request_id"):
                request_ids.append(str(diagnostics.get("request_id")))
            elif detail.get("request_id"):
                request_ids.append(str(detail.get("request_id")))
        lines.append(
            f"- `{file_name}` updated={item.get('updated_file')} "
            f"rounds={item.get('scan_event_id')} / {item.get('apply_event_id')} / {item.get('residual_scan_event_id')} "
            f"request_ids={', '.join(request_ids) or '未知'}"
        )
    return "\n".join(lines)


def _review_case_reason(case: dict[str, Any]) -> str:
    translated = _translate_recommendation_for_report(
        str(case.get("recommendation") or ""),
        str(case.get("error_type") or ""),
    )
    parts = [
        "模型判定：当前对话分支存在指令对齐问题。",
        f"问题根因：{translated or '当前 evidence 指向分支优先级或 exact-message 未被正确执行。'}",
        "修复方向：先强制修复该触发分支的优先级与 exact action，再做一次目标回合重跑验证。",
    ]
    return "\n".join(f"- {part}" for part in parts)


def _review_case_repair_actions(case: dict[str, Any]) -> list[str]:
    error_type = str(case.get("error_type") or "")
    translated = _translate_recommendation_for_report(str(case.get("recommendation") or ""), error_type)
    if error_type == "escalation_action_not_followed":
        return [
            "修复目标：将触发 escalation 的首个 assistant 回合直接进入指定停止分支，并执行 exact escalation text + transfer/hotline。",
            f"Prompt edit 要点：{translated}",
            "禁止行为：不要继续谈判、不要再次向用户追问付款时间，不要省略 required message。",
            "Targeted rerun：对触发该 badcase 的 assistant turn 做一次回放验证，并复核输出必须以 transfer/hotline 结尾。",
            "残留判定标准：若 residual scan 仍出现 same error_type，说明分支路由未闭环，需要继续修复。",
        ]
    if error_type == "missing_required_tool_call":
        return [
            "修复目标：在分支触发点补齐必须执行的工具调用。",
            f"Prompt edit 要点：{translated}",
            "Targeted rerun：对触发该 badcase 的 turn 进行重跑，校验工具调用前置且被真实执行（出现 `not_executed` 视为失败）。",
            "残留判定标准：残留扫描中若仍缺失工具调用，需继续按工具约束修复。",
        ]
    if error_type == "late_payment_proposal_not_rtp_closing":
        return [
            "修复目标：强化超过期限分支直接转入 RTP_Closing，不再进入任何追款协商。",
            f"Prompt edit 要点：{translated}",
            "Targeted rerun：复查 trigger 之后的首个 assistant turn 是否直接给出 closing/结束语。",
            "残留判定标准：若 residual scan 仍出现同类追款行为，需继续收敛分支。"
        ]
    fallback = translated or "请按照该 case 的 evidence 与原始分支规则逐条修正。"
    return [
        f"修复目标：{fallback}",
        "Targeted rerun：对相关 turn 重放并核验首个输出是否符合 required action。",
        "残留判定标准：若 residual scan 仍不通过则继续 bounded 重试。",
    ]


def _review_case_reason(case: dict[str, Any]) -> str:
    translated = _translate_recommendation_for_report(
        str(case.get("recommendation") or ""),
        str(case.get("error_type") or ""),
    )
    return "\n".join(
        [
            "- 模型判定：存在分支执行偏差，当前行为未优先进入规则要求的路径。",
            f"- 根因：{translated or '当前证据显示未执行 required 的分支动作。'}",
            "- 修复方向：先压实分支优先级，再复测触发点的首个 assistant 回合。",
        ]
    )


def _review_case_repair_actions(case: dict[str, Any]) -> list[str]:
    error_type = str(case.get("error_type") or "")
    translated = _translate_recommendation_for_report(str(case.get("recommendation") or ""), error_type)
    base_prompt_point = translated or "按 evidence 指向的分支规则执行 required action。"
    if error_type == "escalation_action_not_followed":
        return [
            "修复目标：将 escalation 触发回合直接转到停止分支，执行 exact escalation text + transfer/hotline。",
            f"Prompt edit 要点：{base_prompt_point}",
            "禁止动作：禁止再次追问付款细节或继续协商。",
            "目标回合重跑：仅验证该 badcase 触发回合的 first assistant reply。",
            "残留标准：若 residual scan 仍出现 same error_type，继续 bounded 修复。",
        ]
    if error_type == "missing_required_tool_call":
        return [
            "修复目标：补齐必需工具调用，防止只输出文本。",
            f"Prompt edit 要点：{base_prompt_point}",
            "目标回合重跑：复跑触发工具缺失的 assistant 回合并确认工具调用。",
            "残留标准：若出现 `not_executed` 或缺失工具调用，继续修复。",
        ]
    if error_type == "late_payment_proposal_not_rtp_closing":
        return [
            "修复目标：逾期分支直接 RTP_Closing，不再协商后续。",
            f"Prompt edit 要点：{base_prompt_point}",
            "目标回合重跑：复测触发回合是否第一句给出 closing/结束语。",
            "残留标准：残留 scan 中仍出现追款协商语句，说明优先级仍未闭环。",
        ]
    return [
        f"修复目标：{base_prompt_point}",
        "目标回合重跑：复测 case 对应的 assistant 回合。",
        "残留标准：若 residual scan 未通过，继续 bounded 重试。",
    ]


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
                "old_response": _truncate_conclusion_text(str(result.get("old_assistant_response") or ""), 260),
                "new_response": _truncate_conclusion_text(str(result.get("new_assistant_response") or ""), 320),
                "error": result.get("error"),
                "request_id": diagnostics.get("request_id") or result.get("request_id"),
                "provider": diagnostics.get("provider"),
                "model": diagnostics.get("model"),
                "logprobs_available": logprobs.get("available") is True,
                "avg_logprob": logprobs.get("avg_logprob"),
                "min_logprob": logprobs.get("min_logprob"),
                "low_confidence_tokens": [
                    {"token": token.get("token"), "logprob": token.get("logprob")}
                    for token in (logprobs.get("low_confidence_tokens") or [])[:5]
                    if isinstance(token, dict)
                ],
            }
        )

    logprob_summary, logprob_interpretation = _summarize_logprobs(rerun_details, residual_cases)
    residual_summary = "; ".join(
        f"turn {case.get('turn_index')} {case.get('error_type')}"
        for case in residual_cases
    ) or "无"
    patch_summary = " | ".join(
        f"{edit.get('case_id')}: {_truncate_conclusion_text(str(edit.get('applied_feedback_summary') or ''), 120)}"
        for edit in prompt_edits
    ) or "无"
    request_ids = [str(detail.get("request_id")) for detail in rerun_details if detail.get("request_id")]
    failed_ids = [
        f"{failure.get('case_id')}->{failure.get('residual_case_id')}"
        for failure in item.get("verification_failures") or []
    ]
    changed_turns = [
        f"turn {detail.get('turn')}: `{detail.get('old_response')}` -> `{detail.get('new_response')}`"
        for detail in rerun_details[:3]
        if detail.get("turn") is not None
    ]
    file_name = Path(str(item.get("source_file") or "unknown")).name
    applied_count = int(item.get("applied_case_count") or 0)
    fixed_count = int(item.get("fixed_case_count") or 0)
    residual_count = int(item.get("residual_badcase_count") or 0)
    unsupported_count = len(item.get("unsupported_cases") or [])
    is_audit_only = applied_count == 0 and fixed_count == 0 and residual_count == 0 and not rerun_details and not prompt_edits

    outcome_summary = (
        f"{file_name}：应用 {applied_count} 个，验证通过 {fixed_count} 个，"
        f"剩余 {residual_count} 个，unsupported {unsupported_count} 个。"
    )
    if changed_turns:
        visible_change = "; ".join(changed_turns)
    elif is_audit_only:
        visible_change = "无 approved case，仅保留为批次审计背景。"
    else:
        visible_change = "未观察到目标回合重跑变化。"
    reasoning_summary = (
        f"{file_name}：修复判断来自目标回合重跑对比与后置 residual scan。"
        f"可见变化：{visible_change}"
    )
    deep_lines = [
        f"{file_name}：",
        f"  - rounds: {item.get('scan_event_id')} / {item.get('apply_event_id')} / {item.get('residual_scan_event_id')}",
        f"  - backend/model: {item.get('apply_provider') or 'unknown'}/{item.get('apply_model') or 'unknown'}",
        f"  - prompt hash: {item.get('before_hash') or '-'} -> {item.get('after_hash') or '-'}",
        f"  - request ids: {', '.join(request_ids) or '无'}",
        f"  - rerun turns: {', '.join(str(detail.get('turn')) for detail in rerun_details if detail.get('turn') is not None) or '无'}",
        f"  - prompt edits: {patch_summary}",
        f"  - residual: {residual_summary}; verification failures: {', '.join(failed_ids) or '无'}",
        f"  - logprobs: {logprob_summary}",
    ]
    return {
        "source_file": item.get("source_file"),
        "diagnosis": f"{outcome_summary} {reasoning_summary}",
        "surface_summary": outcome_summary,
        "reasoning_summary": reasoning_summary,
        "deep_summary": "\n".join(deep_lines),
        "rerun_details": rerun_details,
        "prompt_edit_summaries": prompt_edits,
        "residual_badcases": residual_cases,
        "logprob_summary": logprob_summary,
        "logprob_interpretation": logprob_interpretation,
        "is_audit_only": is_audit_only,
    }


def _summarize_logprobs(rerun_details: list[dict[str, Any]], residual_cases: list[dict[str, Any]]) -> tuple[str, str]:
    logprob_available = [detail for detail in rerun_details if detail["logprobs_available"]]
    if not logprob_available:
        return (
            "不可用；不能把结论归因于 token 置信度。",
            "未记录可用的 logprob 证据",
        )
    residual_turns = {case.get("turn_index") for case in residual_cases}
    parts = []
    interpretations = []
    for detail in logprob_available:
        avg = detail.get("avg_logprob")
        minimum = detail.get("min_logprob")
        turn = detail.get("turn")
        if turn in residual_turns and isinstance(avg, (int, float)) and avg >= -0.2:
            interpretation = "模型稳定偏向残留错误分支"
        elif isinstance(minimum, (int, float)) and minimum <= -2.0:
            interpretation = "局部 token 不确定"
        else:
            interpretation = "仅作置信度参考"
        interpretations.append(f"turn {turn}：{interpretation}")
        tokens = ", ".join(
            str(token.get("token")).strip()
            for token in detail.get("low_confidence_tokens") or []
            if token.get("token")
        ) or "无"
        parts.append(
            f"turn {turn} avg_logprob={_format_logprob_value(avg)}, min_logprob={_format_logprob_value(minimum)}, "
            f"request={detail.get('request_id') or '无'}, 解读={interpretation}, low_tokens={tokens}"
        )
    return (
        "；".join(parts) + "。logprobs 只解释后端置信度，正确性仍以 residual scan 为准。",
        "；".join(interpretations),
    )


def _build_root_cause_analysis(conclusion: dict[str, Any], file_evidence: list[dict[str, Any]]) -> str:
    failures = [
        failure
        for item in conclusion.get("files") or []
        for failure in item.get("verification_failures") or []
    ]
    if not failures:
        if int(conclusion.get("residual_badcase_count") or 0) == 0:
            return (
                "- 原 badcase 的根因是用户已经付款或争议欠款的表达触发 escalation 后，"
                "模型仍进入付款解释、补充确认或 QRIS 追问的协商分支，而不是立即执行 exact escalation terminal branch。"
                "本轮目标回合已改为指定热线/结束语，且 residual_badcase_count=0；"
                "prompt edit 与 rerun 是过程证据，最终通过依据仍是后置 residual scan。"
            )
        return "- 仍有 residual badcase，但没有结构化 verification failure；需要直接围绕 residual case 重新判断触发分支。"

    lines = []
    evidence_by_source = {evidence.get("source_file"): evidence for evidence in file_evidence}
    for item in conclusion.get("files") or []:
        evidence = evidence_by_source.get(item.get("source_file")) or {}
        logprob_read = str(evidence.get("logprob_interpretation") or "未记录 logprob 解释")
        for failure in item.get("verification_failures") or []:
            error_type = str(failure.get("error_type") or "unknown")
            turn = failure.get("turn_index") or failure.get("residual_turn_index")
            case_id = failure.get("case_id") or "unknown"
            residual_id = failure.get("residual_case_id") or "unknown"
            if error_type == "late_payment_proposal_not_rtp_closing":
                cause = "模型仍把付款承诺路由到协商或澄清分支，说明终局分支优先级还不够强。"
            elif error_type == "escalation_action_not_followed":
                cause = "模型仍没有把 escalation 当作终局动作，说明升级分支优先级或 exact-message 约束仍不足。"
            elif error_type == "busy_availability_check_skipped":
                cause = "模型仍跳过可用性检查，说明 busy/unavailable 分支优先级仍不足。"
            else:
                cause = "rerun 仍触发 residual 规则，说明当前改动还没有真正控制该分支。"
            lines.append(f"- `{case_id}` -> `{residual_id}` turn {turn}：{cause} Logprob 解读：{logprob_read}。")
    return "\n".join(lines)


def _build_fix_comparison_lines(conclusion: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in conclusion.get("files") or []:
        evidence = _batch_file_conclusion_evidence(item)
        if evidence.get("is_audit_only"):
            continue
        file_name = Path(str(item.get("source_file") or "unknown")).name
        prompt_changed = "已变更" if item.get("prompt_changed") else "未变更"
        applied_count = int(item.get("applied_case_count") or 0)
        fixed_count = int(item.get("fixed_case_count") or 0)
        residual_count = int(item.get("residual_badcase_count") or 0)
        rerun_deltas = []
        for detail in item.get("rerun_results") or []:
            if not isinstance(detail, dict):
                continue
            turn = detail.get("assistant_turn_index")
            old_response = _truncate_conclusion_text(str(detail.get("old_assistant_response") or ""), 80)
            new_response = _truncate_conclusion_text(str(detail.get("new_assistant_response") or ""), 80)
            if turn is not None and old_response and new_response:
                rerun_deltas.append(f"turn {turn}: `{old_response}` -> `{new_response}`")
        rerun_deltas_text = "; ".join(rerun_deltas[:3]) or "未观察到可比对回合"
        if len(rerun_deltas) > 3:
            rerun_deltas_text += "；..."
        lines.append(
            f"- `{file_name}`：应用 {applied_count} 个，验证通过 {fixed_count} 个，"
            f"prompt {prompt_changed}，重跑对比：{rerun_deltas_text}，残留 {residual_count} 个。"
        )
    return "\n".join(lines) if lines else "- 本轮没有需要展示的已修复文件；0 approved case 文件仅放入审计附录。"


def _build_residual_risk_lines(conclusion: dict[str, Any]) -> str:
    residual = int(conclusion.get("residual_badcase_count") or 0)
    failures = int(conclusion.get("verification_failure_count") or 0)
    unsupported = int(conclusion.get("unsupported_case_count") or 0)
    if residual == 0 and failures == 0 and unsupported == 0:
        return "- 当前无 residual badcase；主要残留风险是未来新样本再次触发同类 paid-claim escalation 时发生分支漂移。"

    lines = [
        f"- 残留 badcase：{residual} 个",
        f"- 验证失败：{failures} 个",
        f"- unsupported：{unsupported} 个",
    ]
    residual_types = []
    for item in conclusion.get("files") or []:
        for failure in item.get("verification_failures") or []:
            error_type = str(failure.get("error_type") or "")
            if error_type and error_type not in residual_types:
                residual_types.append(error_type)
    if residual_types:
        lines.append(f"- 残留风险聚焦于：{', '.join(f'`{item}`' for item in residual_types)}")
    return "\n".join(lines)


def _first_rerun_delta(item: dict[str, Any]) -> tuple[int | None, str, str]:
    for detail in item.get("rerun_results") or []:
        if not isinstance(detail, dict):
            continue
        turn = detail.get("assistant_turn_index")
        old_response = _truncate_conclusion_text(str(detail.get("old_assistant_response") or ""), 180)
        new_response = _truncate_conclusion_text(str(detail.get("new_assistant_response") or ""), 220)
        if old_response or new_response:
            return turn, old_response, new_response
    return None, "", ""


def _error_type_natural_cause(error_type: str) -> str:
    if error_type == "escalation_action_not_followed":
        return "升级触发后没有完整进入终局升级分支，尤其是固定话术或 required action 没有同时落地"
    if error_type == "missing_required_tool_call":
        return "触发分支要求调用工具，但模型绕过了工具动作，只完成了文本回复"
    if error_type == "late_payment_proposal_not_rtp_closing":
        return "用户已经给出应进入终局的付款信息，但模型仍回到协商或澄清路径"
    if error_type == "busy_availability_check_skipped":
        return "用户表达忙碌或不可用后，模型跳过了可用性检查，过早进入后续流程"
    return "模型输出没有服从 system prompt 中对应的分支、工具或 exact-message 约束"


def _logprob_natural_read(evidence: dict[str, Any], has_residual: bool) -> str:
    summary = str(evidence.get("logprob_summary") or "")
    interpretation = str(evidence.get("logprob_interpretation") or "")
    if "不可用" in summary or "未记录" in interpretation:
        return "本次没有可用 logprob，因此不能从 token 置信度推导模型偏好。"
    if has_residual and ("稳定偏向" in interpretation or "avg_logprob=0" in summary):
        return (
            "logprob 显示模型对当前残留输出非常有把握，说明问题更像是指令优先级或固定动作约束不足，"
            "不是一次随机生成抖动；但正确性仍以 residual scan 为准。"
        )
    if "局部 token 不确定" in interpretation:
        return (
            "logprob 显示生成过程中存在局部不确定 token，可以作为排查线索；"
            "不过它只能解释置信度，不能替代 residual scan 判断。"
        )
    return "logprob 只作为后端置信度参考；当前修复是否成立仍由目标回合变化和 residual scan 决定。"


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return "无"


def _first_rerun_detail(evidence: dict[str, Any]) -> dict[str, Any]:
    for detail in evidence.get("rerun_details") or []:
        if isinstance(detail, dict):
            return detail
    return {}


def _low_token_text(detail: dict[str, Any]) -> str:
    tokens = [
        str(token.get("token")).strip()
        for token in detail.get("low_confidence_tokens") or []
        if isinstance(token, dict) and token.get("token")
    ]
    return "、".join(tokens[:5]) or "无"


def _fixed_logprob_reference(file_evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    for evidence in file_evidence:
        if evidence.get("is_audit_only") or evidence.get("residual_badcases"):
            continue
        detail = _first_rerun_detail(evidence)
        if detail.get("logprobs_available"):
            return {
                "file_name": Path(str(evidence.get("source_file") or "unknown")).name,
                "avg": detail.get("avg_logprob"),
                "min": detail.get("min_logprob"),
                "tokens": _low_token_text(detail),
                "new_response": detail.get("new_response"),
            }
    return None


def _logprob_experiment_read(
    evidence: dict[str, Any],
    *,
    has_residual: bool,
    fixed_reference: dict[str, Any] | None,
) -> str:
    detail = _first_rerun_detail(evidence)
    if not detail.get("logprobs_available"):
        return (
            "Logprob 对照：本 case 没有可用 token 置信度数据，因此不能用 logprob 解释模型偏好；"
            "结论只能依赖目标回合输出变化和 residual scan。"
        )

    avg = _format_metric(detail.get("avg_logprob"))
    minimum = _format_metric(detail.get("min_logprob"))
    tokens = _low_token_text(detail)
    if has_residual:
        if fixed_reference:
            return (
                "Logprob 对照：当前未修复输出的 avg_logprob="
                f"{avg}、min_logprob={minimum}，低置信/关键 token 集中在 `{tokens}`。"
                f"同批已修复样本 `{fixed_reference['file_name']}` 的参考值是 "
                f"avg_logprob={_format_metric(fixed_reference['avg'])}、"
                f"min_logprob={_format_metric(fixed_reference['min'])}，并且输出包含用户可见的升级说明。"
                "两者对比可以看出，未修复 case 不是在自然语言升级话术上摇摆，而是非常稳定地选择了纯 function-call 形态；"
                "这说明问题更像是 system prompt 对“固定话术 + required action 必须同时出现”的约束不够强。"
                "注意：logprob 只解释模型偏好，真正的未修复结论仍来自 residual scan。"
            )
        return (
            f"Logprob 观察：当前未修复输出 avg_logprob={avg}、min_logprob={minimum}，"
            f"关键 token 是 `{tokens}`。这些 token 指向模型稳定选择了当前输出形态；"
            "结合 residual scan，可以推导问题不是偶发波动，而是分支约束仍不足；"
            "正确性仍以 residual scan 为准。"
        )

    return (
        f"Logprob 观察：当前已修复输出 avg_logprob={avg}、min_logprob={minimum}，"
        f"低置信 token 主要是 `{tokens}`。这些 token 多出现在自然语言升级说明开头，"
        "而最终 correctness 由 residual scan=0 证明；因此 logprob 在这里只是补充说明模型生成该话术时的置信度。"
    )


def _prompt_change_report(item: dict[str, Any], *, has_residual: bool) -> str:
    edits = [edit for edit in item.get("prompt_edit_summaries") or [] if isinstance(edit, dict)]
    if not edits:
        return "Prompt 修改：本 case 没有记录 prompt edit，因此不能把结果归因于 system prompt 变更。"
    edit = edits[0]
    rationale = str(edit.get("rationale") or "").strip()
    feedback = str(edit.get("applied_feedback_summary") or "").strip()
    feedback_text = _translate_recommendation_for_report(feedback, "escalation_action_not_followed") if feedback else rationale
    feedback_text = feedback_text.rstrip("。.")
    if has_residual:
        return (
            "Prompt 修改推导：本轮确实对 system prompt 做了强化，修改目标是让 escalation trigger 后的第一条 assistant 回复"
            "停止协商，并执行 required spoken text 与 transfer/hotline action。"
            f"从修改说明看，后端主要强化了：{feedback_text}。"
            "但目标回合实际只从原始 tool call 变成了格式更规范的 tool call，没有产生用户可见固定话术，"
            "所以这次 prompt edit 只影响了工具调用形态，没有真正锁住 complete escalation response。"
        )
    return (
        "Prompt 修改推导：本轮将 system prompt 中 escalation 分支从较泛化的“触发后转接/停止协商”，"
        "强化为“第一条 assistant 回复必须同时包含用户可见的升级说明和 required transfer action”。"
        f"修改说明是：{feedback_text}。"
        "这个变化会让模型在命中 dispute/paid-claim 类触发后直接进入终局升级路径，"
        "因此目标回合补上固定升级说明后，residual scan 不再命中同类 badcase。"
    )


def _strip_report_label(text: str, *labels: str) -> str:
    stripped = str(text or "")
    for label in labels:
        if stripped.startswith(label):
            return stripped[len(label):].lstrip()
    return stripped


def _build_human_case_conclusions(
    conclusion: dict[str, Any],
    file_evidence: list[dict[str, Any]],
) -> str:
    evidence_by_source = {evidence.get("source_file"): evidence for evidence in file_evidence}
    fixed_reference = _fixed_logprob_reference(file_evidence)
    paragraphs: list[str] = []
    for item in conclusion.get("files") or []:
        evidence = evidence_by_source.get(item.get("source_file")) or {}
        if evidence.get("is_audit_only"):
            continue
        file_name = Path(str(item.get("source_file") or "unknown")).name
        applied_count = int(item.get("applied_case_count") or 0)
        fixed_count = int(item.get("fixed_case_count") or 0)
        residual_count = int(item.get("residual_badcase_count") or 0)
        has_residual = residual_count > 0
        turn, old_response, new_response = _first_rerun_delta(item)
        case_ids = ", ".join(str(case.get("id")) for case in item.get("applied_cases") or [] if case.get("id")) or "未记录"
        residual_cases = item.get("residual_badcases") or []
        residual_types = sorted({str(case.get("error_type") or "unknown") for case in residual_cases})
        if not residual_types or residual_types == ["unknown"]:
            residual_types = sorted(
                {
                    str(failure.get("error_type") or "unknown")
                    for failure in item.get("verification_failures") or []
                }
            )
        primary_error = residual_types[0] if residual_types else str(
            ((item.get("applied_cases") or [{}])[0]).get("error_type") or ""
        )
        cause = _error_type_natural_cause(primary_error)
        request_ids = [
            str(detail.get("request_id"))
            for detail in evidence.get("rerun_details") or []
            if detail.get("request_id")
        ]
        rounds = (
            f"{item.get('scan_event_id')} / {item.get('apply_event_id')} / "
            f"{item.get('residual_scan_event_id')}"
        )
        backend = f"{item.get('apply_provider') or 'unknown'}/{item.get('apply_model') or 'unknown'}"
        prompt_hash = f"{item.get('before_hash') or '-'} -> {item.get('after_hash') or '-'}"

        if has_residual:
            residual_ids = ", ".join(str(case.get("id")) for case in residual_cases if case.get("id")) or "未记录"
            status_sentence = (
                f"`{file_name}` 仍未修复。case `{case_ids}` apply 后生成了新的 residual case `{residual_ids}`，"
                f"残留类型是 `{', '.join(residual_types)}`。"
            )
            behavior_sentence = (
                f"turn {turn} 的旧输出是 `{old_response}`，新输出变成 `{new_response}`。"
                "这说明本轮修改确实影响了格式或工具调用，但没有补齐系统要求的完整终局升级动作。"
            )
            conclusion_sentence = (
                f"因此当前 badcase 的根因可以判断为：{cause}。"
            )
        else:
            status_sentence = (
                f"`{file_name}` 已修复。case `{case_ids}` 在 apply 后通过后置 residual scan，"
                f"该文件 residual_badcase_count=0。"
            )
            behavior_sentence = (
                f"turn {turn} 的旧输出是 `{old_response}`，新输出变成 `{new_response}`。"
                "新输出补上了用户可见的升级说明，并保留 required transfer action，所以 residual scan 不再命中同类违规。"
            )
            conclusion_sentence = (
                "这里的修复方式是强化该触发分支，让模型在命中 escalation 后直接执行终局动作。"
            )

        prompt_sentence = _prompt_change_report(item, has_residual=has_residual)
        logprob_sentence = _logprob_experiment_read(
            evidence,
            has_residual=has_residual,
            fixed_reference=fixed_reference,
        )
        evidence_sentence = (
            f"实验记录：applied={applied_count}，fixed={fixed_count}，residual={residual_count}；"
            f"本轮使用 {backend} 完成 prompt edit 和 targeted rerun；"
            f"prompt version hash={prompt_hash}；request id={', '.join(request_ids) or '无'}。"
            f"轮次链路为 {rounds}。这些是可追溯信息，不作为单独的 correctness 证明。"
        )
        paragraphs.append(
            "\n".join(
                [
                    f"#### {file_name}",
                    f"- 实验结论：{status_sentence}",
                    f"- 输出观察：{behavior_sentence}",
                    f"- Prompt 修改推导：{_strip_report_label(prompt_sentence, 'Prompt 修改推导：')}",
                    f"- Logprob 分析：{_strip_report_label(logprob_sentence, 'Logprob 观察：', 'Logprob 对照：')}",
                    f"- 根因判断：{conclusion_sentence}",
                    f"- 实验记录：{_strip_report_label(evidence_sentence, '实验记录：')}",
                ]
            )
        )

    return "\n\n".join(paragraphs) if paragraphs else "本轮没有 approved case 需要生成修复结论。"


def _build_human_next_action(
    conclusion: dict[str, Any],
    fixed: int,
    residual: int,
    unsupported: int,
    failures: int,
) -> str:
    def translate_next_experiment(text: str) -> str:
        normalized = " ".join(str(text or "").split())
        if normalized == "Extract or confirm the exact escalation message, then force deterministic spoken text plus any required action.":
            return "抽取或确认系统 prompt 中要求的升级固定话术，然后强制生成“固定话术 + required action”的完整终局回复"
        if normalized == "Use deterministic backend replacement.":
            return "使用确定性的后端替换策略，直接替换目标 assistant 回合为合规终局回复"
        if normalized.startswith("Route recognized beyond-maximum-date triggers"):
            return "将已识别的超期付款触发条件确定性路由到 RTP_Closing，避免再次进入协商分支"
        return normalized

    if residual == 0 and failures == 0 and unsupported == 0:
        return (
            "本轮已满足验收条件：所有 approved badcase 的 residual scan 都清零。"
            "建议保留当前 updated prompt/conversation，不要继续对同一批 case 做重复优化；"
            "后续如果新样本再次出现同类问题，再从新的 residual case 开始 continuation。"
        )
    if unsupported:
        return (
            "下一步先补齐 unsupported case 缺失的 ground truth，例如 exact spoken message、工具动作或业务终局文本。"
            "补齐后再重新构建 Repair Plan；没有 ground truth 时不要强行宣称修复。"
        )
    next_experiments = [
        next_experiment_for_failed_case(failure)
        for item in conclusion.get("files") or []
        for failure in item.get("verification_failures") or []
    ]
    if next_experiments:
        next_experiment = translate_next_experiment(next_experiments[0])
        return (
            f"下一步只处理 residual badcase，不重扫原始文件。建议动作：{next_experiment}。"
            "执行时在 residual continuation 中只替换该目标回合，生成“固定话术 + required action”的完整终局回复。"
            "验收标准是下一次 residual scan 中该 residual case 消失，且相邻非升级分支不被误触发。"
            "如果无法从 prompt 或工具定义中确认 exact message，应标记为 missing ground truth，而不是继续猜测。"
        )
    return (
        "下一步围绕当前 residual evidence 做一次 bounded continuation：只改命中的失败类别和目标 turn，"
        "然后执行一次 residual scan，以 residual 是否清零作为验收标准。"
    )


def build_batch_conclusion_sections(conclusion: dict[str, Any]) -> dict[str, Any]:
    approved = int(conclusion.get("approved_case_count") or 0)
    applied = int(conclusion.get("applied_case_count") or 0)
    fixed = int(conclusion.get("fixed_case_count") or 0)
    residual = int(conclusion.get("residual_badcase_count") or 0)
    unsupported = int(conclusion.get("unsupported_case_count") or 0)
    failures = int(conclusion.get("verification_failure_count") or 0)
    file_evidence = [_batch_file_conclusion_evidence(item) for item in conclusion.get("files") or []]
    focus_evidence = [evidence for evidence in file_evidence if not evidence.get("is_audit_only")]

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
            "验证结论：全部通过。"
            f"本轮共审批 {approved} 个 badcase，applied_case_count={applied}，"
            "后置 residual scan 未再暴露违规，residual_badcase_count=0。"
            f"已验证 turn：{fixed_turns or '无'}。"
        )
    elif fixed > 0:
        verdict = (
            "验证结论：部分通过。"
            f"本轮修复 {fixed} / {approved} 个 approved badcase，"
            f"残留 {residual} 个 badcase，失败 {failures} 个，unsupported {unsupported} 个。"
            f"已验证 turn：{fixed_turns or '无'}；残留 turn：{residual_turns or '无'}。"
        )
    else:
        verdict = (
            "验证结论：未通过。"
            f"本轮应用 {applied} / {approved} 个 badcase，但 residual scan 仍未关闭。"
            f"residual_badcase_count={residual}，verification_failure_count={failures}，unsupported_case_count={unsupported}。"
        )

    human_conclusions = _build_human_case_conclusions(conclusion, file_evidence)
    risk = _build_residual_risk_lines(conclusion)
    diagnosis = (
        "### 结论说明\n"
        f"{human_conclusions}\n\n"
        "### 剩余风险\n"
        f"{risk}"
    )

    next_action = _build_human_next_action(conclusion, fixed, residual, unsupported, failures)

    return {
        "verification_verdict": verdict,
        "badcase_diagnosis_and_backend_evidence": diagnosis,
        "next_action": next_action,
        "evidence": file_evidence,
    }


def render_apply_conclusion_markdown(conclusion: dict[str, Any]) -> str:
    sections = conclusion.get("conclusion_sections") or build_batch_conclusion_sections(conclusion)
    lines = [
        f"# 批量 Apply 结论：{conclusion['batch_id']}",
        "",
        "## 1. 验证结论",
        "",
        str(sections["verification_verdict"]),
        "",
        "## 2. 诊断与证据",
        "",
        str(sections["badcase_diagnosis_and_backend_evidence"]),
        "",
        "## 3. 下一步动作",
        "",
        str(sections["next_action"]),
        "",
        "## 附录：技术追踪摘要",
    ]
    for item in conclusion.get("files") or []:
        file_name = Path(str(item.get("source_file") or "unknown")).name
        request_ids = []
        for detail in item.get("rerun_results") or []:
            if not isinstance(detail, dict):
                continue
            diagnostics = detail.get("response_diagnostics")
            if isinstance(diagnostics, dict) and diagnostics.get("request_id"):
                request_ids.append(str(diagnostics.get("request_id")))
            elif detail.get("request_id"):
                request_ids.append(str(detail.get("request_id")))
        lines.append(
            f"- `{file_name}`：scan/apply/residual rounds="
            f"{item.get('scan_event_id')} / {item.get('apply_event_id')} / {item.get('residual_scan_event_id')}；"
            f"request_ids={', '.join(request_ids) or '无'}；"
            f"status={item.get('status')}；residual={item.get('residual_badcase_count', 0)}"
        )
    return "\n".join(lines)


def new_batch_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    raise SystemExit(main())
