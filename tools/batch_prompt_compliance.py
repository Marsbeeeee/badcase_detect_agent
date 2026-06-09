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
)
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


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "scan":
        return run_scan(args)
    if args.command == "apply":
        return run_apply(args)
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
    scan.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    scan.add_argument("--log", default=str(DEFAULT_ROUND_LOG))
    scan.add_argument("--batch-id", default=None)

    apply = subparsers.add_parser("apply", help="Apply approved cases from a batch review file.")
    apply.add_argument("review_json", help="Path to batch_review.json produced by scan.")
    apply.add_argument("--approve-all", action="store_true", help="Apply every case in the review file.")
    apply.add_argument(
        "--approval-file",
        default=None,
        help="JSON approval file. Accepts a list of case ids or {'approved_case_ids': [...]}",
    )
    apply.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "applied"))
    apply.add_argument("--log", default=str(DEFAULT_ROUND_LOG))
    apply.add_argument("--batch-id", default=None)
    return parser


def run_scan(args: argparse.Namespace) -> int:
    files = discover_files([Path(path).expanduser() for path in args.paths], args.pattern, not args.no_recursive)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    review_json = output_dir / f"{batch_id}_review.json"
    review_md = output_dir / f"{batch_id}_review.md"
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
    review_path = Path(args.review_json).expanduser()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    batch_id = args.batch_id or f"{review.get('batch_id', 'batch')}-apply"
    output_dir = Path(args.output_dir).expanduser() / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)
    approved_ids = load_approved_ids(args, review)
    if not approved_ids:
        raise SystemExit("No approved case ids. Use --approve-all or provide --approval-file.")

    file_results = []
    for index, file_record in enumerate(review.get("files") or [], start=1):
        result = apply_file_record(file_record, approved_ids, output_dir)
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
        "unsupported_case_count": sum(len(item["unsupported_cases"]) for item in file_results),
        "residual_badcase_count": sum(item["residual_badcase_count"] for item in file_results),
        "files": file_results,
    }
    conclusion_json = output_dir / "batch_apply_conclusion.json"
    conclusion_md = output_dir / "batch_apply_conclusion.md"
    conclusion_json.write_text(json.dumps(conclusion, ensure_ascii=False, indent=2), encoding="utf-8")
    conclusion_md.write_text(render_apply_conclusion_markdown(conclusion), encoding="utf-8")

    print(
        json.dumps(
            {
                "batch_id": batch_id,
                "approved_case_count": conclusion["approved_case_count"],
                "applied_case_count": conclusion["applied_case_count"],
                "unsupported_case_count": conclusion["unsupported_case_count"],
                "residual_badcase_count": conclusion["residual_badcase_count"],
                "conclusion_json": str(conclusion_json),
                "conclusion_md": str(conclusion_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


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


def apply_file_record(file_record: dict[str, Any], approved_ids: set[str], output_dir: Path) -> dict[str, Any]:
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
    replacements: dict[int, Interaction] = {}
    applied_case_records: list[dict[str, Any]] = []
    unsupported_cases = []
    for case in approved_cases:
        required_tool = required_tool_for_case(data, case)
        if not required_tool:
            unsupported_cases.append(
                {
                    **case,
                    "reason": (
                        "Batch auto-apply only supports missing-required-tool-call cases when the "
                        "required tool can be inferred from this file's tool definitions."
                    ),
                }
            )
            continue
        target_turn = int(case["turn_index"])
        query = query_for_case(data, case, target_turn)
        replacements[target_turn] = required_tool_interaction(target_turn, required_tool, query)
        applied_case_records.append({**case, "applied": True, "required_tool": required_tool})

    updated_interactions = replace_interactions_with_placeholders(data.interactions, replacements)
    updated_data = data.model_copy(update={"interactions": updated_interactions})
    residual_cases = [
        bad_case_record(case)
        for case in _local_prompt_rule_bad_cases(updated_data)
        if case.turn_index >= 0
    ]

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
            "prompt_changed": False,
            "before_hash": sha1_text(data.system_prompt),
            "after_hash": sha1_text(data.system_prompt),
            "before_version": "external Original",
            "after_version": "external Original",
            "residual_badcase_count": len(residual_cases),
            "residual_badcases": residual_cases,
            "required_tools": sorted(
                {
                    str(case.get("required_tool"))
                    for case in applied_case_records
                    if case.get("required_tool")
                }
            ),
        }
    )
    return result


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
    }


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
        "badcases": result.get("applied_cases", []),
        "unsupported_cases": result.get("unsupported_cases", []),
        "required_tools": result.get("required_tools", []),
        "prompt_changed": result.get("prompt_changed", False),
        "before_hash": result.get("before_hash"),
        "after_hash": result.get("after_hash"),
        "before_version": result.get("before_version"),
        "after_version": result.get("after_version"),
        "status": result.get("status"),
        "status_message": "Batch conversation-only apply completed.",
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


def render_apply_conclusion_markdown(conclusion: dict[str, Any]) -> str:
    lines = [
        f"# Batch Apply Conclusion: {conclusion['batch_id']}",
        "",
        "## Conclusion",
        "",
        (
            f"Fixed {conclusion['applied_case_count']} approved badcase(s) across "
            f"{conclusion['file_count']} file(s). The fix was conversation-only rerun for supported "
            "missing-required-tool-call cases. No new prompt version was created."
        ),
        "",
        "## Verification",
        "",
        (
            f"Residual scan found {conclusion['residual_badcase_count']} remaining badcase(s). "
            f"Unsupported approved cases: {conclusion['unsupported_case_count']}."
        ),
        "",
        "## Files",
        "",
    ]
    for item in conclusion.get("files") or []:
        lines.extend(
            [
                f"- `{item.get('source_file')}`",
                f"  - status: `{item.get('status')}`",
                f"  - updated_file: `{item.get('updated_file')}`",
                f"  - scan/apply/residual: `{item.get('scan_event_id')}` / `{item.get('apply_event_id')}` / `{item.get('residual_scan_event_id')}`",
                f"  - required_tools: {', '.join(item.get('required_tools') or []) or 'none'}",
                f"  - applied: {item.get('applied_case_count', 0)}, residual: {item.get('residual_badcase_count', 0)}, unsupported: {len(item.get('unsupported_cases') or [])}",
            ]
        )
    lines.extend(
        [
            "",
            "## Next",
            "",
            (
                "Cycle complete. Do not start another scan unless requested."
                if conclusion["residual_badcase_count"] == 0
                else "Residual badcases remain and need human review."
            ),
            "",
        ]
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
