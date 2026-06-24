from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt_optimizer_agent.rerun_logprobs import (  # noqa: E402
    DEFAULT_COMPANY_MODEL,
    DEFAULT_COMPANY_PROVIDER,
    DEFAULT_COMPANY_URL,
    RerunLogprobsSettings,
    conversation_payload,
    parse_turn_specs,
    rerun_target_turns_with_logprobs,
    utc_timestamp,
)
from prompt_optimizer_agent.json_utils import parse_conversation_json  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "skill_scan" / "rerun"
LATEST_RERUN_UPDATED_JSON = "latest_updated.json"
LATEST_RERUN_DIAGNOSTICS_JSON = "latest_diagnostics.json"


def main() -> int:
    args = build_parser().parse_args()
    source_path = Path(args.conversation_json).expanduser()
    raw = source_path.read_text(encoding="utf-8")
    parsed = parse_conversation_json(raw)
    if parsed.error or parsed.data is None:
        raise SystemExit(parsed.error or "Could not parse conversation JSON.")

    target_turns = parse_turn_specs(args.turn or [])
    if not target_turns:
        raise SystemExit("At least one --turn value is required, e.g. --turn 9 or --turn 9,11.")

    prompt = parsed.data.system_prompt
    if args.system_prompt_file:
        prompt = Path(args.system_prompt_file).expanduser().read_text(encoding="utf-8")

    settings = RerunLogprobsSettings(
        model=args.model,
        provider=args.provider,
        url=args.url,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        top_logprobs=args.top_logprobs,
        request_logprobs=not args.no_logprobs,
        transport=args.transport,
        timeout_seconds=args.timeout_seconds,
        context_window_turns=args.context_window_turns,
        insert_tool_placeholders=not args.no_tool_placeholders,
    )

    result = rerun_target_turns_with_logprobs(
        data=parsed.data,
        optimized_prompt=prompt,
        target_assistant_turn_indices=target_turns,
        settings=settings,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json
        else output_dir / LATEST_RERUN_UPDATED_JSON
    )
    diagnostics_json = (
        Path(args.diagnostics_json).expanduser()
        if args.diagnostics_json
        else output_dir / LATEST_RERUN_DIAGNOSTICS_JSON
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_json.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(
        json.dumps(conversation_payload(result.updated_data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    diagnostics = result.diagnostics_payload(
        source_path=str(source_path),
        output_path=str(output_json),
        settings=settings,
        include_raw_response=not args.no_raw_response,
        include_request_payload=args.include_request_payload,
    )
    diagnostics["run_id"] = args.run_id or utc_timestamp().replace(":", "").replace(".", "-")
    diagnostics_json.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "source": str(source_path),
                "updated_json": str(output_json),
                "diagnostics_json": str(diagnostics_json),
                "target_turns": sorted(target_turns),
                "result_count": diagnostics["aggregate"]["result_count"],
                "error_count": diagnostics["aggregate"]["error_count"],
                "logprobs_available_count": diagnostics["aggregate"]["logprobs_available_count"],
                "logprobs_requested": not args.no_logprobs,
                "top_logprobs": None if args.no_logprobs else args.top_logprobs,
                "transport": args.transport,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun selected assistant turns through llmparty/direct chat completions "
            "while requesting token logprobs and writing diagnostics."
        )
    )
    parser.add_argument("conversation_json", help="Conversation JSON to rerun.")
    parser.add_argument(
        "--turn",
        action="append",
        default=[],
        help="Assistant turn index to rerun. Accepts repeated values, comma lists, or ranges, e.g. --turn 9,11.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional file containing the updated system prompt. Defaults to the JSON system_prompt.",
    )
    parser.add_argument("--url", default=DEFAULT_COMPANY_URL, help="Chat completions base URL.")
    parser.add_argument("--provider", default=DEFAULT_COMPANY_PROVIDER, help="llmparty provider.")
    parser.add_argument("--model", default=DEFAULT_COMPANY_MODEL, help="Model name.")
    parser.add_argument(
        "--transport",
        choices=["auto", "llmparty", "direct"],
        default="direct",
        help="Use llmparty APIClient, direct /v1/chat/completions, or auto. Default: direct.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--no-logprobs", action="store_true", help="Do not request logprobs.")
    parser.add_argument(
        "--context-window-turns",
        type=int,
        default=-1,
        help="Rerun context window. -1 keeps the app-style auto window; 0 keeps full context.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--no-tool-placeholders",
        action="store_true",
        help="Do not insert not_executed tool placeholder turns after generated tool calls.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--diagnostics-json", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--no-raw-response",
        action="store_true",
        help="Omit raw model responses from diagnostics.",
    )
    parser.add_argument(
        "--include-request-payload",
        action="store_true",
        help="Include full request payloads, including messages, in diagnostics.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
