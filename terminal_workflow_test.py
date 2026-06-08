from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompt_optimizer_agent.agent_logic import (
    BadCase,
    LLMSettings,
    optimize_system_prompt,
    rerun_conversation,
)
from prompt_optimizer_agent.json_utils import ConversationData, parse_conversation_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal test for Prompt Optimizer Agent workflow.")
    parser.add_argument("--input", required=True, help="Conversation JSON file.")
    parser.add_argument("--url", default=None, help="Company API base URL.")
    parser.add_argument("--provider", default=None, help="Company API provider.")
    parser.add_argument("--model", default=None, help="Company model name.")
    parser.add_argument("--manual-turn", type=int, default=5, help="Turn index to flag manually.")
    parser.add_argument(
        "--manual-feedback",
        default="The assistant did not follow the required flow. Update the system prompt to make the next required step explicit.",
        help="Manual bad-case feedback used for prompt optimization.",
    )
    parser.add_argument("--output", default="terminal_rerun_output.json", help="Output JSON path.")
    args = parser.parse_args()

    raw_text = Path(args.input).read_text(encoding="utf-8")
    parse_result = parse_conversation_json(raw_text)
    if parse_result.error or parse_result.data is None:
        raise SystemExit(f"Parse failed: {parse_result.error}")

    data = parse_result.data
    original_system_prompt = data.system_prompt
    model_info = {}
    if isinstance(data.source_meta, dict) and isinstance(data.source_meta.get("model_info"), dict):
        model_info = data.source_meta["model_info"]

    settings = LLMSettings(
        backend="company_api",
        base_url=args.url or str(model_info.get("url") or "http://192.168.101.15:9898"),
        provider=args.provider or str(model_info.get("provider") or "openai_api_like"),
        model=args.model or str(model_info.get("model") or "H200_01_9010"),
        max_completion_tokens=512,
    )

    bad_cases = [
        BadCase(
            turn_index=args.manual_turn,
            role=data.interactions[args.manual_turn].role
            if 0 <= args.manual_turn < len(data.interactions)
            else "unknown",
            error_type="human_flagged",
            evidence=args.manual_feedback,
            recommendation="Make the system prompt explicit about the correct next action for this case.",
            source="human",
        )
    ]

    optimization = optimize_system_prompt(
        data=data,
        bad_cases=bad_cases,
        manual_feedback={args.manual_turn: args.manual_feedback},
        llm_settings=settings,
    )
    approved_prompt = optimization.optimized_prompt
    data.system_prompt = approved_prompt

    rerun_results = rerun_conversation(
        data=data,
        optimized_prompt=approved_prompt,
        llm_settings=settings,
    )
    replacements = {
        result.assistant_turn_index: result.new_assistant_response
        for result in rerun_results
        if result.error is None
        and result.assistant_turn_index is not None
        and result.new_assistant_response
    }
    updated_interactions = [
        turn.model_copy(update={"content": replacements[index]})
        if index in replacements
        else turn
        for index, turn in enumerate(data.interactions)
    ]

    attempted = len([result for result in rerun_results if result.assistant_turn_index is not None])
    replaced = len(replacements)
    errors = [result.error for result in rerun_results if result.error]

    output_payload = {
        "settings": settings.__dict__,
        "system_prompt_updated": data.system_prompt == approved_prompt,
        "old_system_prompt_prefix": original_system_prompt[:300],
        "optimized_prompt_prefix": approved_prompt[:300],
        "optimization_rationale": optimization.rationale,
        "attempted_assistant_turns": attempted,
        "replaced_assistant_turns": replaced,
        "errors": errors,
        "updated_interactions": [turn.model_dump() for turn in updated_interactions],
        "rerun_results": [result.__dict__ for result in rerun_results],
    }
    Path(args.output).write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("parse_ok=True")
    print(f"warnings={parse_result.warnings}")
    print(f"turns={len(data.interactions)} tools={len(data.tools or {})}")
    print(f"company_backend={settings.provider}:{settings.model} url={settings.base_url}")
    print(f"system_prompt_updated={data.system_prompt == approved_prompt}")
    print(f"attempted_assistant_turns={attempted}")
    print(f"replaced_assistant_turns={replaced}")
    print(f"errors={errors}")
    for result in rerun_results:
        if result.assistant_turn_index is not None:
            print(
                "sample_rerun="
                + json.dumps(
                    {
                        "user_turn": result.user_turn_index,
                        "assistant_turn": result.assistant_turn_index,
                        "old": result.old_assistant_response[:120],
                        "new": result.new_assistant_response[:120],
                        "error": result.error,
                    },
                    ensure_ascii=False,
                )
            )
            break
    print(f"output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
