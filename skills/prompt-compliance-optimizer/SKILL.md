---
name: prompt-compliance-optimizer
description: Operate, debug, or explain this repository's Prompt Optimizer Agent and strict system-prompt-compliance workflow. Use only when the user explicitly asks to inspect prompt-compliance badcases, generate or review Trace List items, approve/apply prompt fixes, rerun target conversation turns, run batch scan/apply workflows, analyze Updated Conversation or Conclusion behavior, inspect optimization rounds, or troubleshoot the prompt_optimizer_agent app.
---

# Prompt Compliance Optimizer

Use the existing Prompt Optimizer Agent app and backend as the execution surface. Do not replace the app with an unrelated prompt-review workflow.

Use the current repository root. If invoked elsewhere, locate the repo containing `app.py`, `prompt_optimizer_agent/`, `tools/`, and `logs/`.

Read only the reference needed for the task:

- Read [references/repair-plan.md](references/repair-plan.md) before applying approved badcases.
- Read [references/batch-mode.md](references/batch-mode.md) for multiple JSON files or a folder.
- Read [references/debug-checklist.md](references/debug-checklist.md) when debugging UI state, reruns, prompt versions, or conclusions.
- Read [references/conclusion-analysis.md](references/conclusion-analysis.md) before producing or debugging an Apply conclusion.
- Read [TRIGGERING.md](TRIGGERING.md) only when the user asks how to invoke this skill.

## Core Workflow

1. Load or parse the conversation JSON.
2. Inspect the Original system prompt, Working system prompt, conversation, and tool definitions.
3. Generate badcases and align each trace with its Original or Updated conversation source.
4. Present candidate badcases for human review, record or cite the scan round id, and stop.
5. Continue only after the user approves specific cases or explicitly approves all.
6. Build a complete Repair Plan for every approved case using [references/repair-plan.md](references/repair-plan.md).
7. Apply the smallest valid prompt edit, target-turn rerun, or both.
8. Verify Updated Conversation changes, tool calls and placeholder tool turns, prompt diff, exactly one post-apply residual scan, and Round History.
9. Return the required conclusion and stop.
10. If the conclusion reports residual badcases, wait for human approval before continuing. After approval, continue from the residual conclusion rather than restarting from the original file.

Start the UI only when needed:

```powershell
streamlit run app.py
```

## Compliance Boundary

- Judge strict system-prompt compliance, not generic answer quality.
- Require an explicit violated rule, workflow step, branch condition, tool rule, exact-message requirement, or fact constraint.
- Do not rewrite a prompt merely because an answer could be improved.
- Treat the residual scan as the source of truth. Say `applied changes`, not `fixed`, when violations remain.
- Do not fabricate missing ground truth, tool results, dates, amounts, hotline text, or business decisions.

## Repair Routing

- If the current prompt already contains a clear required-tool rule, prefer a conversation-only target-turn rerun.
- Infer required tools from the current file's system prompt and tool definitions. Never hardcode dataset-specific tool names.
- If exact spoken text is required, the updated assistant turn must contain that text and any required tool/action. A tool-call-only replacement is insufficient.
- If a rerun creates a function call, expect a placeholder `Tool` turn with `status: not_executed`; local reruns do not execute real tools.
- If the prompt changes, require a new prompt version and diff.
- If only the conversation changes, describe it as a conversation-only rerun, not a failed prompt edit.

Use this autonomous repair ladder for approved prompt-edit cases:

1. `exact_patch`
2. `anchored_section_patch`
3. `bounded_full_prompt_rewrite`
4. `conversation_only_rerun`, only when no prompt edit is needed

If all applicable strategies fail, report `backend_failed_to_autonomously_repair`, list the attempted strategies and backend/model details, and stop with residual badcases for review. Do not ask the user to manually author the prompt patch.

## Human Review Gates

- Badcase discovery is automatic; badcase confirmation is manual.
- Stop after presenting candidate badcases.
- Do not apply unapproved cases.
- Treat `approve all`, `fix all`, `all cases`, and equivalent explicit approval as approval of every currently listed candidate.
- Do not ask the user to manually operate the UI during normal use unless debugging UI behavior.

## Round History

- Ground every scan/apply/residual-scan cycle in app state or `logs/optimization_rounds.jsonl`; do not rely on chat context alone.
- Record or cite the scan round id after discovery.
- Record or cite apply and residual-scan round ids after Apply.
- Use round ids to distinguish repeated optimization cycles.

## Output Hygiene

- Keep `outputs/skill_scan` as the latest review snapshot only: `batch_review.json` and `batch_review.md`.
- Keep `outputs/skill_scan/applied` as the latest apply snapshot only: `batch_apply_conclusion.json`, `batch_apply_conclusion.md`, and the current apply's `*_updated.json` file(s).
- Overwrite those stable files on each run. Do not create batch-id-named output files or batch-id-named subdirectories.
- Clean stale generated snapshots before writing new ones. Historical traceability belongs in `logs/optimization_rounds.jsonl`, not in accumulated review/apply files.

## Required Conclusion

After every approved Apply cycle, return exactly three semantic parts:

1. `Verification verdict`: state whether the approved badcases were actually fixed, partially fixed, or not fixed. Treat the residual scan as source of truth.
2. `Badcase diagnosis and backend evidence`: explain what behavior was wrong and connect it to prompt diff/hash, backend/model, rerun results/errors, tool state, and residual evidence.
3. `Next action`: give the single best action for the verified state. Stop when verified; otherwise target the recorded failure category or missing ground truth.

Never equate `prompt_changed` or `rerun_attempted` with successful repair.
Use logprobs only as confidence evidence about the generated output, never as correctness evidence. Follow [references/conclusion-analysis.md](references/conclusion-analysis.md).

For batch conclusions, use the expanded requirements in [references/batch-mode.md](references/batch-mode.md).

## Stop Rules

- Stop when the user says `stop skill`, `stop using the skill`, or equivalent.
- Resume only when the user explicitly mentions `$prompt-compliance-optimizer`, links the skill, or asks to resume it.
- Stop after candidate review output.
- Stop after an Apply cycle is verified, recorded, and concluded.
- Run no fresh scan unless the user explicitly asks. Apply verification permits exactly one post-apply residual scan. A user approval such as `continue`, `apply residual`, or `use the next step` after a residual conclusion authorizes one residual-continuation cycle.
- If no valid badcase exists, report that no fix is needed and stop.
- If required input or ground truth is missing, identify it and stop.

## Repository Surfaces

- `app.py`: Streamlit state, Trace List UI, Apply flow, conversation view, prompt versions.
- `prompt_optimizer_agent/agent_logic.py`: judge, prompt edit, targeted rerun, required-tool forcing, conclusion payload.
- `prompt_optimizer_agent/json_utils.py`: JSON and tool-call wrapper parsing.
- `tools/batch_prompt_compliance.py`: batch scan/apply CLI.
- `logs/optimization_rounds.jsonl`: scan/apply/residual-scan history.
- `logs/company_api_requests.jsonl`: outgoing request audit.
- `logs/company_api_preflight_system_prompt.log`: preflight prompt hash and targeted rerun request.
- `logs/company_api_system_prompts.log`: latest outgoing system prompt.

## Validation

After code changes, run the smallest relevant tests first. On this Windows workspace, prefer:

```powershell
$env:PYTHONPYCACHEPREFIX = (Resolve-Path '.pycache_tmp').Path
uv run --with pydantic --with openai --with requests --with pytest python -m pytest tests/test_agent_logic.py tests/test_batch_prompt_compliance.py
```

Run `python -m py_compile` on changed Python modules when the full tests are unnecessary or unavailable.
