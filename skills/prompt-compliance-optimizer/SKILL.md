---
name: prompt-compliance-optimizer
description: Use only when the user explicitly asks Codex to operate, debug, or explain the Prompt Optimizer Agent workflow, inspect system-prompt compliance badcases, apply prompt fixes, rerun target conversation turns, run batch scan/apply workflows, analyze Trace List or Conclusion behavior, or troubleshoot the prompt_optimizer_agent app.
---

# Prompt Compliance Optimizer

## Overview

Use the existing Prompt Optimizer Agent app/model as the execution surface. This skill does not replace the app; it tells Codex how to drive and debug the workflow consistently.

For examples of how to invoke this skill, see [TRIGGERING.md](TRIGGERING.md).

Default repo path: use the current repository root. If the skill is installed outside the project, first locate the repo that contains `app.py`, `prompt_optimizer_agent/`, `tools/`, and `logs/`.

Start the UI only when needed:

```bash
streamlit run app.py
```

## Workflow

1. Load or parse the conversation JSON.
2. Inspect `Original system prompt`, `Working system prompt`, and available tool definitions.
3. Click or run `generate badcase` to produce Trace List items.
4. Check whether each trace comes from `Original` or `Updated` conversation and keep the right-side conversation view aligned with the trace source.
5. Present candidate badcases for human review and stop before applying fixes.
6. After the user approves specific badcases, produce a Repair Plan with `Root-cause`, `Patch target`, and `Verification cases` for every approved case.
7. Use the Repair Plan to decide whether Apply should edit the prompt, rerun the target turn, or both.
8. After Apply, verify the Updated Conversation, inserted tool turns, prompt diff, residual scan result, and Round History records.
9. Return the required conclusion, then stop.
10. Run `generate badcase` again only when the user explicitly wants a fresh residual scan.

## Stop Directive

- If the user says `停止使用 skill` or `stop skill`, stop applying this skill after acknowledging the instruction.
- After the stop directive, do not read this skill, follow its workflow, or rely on its rules for ordinary project-code tasks.
- Resume this skill only when the user explicitly mentions `$prompt-compliance-optimizer`, links this skill, or says to resume using the skill.

## Decision Rules

- Treat this as strict system-prompt compliance, not generic answer-quality review.
- Do not rewrite a prompt just because an answer could be better; require an explicit violated rule, workflow step, branch condition, tool rule, or fact constraint.
- If a trace requires a tool call and the current prompt already contains the rule, rerun the target conversation turn with the existing prompt instead of forcing a new prompt version.
- For missing required-tool-call traces, infer the exact tool from the current file's system prompt and tool definitions; the target assistant turn should call that tool before factual answers. Do not hardcode dataset-specific tool names.
- Do not treat exact-message escalation as a tool-call-only fix. If the violated rule requires exact spoken text, hotline wording, or a configured escalation message, the updated assistant turn must contain that required text and any required tool/action.
- If rerun creates a function call, expect a placeholder `Tool` turn with `status: not_executed`; the app does not execute real tools locally.
- If the prompt changed, expect a new prompt version and a diff. If the prompt did not change but rerun succeeded, describe it as a conversation-only rerun, not as a failed prompt edit.

## Repair Plan Rules

Before applying approved badcases, produce a Repair Plan. Do not apply prompt edits or reruns until every approved case has a complete `Root-cause`, `Patch target`, and `Verification cases` entry.

`Root-cause` must explain why the assistant could plausibly violate the system prompt, not merely restate that it violated the rule. Include:

- `case_id`, turn, file, and failure type.
- The violated rule or workflow branch.
- The trigger condition in the user turn.
- The observed wrong assistant behavior.
- One root-cause category:
  - `missing_branch`: the prompt lacks the required branch.
  - `weak_branch_priority`: the branch exists but can be overridden by another flow.
  - `missing_forbidden_behavior`: the prompt does not forbid the observed wrong action.
  - `missing_exact_message`: the prompt requires exact text but does not make it operational enough.
  - `ambiguous_trigger`: the trigger is unclear or underspecified.
  - `tool_rule_underbound`: the required tool, timing, or arguments are unclear.
  - `ground_truth_missing`: required date, amount, hotline, metadata, tool result, or business decision is missing.
- A concise root-cause sentence tied to the current system prompt.

`Patch target` must constrain the repair to the smallest prompt or rerun surface that can fix the approved case. Include:

- `section_hint`: the prompt section, branch, or policy area to edit.
- `operation`: one of `add_missing_branch`, `strengthen_existing_branch`, `strengthen_branch_priority`, `add_forbidden_behavior`, `strengthen_exact_action`, `clarify_tool_rule`, or `conversation_only_rerun`.
- `required_change`: the specific rule/action that must be added or strengthened.
- `must_include`: required behaviors, exact text requirements, tool call requirements, priority rules, or terminal-routing language.
- `must_not_change`: neighboring flows that must remain intact.
- `apply_mode`: `prompt edit`, `conversation-only rerun`, or `both`.

`Verification cases` must show how the repair will be proven and bounded. Include at least:

- One `positive` case from the approved badcase turn or an equivalent turn that must now pass.
- One `negative` case where the new rule must not trigger.
- One `regression` case for an adjacent branch, priority rule, exact-message rule, or tool-call rule that must remain unchanged.

If any of the three parts cannot be completed because the prompt or metadata lacks necessary ground truth, report the case as `unsupported_by_missing_ground_truth` and stop before Apply for that case.

The Repair Plan is not a request for the human to manually edit the system prompt. Its purpose is to constrain autonomous repair. After a case is approved, the backend must attempt to produce and apply the prompt change itself.

Autonomous prompt repair should use this fallback ladder:

1. `exact_patch`: apply an exact in-place replacement when the original substring can be located safely.
2. `anchored_section_patch`: if exact replacement fails, locate the target section or branch from `section_hint` and insert or rewrite only that bounded section.
3. `bounded_full_prompt_rewrite`: if section anchoring fails but the full prompt is available, generate a complete updated prompt that preserves unrelated content and changes only the `Patch target` intent.
4. `conversation_only_rerun`: use only when the current prompt already contains the rule and the Repair Plan says no prompt edit is needed.

Do not hand off failed prompt edits to the human as manual prompt-writing work. If every autonomous repair strategy fails, report `backend_failed_to_autonomously_repair`, include the failed strategy names and backend/model details, and stop with residual badcases for review.

## Human Review Rules

- Badcase discovery is automatic.
- Badcase confirmation is manual.
- Do not apply prompt edits or reruns until the user approves specific badcases.
- If the user says "approve all", "全修", or equivalent, apply all currently listed candidate badcases.
- Do not ask the user to manually operate the UI during normal use unless debugging UI behavior.

## Batch Mode

Use Batch Mode only when the user explicitly provides multiple JSON files or a folder.

Batch scan workflow:

1. Run `tools/batch_prompt_compliance.py scan <file-or-folder> ...`.
2. Produce a batch review JSON and Markdown report.
3. Record one scan round per file in `logs/optimization_rounds.jsonl`.
4. Summarize candidate badcases by file.
5. Stop for human review before applying any fixes.

Batch apply workflow:

1. Continue only after the user approves all files, selected files, or selected case ids.
2. Produce a Repair Plan for every approved case and mark unsupported cases before Apply.
3. Run `tools/batch_prompt_compliance.py apply <batch_review.json>` with either `--approve-all` or an approval file.
4. Write one updated output file per input file.
5. Record per-file apply and residual scan rounds.
6. Return one aggregate conclusion plus per-file conclusion details.

Batch apply constraints:

- Batch apply must route approved cases before applying:
  - `missing-required-tool-call`: use deterministic conversation-only required-tool replacement only when the required tool can be inferred from the current file's tool definitions and no exact spoken message is required.
  - `prompt-flow violation`: use LLM prompt edit, then targeted rerun of the affected assistant turn(s), then residual scan.
  - `exact-message escalation`: do not use function-call-only replacement. The fix must produce the required spoken text and any required tool/action; otherwise report the case as unsupported or backend-failed.
- Batch apply must use the Repair Plan to bound prompt edits. Do not make broad prompt rewrites when the approved cases point to a specific branch, priority rule, forbidden behavior, exact message, or tool-call rule.
- Batch apply must attempt the autonomous prompt repair fallback ladder before giving up on an approved prompt-edit case. Do not require a human to manually author the system prompt patch.
- Unsupported approved case types, cases whose required tool cannot be inferred, or prompt edits where the backend fails to produce an applicable in-place patch/full prompt must be reported as unsupported; do not pretend they were fixed.
- Treat residual scan as the source of truth. If changes were applied but residual badcases remain, say "applied changes" instead of "fixed".
- The batch conclusion is mandatory and must include aggregate counts, verified fixed counts, per-file round ids, apply mode, Repair Plan coverage, backend/model for prompt edits, unsupported cases, verification failures, residual badcase counts, failure category counts, and next action.
- Use these failure categories in conclusions:
  - `fixable_by_prompt_clarification`: the intended rule exists but must be made more explicit, operational, or machine-checkable.
  - `unsupported_by_missing_ground_truth`: required exact text, metadata, tool result, date, amount, hotline, or business decision is missing.
  - `backend_failed_to_patch`: the apply backend did not produce an acceptable in-place patch/full prompt.
  - `backend_failed_to_autonomously_repair`: all autonomous prompt repair strategies failed.
  - `patch_applied_but_failed_verification`: prompt edit and/or rerun happened, but residual scan still found the violation.
  - `likely_model_or_context_limited`: use only after multiple controlled experiments show failure despite clear prompt and sufficient metadata.

Company model selection:

- Default to the configured Voyager company model; do not ask the user to choose a model unless they request it or the default model fails.
- When the user wants alternatives, run `tools/batch_prompt_compliance.py apply --list-models --model-contains <keyword>` to show rough matches.
- If applying with a rough match, use `--model-contains <keyword>` only when it matches exactly one company model. If multiple models match, show the candidate list and stop for user selection.

## Round History Rules

- Do not rely on chat context alone to know the current round.
- Every scan/apply/residual-scan cycle must be grounded in the app state or `logs/optimization_rounds.jsonl`.
- After generating badcases, record or reference the scan round id.
- After Apply, record or reference the apply round id and the residual scan round id.
- If multiple optimization cycles happen, use the recorded round ids to distinguish them.

## Required Conclusion After Apply

After every approved Apply or Apply selected cycle, always return a conclusion. This is mandatory.

The conclusion must include:

- What was fixed: approved badcase ids/turns, fix type (`prompt edit`, `target-turn rerun`, or `both`), and whether a new prompt version was created.
- Verification result: Updated Conversation changes, inserted tool calls or placeholder tool turns, prompt diff if any, residual badcase scan result, and Round History event ids.
- Next action: if residual badcases remain, list them and stop for human review; if none remain, state that the cycle is complete.

Use this concise shape:

```text
Conclusion:
Fixed <N> approved badcase(s): <turn/id list>. The fix was <prompt edit / conversation-only rerun / both>. <A new prompt version was created / No new prompt version was needed>.

Verification:
Updated Conversation: <what changed>. Residual scan: <0 or N badcases>. Round History: scan=<id>, apply=<id>, residual_scan=<id>.

Next:
<Cycle complete / residual badcases need human review>. Do not start another scan unless the user asks.
```

## Stop Rules

- Stop after presenting candidate badcases for human review.
- After approved fixes are applied, verified, recorded, and concluded, stop.
- Do not run another residual badcase scan unless the user explicitly asks or the current Apply verification requires exactly one post-apply residual scan.
- If no valid badcase is found, report that no fix is needed and stop.
- If required input is missing or ambiguous, ask for the missing input and stop.

## Debugging

Read [references/debug-checklist.md](references/debug-checklist.md) when debugging UI state, rerun behavior, badcase highlighting, prompt versions, or conclusion wording.

Common files:

- `app.py`: Streamlit state, Trace List UI, Apply flow, conversation view, prompt versions.
- `prompt_optimizer_agent/agent_logic.py`: judge, prompt edit, targeted rerun, required tool forcing, conclusion payload.
- `prompt_optimizer_agent/json_utils.py`: JSON parsing, tool-call wrapper parsing.
- `logs/company_api_requests.jsonl`: outgoing request audit.
- `logs/company_api_preflight_system_prompt.log`: preflight prompt hash and targeted rerun request.
- `logs/company_api_system_prompts.log`: latest outgoing system prompt.
- `logs/optimization_rounds.jsonl`: product-level scan/apply/residual-scan round history.

## Validation

After code changes, run the smallest meaningful checks first:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/prompt_optimizer_pycache python3 -m py_compile app.py prompt_optimizer_agent/agent_logic.py prompt_optimizer_agent/json_utils.py prompt_optimizer_agent/company_demo_client.py
```

If `pytest` is unavailable, manually run the relevant test function with `.venv39/bin/python` or explain the missing dependency.
