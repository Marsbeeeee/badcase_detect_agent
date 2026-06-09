---
name: prompt-compliance-optimizer
description: Use only when the user explicitly asks Codex to operate, debug, or explain the Prompt Optimizer Agent workflow, inspect system-prompt compliance badcases, apply prompt fixes, rerun target conversation turns, analyze Trace List or Conclusion behavior, or troubleshoot the prompt_optimizer_agent app.
---

# Prompt Compliance Optimizer

## Overview

Use the existing Prompt Optimizer Agent app/model as the execution surface. This skill does not replace the app; it tells Codex how to drive and debug the workflow consistently.

For examples of how to invoke this skill, see [TRIGGERING.md](TRIGGERING.md).

Default repo path:

```bash
/Users/zlshlt2501003/Desktop/prompt_optimizer_agent
```

If this repository was cloned to another machine, use the clone path as the repo path.

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
6. After the user approves specific badcases, decide whether Apply should edit the prompt, rerun the target turn, or both.
7. After Apply, verify the Updated Conversation, inserted tool turns, prompt diff, residual scan result, and Round History records.
8. Return the required conclusion, then stop.
9. Run `generate badcase` again only when the user explicitly wants a fresh residual scan.

## Decision Rules

- Treat this as strict system-prompt compliance, not generic answer-quality review.
- Do not rewrite a prompt just because an answer could be better; require an explicit violated rule, workflow step, branch condition, tool rule, or fact constraint.
- If a trace requires a tool call and the current prompt already contains the rule, rerun the target conversation turn with the existing prompt instead of forcing a new prompt version.
- For missing required-tool-call traces, infer the exact tool from the current file's system prompt and tool definitions; the target assistant turn should call that tool before factual answers. Do not hardcode dataset-specific tool names.
- If rerun creates a function call, expect a placeholder `Tool` turn with `status: not_executed`; the app does not execute real tools locally.
- If the prompt changed, expect a new prompt version and a diff. If the prompt did not change but rerun succeeded, describe it as a conversation-only rerun, not as a failed prompt edit.

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
2. Run `tools/batch_prompt_compliance.py apply <batch_review.json>` with either `--approve-all` or an approval file.
3. Write one updated output file per input file.
4. Record per-file apply and residual scan rounds.
5. Return one aggregate conclusion plus per-file conclusion details.

Batch apply constraints:

- Batch auto-apply supports conversation-only rerun only for missing-required-tool-call cases where the required tool can be inferred from the current file's tool definitions.
- Unsupported approved case types, or cases whose required tool cannot be inferred, must be reported as unsupported and left for individual review; do not pretend they were fixed.
- The batch conclusion is mandatory and must include aggregate counts, per-file round ids, unsupported cases, residual badcase counts, and next action.

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
