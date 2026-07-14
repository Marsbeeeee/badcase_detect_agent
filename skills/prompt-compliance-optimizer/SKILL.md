---
name: prompt-compliance-optimizer
description: Run stage-gated prompt-compliance benchmark analysis, controlled experiments, and approved repair workflows. Use when the user explicitly asks to diagnose a known model response in JSON/JSONL, compare run/checkpoint outputs and meta/logprobs, explain a wrong or terminal branch, generate a Markdown badcase report, separately test suspected causes one variable at a time, generate or review Trace List items, approve/apply prompt fixes, rerun target turns, run batch scan/apply workflows, analyze Updated Conversation or Conclusion behavior, inspect optimization rounds, or troubleshoot the prompt_optimizer_agent app. Execute only the requested stage and never advance automatically.
---

# Prompt Compliance Optimizer

Use the existing Prompt Optimizer Agent app and backend as the execution surface. Do not replace the app with an unrelated prompt-review workflow.

Use the current repository root. If invoked elsewhere, locate the repo containing `app.py`, `prompt_optimizer_agent/`, `tools/`, and `logs/`.

Read only the reference needed for the task:

- Read [references/badcase-analysis.md](references/badcase-analysis.md) before Stage 1 analysis of a known benchmark badcase, candidate comparison, meta/logprob interpretation, unexpected conversation ending, or analysis-report creation.
- Read [references/controlled-experiments.md](references/controlled-experiments.md) only before Stage 2 controlled experiments. Use the bundled concurrent runner for repeated or multi-case experiments.
- Read [references/repair-plan.md](references/repair-plan.md) before applying approved badcases.
- Read [references/batch-mode.md](references/batch-mode.md) for multiple JSON files or a folder.
- Read [references/debug-checklist.md](references/debug-checklist.md) when debugging UI state, reruns, prompt versions, or conclusions.
- Read [references/conclusion-analysis.md](references/conclusion-analysis.md) before producing or debugging an Apply conclusion.
- Read [TRIGGERING.md](TRIGGERING.md) only when the user asks how to invoke this skill.

## Mode Selection

- **Stage 1 — Analyze**: use when the user asks why a known response is a badcase, requests a conclusion, or asks for model/meta comparison. Analyze using [references/badcase-analysis.md](references/badcase-analysis.md), always write the Markdown analysis report, link it, and stop. Do not call a model, edit a prompt, or run an experiment.
- **Stage 2 — Experiment**: use only when the user explicitly asks to test, experiment, control variables, or rerun hypotheses. Read the Stage 1 report or reconstruct its hypotheses, then use [references/controlled-experiments.md](references/controlled-experiments.md). Always write a separate Markdown experiment report and stop. Do not Apply a winning variant.
- **Scan/review**: use when the user asks to discover candidate badcases. Follow the Core Workflow through the human review gate and stop.
- **Stage 3 — Apply/repair and verify**: use only after explicit approval of listed cases or an explicitly selected experimental variant. Follow the full Core Workflow and Required Conclusion.
- **Debug**: use the relevant debug reference and inspect repository state without broadening into scan or repair unless requested.

## Stage Gates

- Execute exactly one stage unless the user explicitly requests multiple named stages in the same message.
- Treat `analyze`, `diagnose`, `why`, `give me the conclusion`, and equivalent wording as Stage 1 only.
- Treat `run the experiment`, `test the causes`, `control variables`, `rerun`, and equivalent wording as Stage 2 only.
- Treat `apply`, `use this variant`, `fix`, `approve`, and equivalent wording as Stage 3 only, subject to existing approval gates.
- A completed Stage 1 authorizes no Stage 2 calls. A completed Stage 2 authorizes no Stage 3 mutation.
- End the turn at the requested stage boundary. State the completed stage in the report and final response.

## Automatic Stage Reports

- Always create a Markdown artifact for Stage 1 and Stage 2, even when the user does not separately ask for a file.
- If the user specifies a folder, use it. Otherwise, for a local JSON/JSONL source, create or reuse a sibling `badcase分析报告` directory.
- Use stable filenames derived from case id, turn, target model slug, and stage: `<case_id>_turn_<n>_<model>_analysis.md` and `<case_id>_turn_<n>_<model>_experiment.md`.
- Sanitize filename characters without changing the identifiers written inside the report.
- Overwrite the same stable stage report on a repeated run; do not create timestamped duplicates.
- Verify the report exists and return a clickable link. Keep the chat response concise because the Markdown report is the complete deliverable.
- Render Stage 2 reports from [assets/stage2-experiment-report-template.md](assets/stage2-experiment-report-template.md). A fully completed run must use every template section. A partial or blocked run must still write the same stable report, include every completed experiment, and add the exact errors, skipped experiments, and blockers; never suppress the report because the run was incomplete.
- For Stage 1 analysis of multiple JSONL records, multiple files, or a folder, render one compact aggregate report from [assets/stage1-multi-case-analysis-template.md](assets/stage1-multi-case-analysis-template.md). Analyze every case separately, but restrict each case to exactly three parts: `分析对象、正确流程和实际对比`, `原因分析`, and `最终归因`. Omit experiment suggestions and test plans.

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
- For standalone analysis, distinguish a wrong business-terminal branch from technical generation termination. Do not call a response "truncated" or "EOS-stopped" without supporting meta.
- Treat exact reproduction of the wrong prompt template as branch-selection evidence, not hallucination evidence.
- Distinguish model-visible conversation content from evaluation-only metadata such as LAEP remarks, judge annotations, and root meta unless the harness demonstrably injects them.
- Stage 1 reports end with testable hypotheses and no speculative improvement recommendations. Do not test them until Stage 2 is explicitly requested.
- In Stage 2, test supported hypotheses when the target backend/checkpoint is callable; otherwise report the exact experimental blocker and leave the hypothesis unverified.
- Never present an experimental prompt variant as an applied or production fix. Applying a winning variant still requires the normal approval and repair workflow.

## Deterministic Check Generalization

- Local deterministic checks may supplement the LLM judge, but they must be reusable rule-pattern detectors, not scripts tailored to one uploaded system prompt, one dataset, one file, one state name, one fixed utterance, one brand, or one tool name.
- Do not add prompt-specific detector functions like a hardcoded busy-check, current-year, silence-stop, late-payment, open-ended-payment, or attempt-limit detector unless the implementation first derives the rule, trigger, limits, names, and required behavior from the current `system_prompt`, tool definitions, or structured input.
- Detector names should describe generic violation patterns such as `missing_required_tool_call`, `workflow_branch_not_followed`, `attempt_limit_exceeded`, or `exact_message_not_used`; avoid names that encode a single business prompt's private workflow.
- Every new deterministic check must require three evidence surfaces: the prompt rule evidence, the user trigger evidence, and the assistant violation evidence.
- Every new deterministic check must include at least one positive test, one negative test, and one adjacent-branch regression test proving the detector is not overfit to a single badcase.
- If a case cannot be generalized without hardcoding a specific prompt's private wording or workflow labels, leave it to the LLM judge plus human review rather than adding a specialized function.

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

## Encoding Integrity

- Keep repository Python, Markdown, JSON, and JSONL files valid UTF-8 with LF endings.
- If source text shows mojibake or unterminated string literals, repair the damaged helper or prompt text before running scan/apply.
- After repairing encoding-sensitive code, run `python -m py_compile` on the changed module and the focused batch tests.

## Required Conclusion

After every approved Apply cycle, return exactly three semantic parts:

1. `Verification verdict`: state whether the approved badcases were actually fixed, partially fixed, or not fixed. Treat the residual scan as source of truth.
2. `Badcase diagnosis and backend evidence`: lead with natural-language `Root cause analysis` that explains why the model reached the wrong or verified conclusion. Then preserve supporting data as `Evidence data (surface)` for visible behavior/residual scan results and `Evidence data (deep)` for prompt hash/diff, backend/model, request ids, rerun state, tool state, and logprob interpretation. Do not dump full rerun arrays or long raw evidence in Markdown.
3. `Next action`: give the single best action for the verified state. Stop when verified; otherwise target the recorded failure category or missing ground truth.

Never equate `prompt_changed` or `rerun_attempted` with successful repair.
Use logprobs only as confidence evidence about the generated output, never as correctness evidence. Follow [references/conclusion-analysis.md](references/conclusion-analysis.md).

For batch conclusions, use the expanded requirements in [references/batch-mode.md](references/batch-mode.md).

## Stop Rules

- Stop when the user says `stop skill`, `stop using the skill`, or equivalent.
- Resume only when the user explicitly mentions `$prompt-compliance-optimizer`, links the skill, or asks to resume it.
- Stop after candidate review output.
- Stop after Stage 1 analysis report creation. Do not run baseline or variant experiments.
- Stop after Stage 2 experiment report creation. Do not Apply the winning variant.
- Stop after an Apply cycle is verified, recorded, and concluded.
- Run no fresh scan unless the user explicitly asks. Apply verification permits exactly one post-apply residual scan. A user approval such as `continue`, `apply residual`, or `use the next step` after a residual conclusion authorizes one residual-continuation cycle.
- If no valid badcase exists, report that no fix is needed and stop.
- If required input or ground truth is missing, identify it and stop.

## Repository Surfaces

- `app.py`: Streamlit state, Trace List UI, Apply flow, conversation view, prompt versions.
- `prompt_optimizer_agent/agent_logic.py`: judge, prompt edit, targeted rerun, required-tool forcing, conclusion payload.
- `prompt_optimizer_agent/json_utils.py`: JSON and tool-call wrapper parsing.
- `tools/batch_prompt_compliance.py`: batch scan/apply CLI.
- `scripts/run_stage2_experiments.py`: baseline-gated concurrent Stage 2 runner with resume state and stable partial/full Markdown reports.
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
