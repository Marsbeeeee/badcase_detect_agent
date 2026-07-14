# Batch Mode

Use Batch Mode only when the user explicitly provides multiple JSON files or a folder.

## Multi-Case Stage 1 Analysis

When the user already identifies the target badcase response/model and asks to analyze multiple JSONL records, treat the task as Stage 1 analysis rather than candidate discovery. Parse every record, inspect its visible conversation state, target candidate, comparison candidates, judge/human evidence, and meta/logprobs, then write one stable aggregate Markdown report using `assets/stage1-multi-case-analysis-template.md`.

Use `<input-stem>_multi_case_analysis.md` for one multi-record JSONL file. Use `multi_case_badcase_analysis.md` for an explicit file list or folder unless the user specifies a name. Overwrite the same stable report on repeat runs.

Each case must contain only:

1. analysis object, correct flow, and actual comparison;
2. concise cause analysis, with each cause limited to two or three sentences and Meta expressed as causal evidence;
3. final attribution in one or two sentences plus compact labels.

Do not append experiment suggestions, improvement recommendations, or testable experiment plans. Record parsing failures, absent target candidates, and insufficient evidence in the report instead of dropping cases. Stop after linking the Stage 1 report.

## Scan

1. Run `tools/batch_prompt_compliance.py scan <file-or-folder> ...`.
2. Update the latest snapshot files `batch_review.json` and `batch_review.md` in the output directory.
3. Remove stale generated review files such as `*_review.json` and `*_review.md` before writing the new snapshot.
4. Record one scan round per file in `logs/optimization_rounds.jsonl`.
5. Summarize candidate badcases by file.
6. Stop for human review before Apply.

## Apply

1. Continue only after approval of all files, selected files, or selected case ids.
2. Build a Repair Plan for every approved case and mark unsupported cases.
3. Run `tools/batch_prompt_compliance.py apply <batch_review.json>` with `--approve-all` or an approval file.
4. Update one `<source-stem>_updated.json` output per input file and the stable `batch_apply_conclusion.json/.md` snapshots directly in the apply output directory.
5. Remove stale generated apply files such as `*_updated.json`, `*_conclusion.json`, and `*_conclusion.md` before writing the new snapshot.
6. Record per-file apply and residual-scan rounds.
7. Return one aggregate conclusion plus per-file conclusions.

The output directories must contain only the latest generated snapshot for each function. Do not create batch-id-named review files, batch-id-named conclusion files, or apply subdirectories. Keep historical round records in `logs/optimization_rounds.jsonl`; `batch_id` remains a tracing identifier, not an output path.

## Residual Continuation

Use this only after an Apply conclusion reports residual badcases and the user approves continuing from the conclusion's next action.

1. Run `tools/batch_prompt_compliance.py continue-residual <batch_apply_conclusion.json>`.
2. This creates the next `batch_review.json/.md` from residual badcases, not from the original scan.
3. It copies the conclusion's next action or each verification failure's `next_experiment` into the residual case recommendation.
4. Stop for human review unless the user already approved applying residual cases.
5. If approved, run with `--approve-all` to create the residual review, apply it, rerun target turns, and run one residual scan in the same command.

Example:

```powershell
uv run --with pydantic --with openai --with requests python tools\batch_prompt_compliance.py continue-residual outputs\skill_scan\applied\batch_apply_conclusion.json --approve-all --review-output-dir outputs\skill_scan --apply-output-dir outputs\skill_scan\applied
```

Do not repair already verified cases during continuation. Use `parent_case_id`, `parent_apply_batch_id`, and Round History ids to keep the chain auditable.

## Routing

- `missing-required-tool-call`: use deterministic conversation-only required-tool replacement only when the tool can be inferred and no exact spoken message is required.
- `prompt-flow violation`: use a bounded prompt edit, targeted rerun, and residual scan.
- `exact-message escalation`: require the spoken text and any required tool/action; a function-call-only replacement is insufficient.
- Unsupported case types, uninferable required tools, and failed autonomous repairs must remain unsupported. Never pretend they were fixed.

## Apply Constraints

- Apply only approved cases.
- Use the Repair Plan to bound edits.
- Attempt the autonomous repair ladder before reporting failure.
- Treat the residual scan as the source of truth.
- Default to the configured Voyager company model. Ask about models only when requested or when the default fails.
- For alternatives, run `apply --list-models --model-contains <keyword>`.
- Use `--model-contains` for Apply only when it matches exactly one company model.

## Failure Categories

- `fixable_by_prompt_clarification`
- `unsupported_by_missing_ground_truth`
- `backend_failed_to_patch`
- `backend_failed_to_autonomously_repair`
- `patch_applied_but_failed_verification`
- `likely_model_or_context_limited`, only after multiple controlled experiments

## Aggregate Conclusion

Return exactly three semantic parts:

1. Verification verdict based on fixed, residual, unsupported, and verification-failure counts.
2. Badcase diagnosis connected to per-file round ids, backend/model, prompt hashes, rerun results, residual evidence, and failure categories.
3. One next action selected from the verified state and recorded `next_experiment`.
