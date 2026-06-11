# Conclusion Analysis

Read this reference before producing or debugging an Apply conclusion.

## Three Required Parts

### 1. Verification Verdict

State `verified fixed`, `partially fixed`, `not fixed`, or `verification incomplete`.

Support the verdict with:

- approved, applied, fixed, residual, unsupported, and verification-failure counts
- verified and residual turn ids
- residual scan evidence or proof that no residual cases remain
- relevant old/new response excerpts

Treat residual scan and observable target behavior as correctness evidence. Prompt edits, rerun completion, model confidence, and absence of backend errors are execution evidence, not proof of correctness.

### 2. Badcase Diagnosis And Backend Evidence

Keep this section concise, causal, and evidence-layered. Explain why the model reached the conclusion before listing the data. Use three visible layers:

- `Root cause analysis`: natural-language diagnosis of the likely branch-priority, routing, exact-message, tool, or missing-ground-truth cause. This is what humans should read first.
- `Evidence data (surface)`: user-visible behavior, residual verdict, file/turn/case ids, and the old-vs-new behavior difference.
- `Evidence data (deep)`: backend/model, prompt hash/diff summary, request ids, rerun targets/errors, tool state, and summarized logprob interpretation.

Connect:

- residual `error_type` and evidence
- old versus rerun response behavior
- prompt diff/hash and prompt-edit rationale
- backend provider/model, request ids, rerun targets, errors, and tool-call state
- source meta when available
- logprob summary when available

Do not paste full rerun response arrays, full token lists, or long residual evidence into Markdown. Truncate examples and keep the complete machine-readable detail in the JSON conclusion. The data should support the root-cause prose, not replace it.

Interpret logprobs carefully:

- High probability on a wrong response means the backend confidently preferred the wrong behavior; it suggests a routing, instruction-priority, or model/context problem.
- Low probability near the violating phrase may suggest generation uncertainty.
- Missing logprobs means no token-confidence conclusion is supported.
- Logprobs never override the residual scan verdict.

### 3. Next Action

Give:

- one primary action
- why current evidence supports it
- concrete implementation steps
- acceptance criteria
- one fallback if the primary action cannot be applied safely

Do not recommend repeating the same failed prompt-only experiment unless new evidence explains why the next attempt differs.

## Continuation Contract

When residual cases remain, the next action must be usable as a repair instruction for `continue-residual`.

- Name whether to continue with prompt edit, deterministic replacement, conversation-only rerun, or unsupported/missing-ground-truth handling.
- Keep the action scoped to residual cases only.
- Include acceptance criteria for the next residual scan.
- If the user approves the next action, run the residual-continuation workflow instead of rescanning the original file.
