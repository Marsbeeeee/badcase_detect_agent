# Standalone Badcase Analysis

Read this reference before diagnosing a known prompt-compliance benchmark badcase or writing its report.

## Contents

- Objective
- Analysis Workflow
- Required Standalone Report Format
- Evidence Quality Rules

## Objective

Explain causally why the target model produced the violating response, why comparison models differed, and whether an apparent conversation ending was a business decision or a technical termination. Produce testable hypotheses, write the Stage 1 Markdown report, and stop without running experiments.

Analyze strict prompt compliance rather than generic response quality. Ground every conclusion in the conversation, system rules, candidate output, judge evidence, or recorded meta.

## Analysis Workflow

### 1. Parse the evaluation topology

Identify:

- case id, file, target turn, target candidate model, and target output;
- chronological `dialog` content before the target turn;
- candidate outputs under the target turn's evaluation map;
- LAEP/key-guide remark, judge result, metrics, and root meta;
- generation config and token-level meta when present.

Do not confuse the root `assistant_model` with a nested candidate model. In turn-level benchmark files, the chronological dialog may come from a base conversation while each evaluated model supplies only the target-turn candidate. Do not attribute another candidate's earlier response to the target model unless the file explicitly contains a model-specific rollout.

Treat LAEP remarks, personas, judge annotations, and evaluation meta as non-model-visible unless the harness proves they were injected into the request.

### 2. Reconstruct the active workflow state

Build a concise state timeline from observable turns. Record completed gates and active branches, for example:

```text
identity_confirmed = true
consent_obtained = true
active_step = payment_negotiation
caller_legitimacy_disputed = true
```

Check whether the target response:

- skips a mandatory gate;
- reopens a completed gate;
- advances one or more steps without the required trigger;
- regresses to an earlier branch;
- confuses similarly named states or actors;
- selects a terminal branch whose preconditions are absent.

Prefer precise diagnoses such as `mandatory_gate_skipped`, `workflow_state_regression`, or `wrong_terminal_branch` over vague labels such as "did not understand the prompt."

### 3. Map triggers to prompt rules

Extract the minimum relevant rules and rank them by explicit prompt priority:

1. policy and safety rules;
2. mandatory gates and branch preconditions;
3. FAQ or exception handling;
4. active workflow step;
5. tone and formatting constraints.

For compound user turns, separate intents before routing. Explicitly distinguish direction and actor, for example:

- bank verifying customer identity;
- customer verifying caller legitimacy;
- refusing payment;
- refusing identity confirmation;
- asking for sensitive credentials;
- claiming identity theft.

If two prompt rules genuinely compete, describe the ambiguity. Still determine whether the target response is invalid under every reasonable route or only under the judge's narrower expected route.

### 4. Identify template and branch selection

Compare the target output against exact strings in the system prompt.

If it reproduces a fixed message from the wrong branch, conclude that the model likely selected the wrong workflow node and then generated that node confidently. Do not label this as free-form hallucination merely because the business behavior is wrong.

State both:

- the branch the model selected;
- the branch it should have selected and why.

### 5. Compare all candidate models fairly

Create a compact comparison containing model, behavior, judge/human score, and whether it continued or closed the workflow.

Do not claim every non-target model is correct. Separate:

- fully compliant candidates;
- safe but incomplete candidates;
- differently noncompliant candidates;
- the target model's distinctive failure.

Use comparison models to show which state distinction or rule priority the target missed.

### 6. Determine whether the conversation technically stopped

When the user says the model "directly ended," test two hypotheses.

#### Business-terminal output

Evidence includes:

- explicit closing language such as goodbye or a prompt-defined close template;
- exact match to a terminal workflow branch;
- a semantically complete response;
- successful generation status.

#### Technical termination

Check when available:

- `finish_reason` or stop reason;
- `task_success` and backend errors;
- output token count versus `max_completion_tokens`;
- malformed or abruptly incomplete text;
- tool-call failure or missing tool continuation;
- final visible token and EOS alternatives;
- request or server failure logs.

Do not infer technical truncation merely because the file ends at the target turn. A benchmark slice can end while the response itself is asking a question and awaiting the next turn.

Use exact wording in the conclusion:

- `business decision caused by wrong terminal branch`, or
- `technical termination supported by ...`, or
- `the model ended the current workflow stage, not the conversation`.

### 7. Interpret meta and logprobs conservatively

For a turn-level candidate, first read the badcase model's own sequence at
`dialog[i].evaluate.<target-model>.meta.logprobs`. Do not use the root model's
meta or a comparison candidate's logprobs. Treat each item as a token, not
necessarily a human word. Reconstruct the visible response by concatenating
`bytes` when valid; otherwise concatenate `token`. Verify that this reconstructed
text aligns with the badcase response before drawing token-level conclusions.

Scan the complete sequence in response order. The goal is to locate semantic
departure, not to sort tokens by logprob or find the numeric minimum. Determine:

1. the last prefix that is still compatible with the correct route;
2. the earliest token or minimal token span after which the response commits to
   a violating act, wrong branch, prohibited claim, or wrong terminal template;
3. the uncertainty window immediately before that boundary; and
4. whether later tokens show branch lock-in or recovery.

Call item 2 the `first bad token/span`. If generic opening tokens are shared by
correct and incorrect routes, do not mark them bad merely because their
probability is low. If compliance cannot be decided until a multi-token phrase
is complete, report a span rather than inventing single-token precision. If the
boundary cannot be aligned because token meta is missing, incomplete, or does
not reconstruct the response, report `token boundary unavailable` and explain
the exact mismatch.

For the boundary and a compact surrounding window, inspect internally:

- zero-based token index, displayed token, and chosen-token logprob;
- chosen-token probability `exp(logprob)` when the value is finite and not a
  saturated sentinel;
- the most relevant `top_logprobs` alternatives and their logprobs;
- the margin between the chosen token and the best recorded alternative;
- the semantic effect of the chosen token on the workflow route.

Use the recorded sequence to answer only the diagnostic questions supported by
the evidence:

- Where did the model first become locally uncertain before the bad branch?
- Did the wrong continuation beat a semantically correct alternative by only a
  small recorded margin, or was it clearly ahead?
- Was a semantically correct local continuation present as the second- or
  third-ranked `top_logprobs` candidate?
- When aligned pre-change and post-change generations exist, did the target
  continuation become more confident after the prompt change?
- Did the model commit to the wrong branch confidently, or continue while
  wavering among alternatives?

Do not manufacture an answer for every question. A correct candidate must be
semantically compatible with the correct workflow route, not merely a plausible
synonym. Compare pre-change and post-change confidence only for the same target
turn, model/checkpoint, aligned semantic decision point, and materially
unchanged decoding settings; otherwise state that the confidence change is not
comparable. If the correct continuation is absent from the recorded top-k, do
not infer its rank or probability.

The analysis must inspect every token, but the Stage 1 report should include only
the compact token evidence needed to support its causal conclusion. For a
confirmed badcase, include the earliest reliable human-readable word or minimal
phrase where the response commits to the error, joining subword tokens into
normal text. For an omission, identify the premature closing or transition
phrase that made the required content unrecoverable; if no reliable lexical
boundary exists, say so. Do not show an annotated full response, token indices,
token rows, raw logprob dumps, or full top-alternative lists unless the user
explicitly asks for raw evidence or a token trace. A compact Meta interpretation
table is required by the single-case template.

Report relevant configuration:

- temperature, top-p/top-k, sample count, and completion limit;
- task success, output length, model/checkpoint, and backend;
- PPL and judge/human scores;
- first decisive token, violating phrase, and final token when useful.

Use logprobs only as confidence evidence:

- close alternative probabilities at the first branch token can support sampling-sensitive routing uncertainty;
- a high-confidence wrong template supports a confident routing or instruction-priority error;
- saturated values such as chosen token `0` with alternatives `-9999` are not calibrated probabilities;
- absent lexical alternatives do not prove that no competing internal path existed;
- low PPL does not prove compliance and can reflect confident reproduction of the wrong template.
- the lowest-probability token is not necessarily the first bad token;
- a top alternative is evidence about local lexical competition, not proof of a
  complete hidden correct branch;
- if the correct-route wording is absent from `top_logprobs`, say it was not
  observed in the recorded candidate set rather than assigning it zero chance.

Do not call sampling the primary cause unless the recorded alternatives show a meaningful competing path. Separate:

- primary semantic/state error;
- contributing prompt ambiguity;
- sampling amplifier, if supported.

### 8. Produce testable causal hypotheses

Use only labels supported by evidence. Prefer this hierarchy:

- **Primary**: `workflow_state_transition_error`, `workflow_state_regression`, `policy_precedence_error`, `wrong_terminal_branch`, `missing_required_tool_call`, or `exact_message_not_used`;
- **Subtype**: the concrete gate, branch, actor, or trigger misclassification;
- **Semantic error**: e.g. `verification_direction_confusion`;
- **Contributing factor**: e.g. `overlapping_rule_precedence` or `fixed_template_overactivation`;
- **Amplifier**: e.g. `sampling_variance`, only when supported;
- **Excluded causes**: truncation, tool failure, abnormal EOS, or missing ground truth when evidence rules them out.

Convert each non-excluded cause into a falsifiable hypothesis. Define:

- the single variable to change;
- the expected target-turn behavior if the hypothesis is correct;
- the unchanged controls;
- the pass/fail criterion.

Do not execute the experiments during Stage 1. Do not call a hypothesis experimentally verified merely because it sounds plausible.

Use these labels only as internal taxonomy. In the report, translate them into a
natural-language causal explanation that states the lost or misread state, the
incorrect interpretation, the resulting branch or omission, and the role of any
contributing factor. Do not end a case with labels or code-like identifiers, and
do not use a label as a substitute for the explanation. Show taxonomy labels
only when the user explicitly requests them.

## Required Standalone Report Format

Render a single-case Stage 1 report from
`assets/stage1-single-case-analysis-template.md`. Lead with the outcome and use
the complete evidence-backed structure demonstrated by the template: `分析对象`,
`结论`, `关键对话状态`, `正确路由`, `目标模型的实际输出`, `原因分析`,
`与其他模型的对比`, `Meta 信息分析`, `终止类型判断`, `待验证原因假设`,
`最终归因`, and `阶段状态`.

The `结论` must answer in the first paragraph:

- what the model did;
- whether it ended the conversation;
- the primary cause;
- the correct branch.

The `原因分析` must be causal, not a repetition of the judge. Explain how state,
trigger interpretation, rule priority, and fixed-template selection caused the
observed result. Use only the minimum conversation and prompt excerpts needed to
prove routing.

The candidate comparison must distinguish fully compliant, safe but incomplete,
and differently noncompliant outputs. The Meta section must interpret the target
candidate's complete ordered token-logprob sequence, generation status, length,
and stopping evidence. Keep token evidence compact and relevant; do not dump the
full raw sequence.

The `待验证原因假设` section must separate observation from inference. Express
each supported cause as a falsifiable hypothesis with exactly one changed
variable, expected behavior, unchanged controls, and a pass/fail criterion. Do
not execute those experiments during Stage 1.

The `阶段状态` section must say `Stage 1 analysis complete` and explicitly record `experiments not run` and `prompt not modified`.

Do not test prompt edits that merely hardcode the private case, brand, exact user sentence, or one model name.

## Compact Multi-Case Report Exception

When the input contains multiple JSONL records, multiple files, or a folder, use
`assets/stage1-multi-case-analysis-template.md` and write one aggregate Markdown
conclusion report with a compact index plus one conclusion block per successfully
parsed case.

For every case, include only the causal conclusion, correct behavior, root cause,
logprob-supported conclusion, comparison conclusion, termination conclusion, and
a natural-language final attribution. For a confirmed badcase, also include the
first reliable error word or phrase when one can be aligned. Analyze the complete
conversation and ordered logprob sequence internally, but do not display per-turn
content, full quoted responses, token tables, raw numbers, or Meta parameter
listings.

Fold comparison-model evidence into a one-sentence conclusion; do not create a
per-case model table or quote the target turn and response.

Do not include `待验证原因假设`, experiment variables, experiment suggestions,
improvement recommendations, or per-case test plans in a multi-case Stage 1
report. List parse failures, missing target candidates, and insufficient-meta
cases in `暂无结论的 Case` rather than silently omitting them.

Token-level language must be causal and conservative. Use conclusions such as `the first reliable divergence is the phrase “cannot verify your identity,” which commits to the already-completed identity gate`, `the wrong continuation only narrowly beat a correct-route alternative, so sampling may have amplified the routing error`, `the wrong template was already clearly ahead at the decisive phrase, so sampling is not the primary explanation`, or `task_success and complete short output exclude technical truncation`. Never infer sampling from temperature alone, equate the minimum-logprob token with the bad boundary, or treat PPL/logprob as correctness evidence.

## Evidence Quality Rules

- Quote only the minimum prompt and response excerpts needed to prove routing.
- Link the local source file when reporting to the user.
- Use exact numbers from meta; do not estimate missing values.
- Mark inference explicitly when checkpoint training data or internal reasoning is unavailable.
- Do not claim a training cause from one sample. Say the behavior is consistent with a state-tracking or routing weakness.
- Preserve the distinction between a judge verdict and independent causal analysis.
- Preserve the distinction between an observational hypothesis and an experimentally supported cause.
- Always write the Stage 1 Markdown report. Use the user-specified folder or the source-adjacent `badcase分析报告` default, verify the file exists, and link it in the final response.
