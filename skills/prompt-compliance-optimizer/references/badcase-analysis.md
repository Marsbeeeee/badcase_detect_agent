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

## Required Standalone Report Format

Lead with the outcome. Use the following sections, omitting only sections for which no evidence exists:

1. `分析对象`
2. `结论`
3. `关键对话状态` or `当前轮次的正确流程`
4. `正确路由`
5. `目标模型的实际输出`
6. `原因分析`
7. `与其他模型的对比`
8. `Meta 信息分析`
9. `为什么不是技术性提前停止` or `终止类型判断`
10. `待验证原因假设`
11. `阶段状态`

The `结论` must answer in the first paragraph:

- what the model did;
- whether it ended the conversation;
- the primary cause;
- the correct branch.

The `原因分析` must be causal, not a repetition of the judge. Cover state, trigger interpretation, rule priority, and fixed-template selection when applicable.

The `待验证原因假设` section must separate observational evidence from inference. Express every proposed cause as a falsifiable hypothesis and map it to exactly one experimental variable.

The `阶段状态` section must say `Stage 1 analysis complete` and explicitly record `experiments not run` and `prompt not modified`.

Do not test prompt edits that merely hardcode the private case, brand, exact user sentence, or one model name.

## Compact Multi-Case Report Exception

When the input contains multiple JSONL records, multiple files, or a folder, do not repeat the full standalone report for every case. Use `assets/stage1-multi-case-analysis-template.md` and write one aggregate Markdown report with a compact index plus one three-part card per successfully parsed case.

For every case, include exactly:

1. `分析对象、正确流程和实际对比`: identify the case, target turn/model, current user input, badcase response, correct route, and visible deviation. Limit the correct-route and deviation explanations to one or two sentences each.
2. `原因分析`: give only evidence-supported causes. Limit each cause to two or three sentences. Include a `Meta 证据` cause that interprets why generation config, decisive-token logprobs, output length, task status, PPL, EOS, or tool state supports, amplifies, weakens, or excludes a cause; do not merely list parameters.
3. `最终归因`: summarize the causal chain in one or two sentences, then list compact labels and termination type.

Fold comparison-model evidence into the cause paragraphs, for example, `6/7 comparison models remained at the consent gate`; do not create a per-case model table. Quote only the target user input and target badcase response unless one prompt sentence is indispensable.

Do not include `待验证原因假设`, experiment variables, experiment suggestions, improvement recommendations, or per-case test plans in a multi-case Stage 1 report. List parse failures, missing target candidates, and insufficient-meta cases in `未完成分析的 Case` rather than silently omitting them.

Meta language must be causal and conservative. Use conclusions such as `sampling variance amplified the branch error because the correct and wrong first-token logprobs were close`, `the first token confidently entered the wrong fixed template, so sampling is not the primary explanation`, or `task_success and complete short output exclude technical truncation`. Never infer sampling from temperature alone or treat PPL/logprob as correctness evidence.

## Evidence Quality Rules

- Quote only the minimum prompt and response excerpts needed to prove routing.
- Link the local source file when reporting to the user.
- Use exact numbers from meta; do not estimate missing values.
- Mark inference explicitly when checkpoint training data or internal reasoning is unavailable.
- Do not claim a training cause from one sample. Say the behavior is consistent with a state-tracking or routing weakness.
- Preserve the distinction between a judge verdict and independent causal analysis.
- Preserve the distinction between an observational hypothesis and an experimentally supported cause.
- Always write the Stage 1 Markdown report. Use the user-specified folder or the source-adjacent `badcase分析报告` default, verify the file exists, and link it in the final response.
