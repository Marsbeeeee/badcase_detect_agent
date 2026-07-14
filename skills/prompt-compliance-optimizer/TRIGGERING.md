# Triggering Examples

Use this skill only for explicit Prompt Optimizer Agent, benchmark badcase diagnosis, or strict system-prompt-compliance work.

## Analyze A Known Benchmark Badcase

```text
$prompt-compliance-optimizer

Analyze the target model's final response in <benchmark.jsonl>. Reconstruct the active workflow state,
compare the candidate models and judge results, interpret relevant meta/logprobs, determine whether the
conversation ended through a wrong business branch or technical termination, write the Markdown analysis
report, and stop without running experiments.
```

## Run Controlled Experiments Separately

```text
$prompt-compliance-optimizer

Using the Stage 1 report for <benchmark.jsonl> turn <n> model <checkpoint>, run Stage 2 only.
Test each cause with isolated single-variable target-turn reruns, write the separate Markdown experiment
report, and stop without applying any variant.
```

## Inspect And Stop For Review

```text
$prompt-compliance-optimizer

Inspect <conversation.json> for strict system-prompt compliance badcases.
Generate the candidate list, cite the scan round id, and stop before Apply.
```

## Apply Approved Cases

```text
$prompt-compliance-optimizer

Approve cases 1 and 3 from <batch_review.json>.
Build the Repair Plan, apply only those cases, verify one residual scan,
return the required Conclusion, and stop.
```

## Continue From Residual Conclusion

```text
$prompt-compliance-optimizer

Continue from outputs/skill_scan/applied/batch_apply_conclusion.json.
Use the Conclusion next action for the remaining residual badcases, apply approved residual cases,
run one residual scan, return the three-part Conclusion, and stop.
```

## Debug A Workflow

```text
$prompt-compliance-optimizer

Debug why Turn 8 remains a badcase after Apply. Inspect Updated Conversation,
the prompt diff, required tool call, residual scan, and Round History.
```

## Batch Scan

```text
$prompt-compliance-optimizer

Batch scan all conversation JSON files in <folder>. Produce the review JSON
and Markdown report, summarize candidates by file, then stop for approval.
```

Natural-language requests that explicitly mention benchmark or prompt-compliance badcase analysis, candidate model/run/checkpoint comparison, unexpected terminal responses, controlled experiments, Prompt Optimizer Agent, Trace List, targeted rerun, Updated Conversation, batch scan/apply, or optimization-round conclusions may also trigger the skill. Execute only the requested stage.
