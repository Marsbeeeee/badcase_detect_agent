# Triggering Examples

Use this skill only for explicit Prompt Optimizer Agent or strict system-prompt-compliance work.

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

Natural-language requests that explicitly mention Prompt Optimizer Agent, prompt-compliance badcases, Trace List, targeted rerun, Updated Conversation, batch scan/apply, or optimization-round conclusions may also trigger the skill.
