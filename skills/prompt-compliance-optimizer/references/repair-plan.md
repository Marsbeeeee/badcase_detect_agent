# Repair Plan

Read this reference before applying approved badcases.

## Required Coverage

Create one complete Repair Plan entry per approved case. Do not Apply a case until its entry contains `Root-cause`, `Patch target`, and `Verification cases`.

If required ground truth is absent, mark the case `unsupported_by_missing_ground_truth` and stop before Apply for that case.

## Root-cause

Include:

- `case_id`, turn, file, and failure type
- violated rule or workflow branch
- trigger condition in the user turn
- observed wrong assistant behavior
- one root-cause category
- a concise cause tied to the current system prompt

Allowed categories:

- `missing_branch`
- `weak_branch_priority`
- `missing_forbidden_behavior`
- `missing_exact_message`
- `ambiguous_trigger`
- `tool_rule_underbound`
- `ground_truth_missing`

Explain why the assistant could plausibly violate the prompt; do not merely restate the violation.

## Patch Target

Constrain the repair to the smallest valid surface. Include:

- `section_hint`
- `operation`
- `required_change`
- `must_include`
- `must_not_change`
- `apply_mode`

Allowed operations:

- `add_missing_branch`
- `strengthen_existing_branch`
- `strengthen_branch_priority`
- `add_forbidden_behavior`
- `strengthen_exact_action`
- `clarify_tool_rule`
- `conversation_only_rerun`

Allowed apply modes are `prompt edit`, `conversation-only rerun`, and `both`.

The plan constrains autonomous repair; it is not a request for the human to edit the prompt.

## Verification Cases

Include at least:

- one `positive` case from the approved turn or an equivalent turn
- one `negative` case where the new rule must not trigger
- one `regression` case for an adjacent branch, priority rule, exact-message rule, or tool-call rule

After Apply, verify the actual Updated Conversation and residual scan rather than assuming the plan succeeded.
