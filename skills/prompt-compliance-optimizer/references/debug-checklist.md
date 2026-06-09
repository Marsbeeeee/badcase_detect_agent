# Debug Checklist

Use this reference when the Prompt Optimizer Agent UI or rerun behavior looks inconsistent.

## Trace And Conversation Alignment

- Check `st.session_state.bad_cases`.
- Check `st.session_state.bad_cases_conversation_view`: expected values are `Original`, `Updated`, or `None`.
- Check `st.session_state.trace_source_prompt_label`.
- Check `st.session_state.trace_list_version_index` when a Prompt Version button was clicked.
- Right-side turn highlighting should use the active trace source view. Historical version traces are read-only but should still drive the visible conversation view when possible.
- Automatic badcases include both `ai_judge` and deterministic `local_scan`; only `human` is human-marked.

## Apply Flow

- Single Apply enters `apply_bad_case_to_prompt`.
- Batch Apply enters `apply_selected_bad_cases_to_prompt`.
- `target_indices = rerun_targets_for_cases(...)`.
- `required_tools_by_turn = required_tools_for_cases(...)`.
- If the prompt changes:
  - add a prompt version,
  - run `rerun_with_prompt`,
  - generate/update conclusion.
- If the prompt does not change but `required_tools_by_turn` is non-empty:
  - do not call it a failed prompt edit,
  - run the target turn with the existing prompt,
  - describe it as a conversation-only rerun.
- If the prompt does not change and no required tool/action exists, keep the trace and report that no runnable change was found.

## Required Tool Rerun

- For missing required-tool-call traces, `required_tools_for_cases` should infer the exact tool from the current file's system prompt, badcase text, and tool definitions.
- `rerun_conversation` must handle consecutive assistant turns. A target assistant turn may not be immediately preceded by a user turn.
- If the model returns natural language despite required tool instructions, `_forced_required_tool_call` should replace the response with a function-call wrapper.
- If the latest user message is only a name or short gate response, build the forced tool query from recent user messages plus the old target assistant response.

Expected wrapper:

```text
<function-call><required_tool_name>:{"query":"..."}</function-call>
```

## Updated Conversation

- `build_rerun_interactions` replaces target assistant turns by index.
- Function-call wrappers are parsed into structured `tool_calls`.
- Each parsed tool call inserts a placeholder `Tool` turn.
- Placeholder tool output should say `status: not_executed`; do not fabricate real search results.

## Conclusion

- If `before_prompt == optimized_prompt`, conclusion paragraph 1 should say no new prompt version was needed and the conversation turn was rerun with the existing prompt.
- Do not let conclusion imply the experiment was skipped when the target response changed.
- Use target-turn behavior, tool-call state, residual badcase scan, and available logprob diagnostics as evidence.

## Logs

- `logs/company_api_preflight_system_prompt.log`: confirm `targeted_rerun_preflight`, expected prompt hash, and outgoing prompt.
- `logs/company_api_requests.jsonl`: confirm purpose, tools, tool_choice, model/provider, and target request.
- `logs/company_api_system_prompts.log`: confirm latest outgoing prompt text.
- `logs/company_api_logprobs.json`: inspect token diagnostics when available.

## Common Symptoms

- Left trace card exists but AI count is zero: deterministic `local_scan` is not being counted as automatic.
- Right conversation does not show analysis: active trace source view and conversation view are misaligned.
- Apply says prompt unchanged but conversation changed: this is valid conversation-only rerun; improve wording, not behavior.
- Apply says rerun skipped even though a required tool exists: unchanged-prompt rerun branch is missing or not reached.
- Turn remains natural language after required tool use: check `required_tools_by_turn`, `tool_choice`, and forced tool-call fallback.
