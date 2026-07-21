# Controlled Badcase Experiments

Use this protocol only when the user explicitly requests Stage 2 after observational analysis identifies plausible causes for a known target-turn badcase.

## Contents

- Goal and Preconditions
- Baseline First
- Hypothesis Design
- One Variable Per Variant
- Target-Turn Rerun
- Fixed Acceptance Criteria
- Result Classification
- Sequential Order
- Efficient Multi-Case Execution
- Runner Manifest
- Report Format

## Goal

Determine which suspected cause actually changes the target model's behavior. Change one variable at a time, rerun the same target turn, and compare against an unchanged baseline.

Experiments are diagnostic only. Never modify the source dataset, active Working prompt, prompt-version history, approved-case state, or production configuration.

Read the Stage 1 Markdown report when available. Do not repeat the full Stage 1 narrative; use its hypotheses as experiment inputs.

## Preconditions

Before experimenting, require:

- the original system prompt;
- exact conversation history through the target user turn;
- target model/checkpoint and callable backend;
- original tool definitions and message format;
- original generation parameters when available;
- a concrete compliance criterion derived from the prompt and key guide.

If the exact target model or backend is unavailable, try the configured repository rerun backend only when it resolves to the same target model. Do not silently substitute another model and call it causal validation. Report `experiment_blocked_target_model_unavailable` when equivalence cannot be established.

## Baseline First

Rerun the unchanged target turn before testing variants.

Keep constant:

- system prompt;
- conversation history;
- tool definitions;
- model/checkpoint and endpoint;
- temperature, top-p, top-k, max tokens, chat template, and stop settings;
- judge and deterministic acceptance criteria.

Use the original seed when supported. If no seed is supported, run equal-size baseline and variant samples.

Default to three baseline trials and three trials per variant when backend cost and latency are reasonable. A single trial is allowed only when constrained; label it `single-sample evidence` and do not claim robust causality.

If the baseline cannot reproduce the badcase:

- record the reproduction rate;
- test a temperature/seed hypothesis before prompt variants when sampling evidence supports it;
- stop ordinary prompt variants immediately; run only sampling or execution-equivalence controls explicitly marked for non-reproducing baselines;
- compare rates rather than one old output versus one new output;
- do not claim a prompt change fixed a non-reproducing failure.

## Hypothesis Design

Create hypotheses only from observed evidence. Common generic classes include:

| Observed pattern | Hypothesis | Single experimental variable |
|---|---|---|
| Completed gate is reopened | State retention weakness | Add one monotonic-state rule near the gate |
| Two actors use the same verification term | Direction/actor confusion | Add one actor-disambiguation rule |
| FAQ and workflow compete | Rule precedence ambiguity | Add one explicit precedence resolver |
| Wrong exact close template is reproduced | Terminal precondition weakness | Add explicit preconditions to that terminal branch |
| First-token alternatives are close | Sampling sensitivity | Change temperature to 0; leave prompt unchanged |
| Required action is omitted | Action/tool timing ambiguity | Clarify only the timing/required-action rule |

Do not test every generic class. Test only causes supported by the current artifact.

## One Variable Per Variant

Start every variant from the original prompt/config, not from the previous experiment.

Valid examples:

- Experiment A changes only a state-monotonicity sentence.
- Experiment B changes only actor/direction wording.
- Experiment C changes only terminal-branch preconditions.
- Experiment D changes only temperature.

Invalid examples:

- adding state, precedence, examples, and terminal guards in one prompt;
- changing prompt and temperature together;
- changing model while testing a prompt hypothesis;
- comparing a target-turn rerun against a full regenerated conversation;
- using a different judge for variants.

Summarize each delta with an exact unified diff or old/new excerpt. Keep the rest byte-identical where possible and verify prompt hashes.

## Target-Turn Rerun

Use the repository's existing target-turn rerun surface. Construct an isolated conversation object with the experiment prompt and rerun only the target assistant turn.

Do not execute real business tools. Preserve the existing placeholder tool-turn behavior when the correct response requires a tool call.

Tag requests and artifacts as `experiment_only` when the execution surface supports metadata. Do not record them as Apply or residual-scan rounds.

For each trial, capture:

- experiment id and hypothesis id;
- original and variant prompt hash;
- exact changed variable;
- model, provider, endpoint, request id, and seed if available;
- generation config;
- response text and tool-call state;
- logprob summary when available;
- compliance verdict and violated/satisfied rule evidence;
- latency or backend error.

## Fixed Acceptance Criteria

Define criteria before seeing variant outputs. Use the same criteria for baseline and every experiment.

Criteria should check the actual failure surface, such as:

- correct active workflow branch;
- no reopening of a completed gate;
- required exact text present;
- prohibited terminal template absent;
- required payment/tool follow-up present;
- no new policy violation.

Do not use output similarity or PPL as the main pass criterion.

## Result Classification

Compare compliance rates using equal trial counts.

- `supported`: the isolated change improves the target failure rate and introduces no new compliance violation; strongest when all variant trials pass and baseline trials reproduce the failure.
- `not supported`: the isolated change produces no meaningful improvement or changes irrelevant wording only.
- `inconclusive`: backend mismatch, baseline non-reproduction, insufficient trials, high sampling variance, judge disagreement, or execution failure prevents attribution.
- `regressed`: the variant worsens compliance or creates a new violation.

Do not use `fixed` for an experiment. Use `improved under isolated experiment`. A production fix requires approval, Apply, and residual verification.

## Sequential Order

Use the least invasive and most directly evidenced experiment first:

1. sampling-only control, but only when logprobs/config suggest sampling sensitivity;
2. state or actor disambiguation;
3. explicit rule precedence;
4. terminal-branch preconditions;
5. a combined variant only after individual experiments identify supported variables.

The combined variant is confirmatory, not a single-variable causal test. Label it separately and include only individually supported changes.

Stop early when one hypothesis is strongly supported and remaining hypotheses are derivative of the same mechanism. Continue when hypotheses are genuinely independent.

## Efficient Multi-Case Execution

Use `scripts/run_stage2_experiments.py` whenever there is more than one case, more than one variant, or repeated trials. Do not hand-write serial request loops.

Apply these scheduling rules:

1. Run all cases' B0 trials as the first wave.
2. Run trials within a wave concurrently under one global semaphore. Default to `max_concurrency=6`; lower it after rate-limit or capacity errors.
3. Do not schedule ordinary prompt variants until that case's complete baseline is available.
4. When B0 reproduces 0 times, stop prompt variants and run only variants with `run_when_baseline_not_reproduced=true`.
5. When B0 reproduces, test at most the two highest-priority prompt variants by default. Include additional variants only when their hypotheses are genuinely independent or the user requests them.
6. After a supported result, skip later derivative variants; continue explicitly independent variants.
7. Use deterministic acceptance criteria first. Call an LLM judge only for trials whose branch or compliance result cannot be decided from prompt-derived rules.
8. Save each completed trial immediately. Resume by the case/model/endpoint/prompt-hash/config/experiment/trial fingerprint; never count the same cached trial as a newly sampled run.
9. Discover endpoint/model capabilities once per endpoint-provider-model tuple, not once per trial.

Do not parallelize dependent full-conversation turns. Stage 2 target-turn trials and variants are independent after baseline gating and are safe to schedule concurrently.

## Runner Manifest

Create one UTF-8 JSON manifest and run:

```powershell
python C:\Users\ZLSHLT2604010\.codex\skills\prompt-compliance-optimizer\scripts\run_stage2_experiments.py <manifest.json>
```

Use this minimal shape:

```json
{
  "max_concurrency": 6,
  "timeout_s": 120,
  "defaults": {
    "trials": 3,
    "max_prompt_variants": 2,
    "generation": {
      "temperature": 0.3,
      "top_p": 0.95,
      "max_completion_tokens": 4096
    }
  },
  "cases": [
    {
      "id": "case-id",
      "source": "input.jsonl",
      "target_turn": 10,
      "report_path": "badcase分析报告/case-id_turn_10_model_experiment.md",
      "endpoint": "http://host:9898",
      "provider": "openai_api_like",
      "model": "serve-name",
      "reproduction_signature": {
        "any_substrings": ["the known wrong terminal template"]
      },
      "acceptance_criteria": {
        "prohibited_substrings": ["the known wrong terminal template"],
        "required_all_regex": ["prompt-derived required follow-up"]
      },
      "variants": [
        {
          "id": "E1",
          "priority": 1,
          "hypothesis": "state retention weakness",
          "only_changed_variable": "one monotonic-state rule",
          "prompt_edit": {
            "type": "insert_before",
            "anchor": "unique prompt section heading",
            "text": "one isolated rule\n\n"
          }
        },
        {
          "id": "E2",
          "priority": 2,
          "hypothesis": "sampling sensitivity",
          "only_changed_variable": "temperature 0.3 to 0",
          "generation_override": {"temperature": 0},
          "run_when_baseline_not_reproduced": true,
          "independent": true
        }
      ]
    }
  ]
}
```

Require every prompt-edit anchor to occur exactly once. The runner verifies prompt hashes before dispatch. Use `--validate-only` before a costly batch. The default sibling `<manifest-stem>.state.json` is the resumable machine-readable artifact; pass `--state` only when a different stable path is required.

## Report Format

Use a compact table:

| Experiment | Hypothesis | Only changed variable | Trials | Pass rate | Result |
|---|---|---|---:|---:|---|

Then provide per-experiment evidence:

1. exact delta summary;
2. controls held constant;
3. representative output difference;
4. judge/rule evidence;
5. logprob interpretation, if available: align the baseline and variant at the
   same semantic decision word or phrase, then state whether the desired
   continuation's confidence or recorded rank increased, whether the wrong
   continuation's margin narrowed or reversed, and whether the model changed
   from confident-wrong to uncertain/correct; omit dimensions not supported by
   the recorded top-k;
6. classification and confidence limitation.

Add `Logprob 前后变化结论` before `实验结论`. Use natural language and the
human-readable decision word or phrase. Do not compare unaligned tokenizer
positions, different models/checkpoints, or materially different decoding
settings; record the comparison as unavailable instead.

Finish with `实验结论`:

- rank supported causes by evidence strength;
- list rejected causes;
- list blocked or inconclusive causes and why;
- state whether any combined confirmation was run;
- do not append speculative improvement recommendations.

Always write a separate stable Stage 2 Markdown report using the user-specified folder or the source-adjacent `badcase分析报告` default. End with `Stage 2 experiment complete` and `variant not applied`. Preserve machine-readable details in a sibling JSON only if the user requests artifacts or the workflow already produces them. Do not paste full token arrays into the report.

Use `assets/stage2-experiment-report-template.md` as the fixed report structure. When all scheduled experiments complete, populate every section and every experiment result. When backend errors, invalid prompt anchors, incomplete baseline trials, or other blockers prevent a full run, still write the stable report: include all completed experiment sections, then list each failed/skipped experiment and exact blocker under `未完成实验与阻塞证据`. Do not invent missing results or replace the report with a chat-only error.
