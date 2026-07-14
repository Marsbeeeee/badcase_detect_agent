#!/usr/bin/env python3
"""Run stage-gated, concurrent Stage 2 prompt experiments from a JSON manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = SCRIPT_DIR.parent / "assets" / "stage2-experiment-report-template.md"
STATE_SUFFIX = ".state.json"
SUPPORTED_PROMPT_EDITS = {"insert_before", "insert_after", "replace"}


@dataclass(frozen=True)
class TrialTask:
    case: dict[str, Any]
    experiment: dict[str, Any]
    trial: int
    prompt: str
    prompt_hash: str
    generation: dict[str, Any]
    cache_key: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json_or_jsonl(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Empty source file: {path}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        first = next((line for line in raw.splitlines() if line.strip()), "")
        parsed = json.loads(first)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return parsed


def normalized_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if isinstance(raw_tools, list):
        return raw_tools
    if isinstance(raw_tools, dict):
        return [item for item in raw_tools.values() if isinstance(item, dict)]
    return []


def source_context(case: dict[str, Any]) -> tuple[dict[str, Any], str, list[dict[str, str]], list[dict[str, Any]]]:
    source = Path(case["source"]).expanduser().resolve()
    row = load_json_or_jsonl(source)
    dialog = row.get("dialog")
    if not isinstance(dialog, list) or not dialog:
        raise ValueError(f"Source has no dialog list: {source}")
    target_turn = int(case["target_turn"])
    target_pos = next(
        (
            pos
            for pos, turn in enumerate(dialog)
            if isinstance(turn, dict)
            and int(turn.get("turn_index", pos)) == target_turn
            and turn.get("role") == "assistant"
        ),
        None,
    )
    if target_pos is None:
        raise ValueError(f"Assistant target turn {target_turn} not found in {source}")
    system = dialog[0]
    if system.get("role") != "system" or not isinstance(system.get("content"), str):
        raise ValueError(f"First dialog turn is not a text system prompt: {source}")
    messages = [
        {"role": str(turn.get("role")), "content": str(turn.get("content") or "")}
        for turn in dialog[:target_pos]
        if isinstance(turn, dict) and turn.get("role") in {"system", "user", "assistant", "tool"}
    ]
    return row, str(system["content"]), messages, normalized_tools(row.get("tools"))


def apply_prompt_edit(original: str, edit: dict[str, Any] | None) -> str:
    if not edit:
        return original
    edit_type = str(edit.get("type") or "")
    if edit_type not in SUPPORTED_PROMPT_EDITS:
        raise ValueError(f"Unsupported prompt edit type: {edit_type}")
    anchor = str(edit.get("anchor") or "")
    text = str(edit.get("text") or "")
    if not anchor or original.count(anchor) != 1:
        raise ValueError(f"Prompt edit anchor must occur exactly once: {anchor!r}")
    if edit_type == "insert_before":
        return original.replace(anchor, text + anchor, 1)
    if edit_type == "insert_after":
        return original.replace(anchor, anchor + text, 1)
    return original.replace(anchor, text, 1)


def completion_url(endpoint: str) -> str:
    clean = endpoint.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    if clean.endswith("/v1"):
        return clean + "/chat/completions"
    return clean + "/v1/chat/completions"


def response_content(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("content"), str):
        return payload["content"]
    choices = payload.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message.get("content"), str):
            return message["content"]
    return ""


def first_token_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    logprobs = payload.get("logprobs")
    if not isinstance(logprobs, list):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            logprobs = ((choices[0].get("logprobs") or {}).get("content") or [])
    if not isinstance(logprobs, list) or not logprobs or not isinstance(logprobs[0], dict):
        return None
    first = logprobs[0]
    return {
        "token": first.get("token"),
        "logprob": first.get("logprob"),
        "top_logprobs": [
            {"token": item.get("token"), "logprob": item.get("logprob")}
            for item in (first.get("top_logprobs") or [])[:3]
            if isinstance(item, dict)
        ],
    }


def matches_spec(content: str, spec: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if not spec:
        return False, ["no reproduction signature configured"]
    evidence: list[str] = []
    checks: list[bool] = []
    all_substrings = [str(item) for item in spec.get("all_substrings") or []]
    if all_substrings:
        ok = all(item in content for item in all_substrings)
        checks.append(ok)
        evidence.append(f"all_substrings={ok}")
    any_substrings = [str(item) for item in spec.get("any_substrings") or []]
    if any_substrings:
        ok = any(item in content for item in any_substrings)
        checks.append(ok)
        evidence.append(f"any_substrings={ok}")
    all_regex = [str(item) for item in spec.get("all_regex") or []]
    if all_regex:
        ok = all(re.search(pattern, content, re.IGNORECASE | re.MULTILINE) for pattern in all_regex)
        checks.append(ok)
        evidence.append(f"all_regex={ok}")
    any_regex = [str(item) for item in spec.get("any_regex") or []]
    if any_regex:
        ok = any(re.search(pattern, content, re.IGNORECASE | re.MULTILINE) for pattern in any_regex)
        checks.append(ok)
        evidence.append(f"any_regex={ok}")
    return bool(checks) and all(checks), evidence


def evaluate_acceptance(content: str, spec: dict[str, Any] | None) -> tuple[bool | None, list[str]]:
    if not spec:
        return None, ["manual review required: no deterministic acceptance criteria"]
    evidence: list[str] = []
    checks: list[bool] = []
    required_all = [str(item) for item in spec.get("required_all_substrings") or []]
    if required_all:
        ok = all(item in content for item in required_all)
        checks.append(ok)
        evidence.append(f"required_all_substrings={ok}")
    required_any = [str(item) for item in spec.get("required_any_substrings") or []]
    if required_any:
        ok = any(item in content for item in required_any)
        checks.append(ok)
        evidence.append(f"required_any_substrings={ok}")
    prohibited = [str(item) for item in spec.get("prohibited_substrings") or []]
    if prohibited:
        ok = not any(item in content for item in prohibited)
        checks.append(ok)
        evidence.append(f"prohibited_substrings_absent={ok}")
    required_regex = [str(item) for item in spec.get("required_all_regex") or []]
    if required_regex:
        ok = all(re.search(pattern, content, re.IGNORECASE | re.MULTILINE) for pattern in required_regex)
        checks.append(ok)
        evidence.append(f"required_all_regex={ok}")
    prohibited_regex = [str(item) for item in spec.get("prohibited_regex") or []]
    if prohibited_regex:
        ok = not any(re.search(pattern, content, re.IGNORECASE | re.MULTILINE) for pattern in prohibited_regex)
        checks.append(ok)
        evidence.append(f"prohibited_regex_absent={ok}")
    return (all(checks) if checks else None), evidence


def run_trial(task: TrialTask, timeout: float) -> dict[str, Any]:
    case = task.case
    _, _, base_messages, tools = source_context(case)
    messages = deepcopy(base_messages)
    messages[0]["content"] = task.prompt
    body: dict[str, Any] = {
        "model": case["model"],
        "provider": case.get("provider"),
        "messages": messages,
        "temperature": task.generation.get("temperature", 0.3),
        "top_p": task.generation.get("top_p", 0.95),
        "max_completion_tokens": task.generation.get("max_completion_tokens", 4096),
        "logprobs": task.generation.get("logprobs", True),
        "top_logprobs": task.generation.get("top_logprobs", 3),
    }
    if tools:
        body["tools"] = tools
    for optional in (
        "seed",
        "stop",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
    ):
        if optional in task.generation:
            body[optional] = task.generation[optional]
    if not body.get("provider"):
        body.pop("provider", None)
    started = time.perf_counter()
    try:
        response = requests.post(completion_url(str(case["endpoint"])), json=body, timeout=timeout)
        latency = round(time.perf_counter() - started, 3)
        response.raise_for_status()
        payload = response.json()
        content = response_content(payload)
        reproduced, reproduction_evidence = matches_spec(content, case.get("reproduction_signature"))
        accepted, acceptance_evidence = evaluate_acceptance(content, case.get("acceptance_criteria"))
        return {
            "cache_key": task.cache_key,
            "experiment_id": task.experiment["id"],
            "trial": task.trial,
            "status": "completed",
            "prompt_hash": task.prompt_hash,
            "generation": task.generation,
            "content": content,
            "reproduced": reproduced,
            "accepted": accepted,
            "reproduction_evidence": reproduction_evidence,
            "acceptance_evidence": acceptance_evidence,
            "first_token": first_token_summary(payload),
            "usage": payload.get("usage"),
            "request_id": response.headers.get("x-request-id") or response.headers.get("request-id"),
            "latency_s": latency,
            "error": None,
        }
    except Exception as exc:  # Preserve partial evidence and continue other cases.
        return {
            "cache_key": task.cache_key,
            "experiment_id": task.experiment["id"],
            "trial": task.trial,
            "status": "error",
            "prompt_hash": task.prompt_hash,
            "generation": task.generation,
            "content": "",
            "reproduced": None,
            "accepted": None,
            "latency_s": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def experiment_summary(experiment: dict[str, Any], trials: list[dict[str, Any]], baseline: dict[str, Any] | None) -> dict[str, Any]:
    completed = [item for item in trials if item.get("status") == "completed"]
    errors = [item for item in trials if item.get("status") == "error"]
    accepted = [item for item in completed if item.get("accepted") is True]
    reproduced = [item for item in completed if item.get("reproduced") is True]
    result = "inconclusive"
    if experiment["id"] != "B0" and completed and not errors and baseline:
        baseline_reproduced = int(baseline.get("reproduced_count") or 0)
        baseline_pass = int(baseline.get("accepted_count") or 0)
        if baseline_reproduced == 0:
            result = "inconclusive"
        elif len(reproduced) < baseline_reproduced and len(accepted) > baseline_pass:
            result = "supported"
        elif len(reproduced) > baseline_reproduced or len(accepted) < baseline_pass:
            result = "regressed"
        else:
            result = "not supported"
    return {
        "id": experiment["id"],
        "hypothesis": experiment.get("hypothesis") or ("原始 Prompt 基线" if experiment["id"] == "B0" else ""),
        "only_changed_variable": experiment.get("only_changed_variable") or ("无" if experiment["id"] == "B0" else ""),
        "planned_trials": int(experiment.get("trials") or 0),
        "completed_trials": len(completed),
        "error_trials": len(errors),
        "accepted_count": len(accepted),
        "reproduced_count": len(reproduced),
        "prompt_hash": (completed or errors or [{}])[0].get("prompt_hash"),
        "result": "baseline" if experiment["id"] == "B0" else result,
        "trials": sorted(trials, key=lambda item: int(item.get("trial") or 0)),
    }


def md_escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def representative(summary: dict[str, Any]) -> str:
    trial = next((item for item in summary.get("trials") or [] if item.get("content")), None)
    return str((trial or {}).get("content") or "")


def render_report(case_state: dict[str, Any], template_path: Path) -> str:
    case = case_state["case"]
    experiments = case_state.get("experiments") or []
    completed = [item for item in experiments if item.get("completed_trials")]
    baseline = next((item for item in experiments if item.get("id") == "B0"), None)
    incomplete = case_state.get("incomplete") or []
    scheduling_notes = case_state.get("scheduling_notes") or []
    all_planned_complete = bool(experiments) and all(
        item.get("completed_trials") == item.get("planned_trials") and not item.get("error_trials")
        for item in experiments
        if item.get("status") != "skipped"
    ) and not incomplete
    if all_planned_complete:
        status_text = "实验按计划完整执行。"
    else:
        status_text = "实验未完整执行；以下报告保留所有已完成证据，并列出错误、跳过项和阻塞原因。"
    if baseline and (
        baseline.get("completed_trials") < baseline.get("planned_trials")
        or baseline.get("error_trials")
    ):
        conclusion = "Baseline 未完整执行，后续变量实验已停止；当前结论为 inconclusive。"
    elif baseline and baseline.get("reproduced_count") == 0:
        conclusion = (
            "Baseline 未复现原 badcase，普通 prompt 变量已由实验闸门停止。"
            "当前结果只能用于描述已完成对照，不能声称某个变体修复了原问题。"
        )
    elif any(item.get("result") == "supported" for item in completed):
        supported = "、".join(item["id"] for item in completed if item.get("result") == "supported")
        conclusion = f"受控实验中 {supported} 得到支持；这仍是实验性证据，variant 未 Apply。"
    elif incomplete:
        conclusion = "实验存在执行错误或未完成项，当前结论为 inconclusive。"
    else:
        conclusion = "已完成实验未形成足够的因果改善证据，当前结论为 inconclusive。"

    object_lines = [
        f"- Case ID：`{case['id']}`",
        f"- 数据文件：`{case['source']}`",
        f"- 目标轮次：`{case['target_turn']}`",
        f"- 目标模型/服务：`{case['model']}`",
        f"- Provider：`{case.get('provider') or '未提供'}`",
        f"- Endpoint：`{case['endpoint']}`",
        f"- 实验日期：{date.today().isoformat()}",
        "- 实验方式：Stage 2 单变量实验；未修改源数据、正式 prompt 或 Apply 状态",
    ]
    generation = case.get("generation") or {}
    control_lines = [
        "- 所有实验只重跑目标 assistant turn；",
        "- 每个变体都从原始 prompt 独立构造，不叠加上一变体；",
        f"- 默认 trials：`{case.get('trials')}`；全局并发由运行 manifest 控制；",
        f"- 固定生成参数：`{json.dumps(generation, ensure_ascii=False, sort_keys=True)}`；",
        "- 同一判定规则用于 baseline 和所有变体；",
        "- baseline 未复现时，普通 prompt 变量自动跳过，仅运行显式允许的采样/环境对照。",
    ]
    table = [
        "| 实验 | 唯一变量 | 完成/计划 | Badcase 复现 | 验收通过 | 判定 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in experiments:
        table.append(
            "| {id} | {change} | {done}/{planned} | {repro}/{done} | {passed}/{done} | {result} |".format(
                id=md_escape(item.get("id")),
                change=md_escape(item.get("only_changed_variable")),
                done=item.get("completed_trials", 0),
                planned=item.get("planned_trials", 0),
                repro=item.get("reproduced_count", 0),
                passed=item.get("accepted_count", 0),
                result=md_escape(item.get("result") or item.get("status")),
            )
        )
    sections: list[str] = []
    for item in completed:
        output = representative(item)
        sections.extend(
            [
                f"## {item['id']}：{item.get('hypothesis') or '实验'}",
                "",
                f"- 唯一变量：{item.get('only_changed_variable') or '无'}",
                f"- Prompt hash：`{item.get('prompt_hash') or 'unknown'}`",
                f"- 完成情况：{item.get('completed_trials')}/{item.get('planned_trials')}；错误 {item.get('error_trials')} 次",
                f"- Badcase 复现：{item.get('reproduced_count')}/{item.get('completed_trials')}；验收通过：{item.get('accepted_count')}/{item.get('completed_trials')}",
                f"- 判定：`{item.get('result')}`",
                "",
                "代表输出：",
                "",
                f"> {output.replace(chr(10), ' ') if output else '无成功输出'}",
                "",
            ]
        )
    blockers: list[str] = []
    blockers.extend(f"- {item}" for item in incomplete)
    for item in experiments:
        for trial in item.get("trials") or []:
            if trial.get("error"):
                blockers.append(f"- {item['id']} trial {trial.get('trial')}：`{md_escape(trial['error'])}`")
    blocker_section = ""
    if blockers:
        blocker_section = "## 未完成实验与阻塞证据\n\n" + "\n".join(blockers) + "\n"
    notes = case.get("execution_equivalence_notes") or []
    equivalence = "\n".join(f"- {item}" for item in notes) if notes else "- 未记录额外执行等价性限制。"
    experiment_conclusion_lines = [
            f"- {status_text}",
            f"- {conclusion}",
    ]
    experiment_conclusion_lines.extend(f"- 调度说明：{item}" for item in scheduling_notes)
    experiment_conclusion_lines.extend(
        [
            "- Stage 2 experiment complete.",
            "- Variant not applied.",
        ]
    )
    experiment_conclusion = "\n".join(experiment_conclusion_lines)
    replacements = {
        "{{TITLE}}": str(case.get("title") or f"{case['id']} Turn {case['target_turn']} 控制变量实验报告"),
        "{{EXPERIMENT_OBJECT}}": "\n".join(object_lines),
        "{{CONCLUSION}}": conclusion,
        "{{EXPERIMENT_CONTROLS}}": "\n".join(control_lines),
        "{{RESULT_TABLE}}": "\n".join(table),
        "{{EXPERIMENT_SECTIONS}}": "\n".join(sections).rstrip(),
        "{{BLOCKER_SECTION}}": blocker_section.rstrip(),
        "{{EQUIVALENCE_LIMITS}}": equivalence,
        "{{EXPERIMENT_CONCLUSION}}": experiment_conclusion,
    }
    rendered = template_path.read_text(encoding="utf-8")
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered.rstrip() + "\n"


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {"version": 1, "trials": {}}

    def completed(self, key: str) -> dict[str, Any] | None:
        item = self.data.get("trials", {}).get(key)
        return item if isinstance(item, dict) and item.get("status") == "completed" else None

    def save_trial(self, item: dict[str, Any]) -> None:
        with self.lock:
            self.data.setdefault("trials", {})[item["cache_key"]] = item
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.path)


def build_task(case: dict[str, Any], experiment: dict[str, Any], trial: int) -> TrialTask:
    _, original, _, _ = source_context(case)
    prompt = apply_prompt_edit(original, experiment.get("prompt_edit"))
    if experiment.get("prompt_edit") and sha256_text(prompt) == sha256_text(original):
        raise ValueError(f"{case['id']} {experiment['id']} prompt hash did not change")
    generation = {**(case.get("generation") or {}), **(experiment.get("generation_override") or {})}
    fingerprint = {
        "case_id": case["id"],
        "source": str(Path(case["source"]).resolve()),
        "target_turn": case["target_turn"],
        "endpoint": case["endpoint"],
        "provider": case.get("provider"),
        "model": case["model"],
        "experiment_id": experiment["id"],
        "trial": trial,
        "prompt_hash": sha256_text(prompt),
        "generation": generation,
        "criteria": case.get("acceptance_criteria"),
        "reproduction": case.get("reproduction_signature"),
    }
    return TrialTask(case, experiment, trial, prompt, sha256_text(prompt), generation, stable_hash(fingerprint))


def run_task_group(
    tasks: list[TrialTask],
    *,
    executor: concurrent.futures.ThreadPoolExecutor,
    store: StateStore,
    timeout: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    futures: dict[concurrent.futures.Future[dict[str, Any]], TrialTask] = {}
    for task in tasks:
        cached = store.completed(task.cache_key)
        if cached:
            results.append(cached)
        else:
            futures[executor.submit(run_trial, task, timeout)] = task
    for future in concurrent.futures.as_completed(futures):
        item = future.result()
        store.save_trial(item)
        results.append(item)
    return results


def prepare_manifest(raw: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    defaults = raw.get("defaults") or {}
    cases = raw.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain a non-empty cases list")
    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_reports: set[str] = set()
    for index, source_case in enumerate(cases, start=1):
        case = {**defaults, **source_case}
        for required in ("id", "source", "target_turn", "report_path", "endpoint", "model"):
            if case.get(required) in (None, ""):
                raise ValueError(f"Case {index} missing required field: {required}")
        source = Path(str(case["source"])).expanduser()
        report = Path(str(case["report_path"])).expanduser()
        if not source.is_absolute():
            source = (manifest_path.parent / source).resolve()
        if not report.is_absolute():
            report = (manifest_path.parent / report).resolve()
        case["source"] = str(source)
        case["report_path"] = str(report)
        if str(case["id"]) in seen_ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        if str(report).lower() in seen_reports:
            raise ValueError(f"Duplicate report_path: {report}")
        seen_ids.add(str(case["id"]))
        seen_reports.add(str(report).lower())
        case["trials"] = int(case.get("trials") or 3)
        case["max_prompt_variants"] = int(case.get("max_prompt_variants") or 2)
        if case["trials"] < 1:
            raise ValueError(f"Case {case['id']} trials must be positive")
        if case["max_prompt_variants"] < 0:
            raise ValueError(f"Case {case['id']} max_prompt_variants cannot be negative")
        case["generation"] = {
            "temperature": 0.3,
            "top_p": 0.95,
            "max_completion_tokens": 4096,
            "logprobs": True,
            "top_logprobs": 3,
            **(case.get("generation") or {}),
        }
        variants = case.get("variants") or []
        if not isinstance(variants, list):
            raise ValueError(f"Case {case['id']} variants must be a list")
        for variant in variants:
            if not variant.get("id") or variant.get("id") == "B0":
                raise ValueError(f"Case {case['id']} has invalid variant id")
            variant["trials"] = int(variant.get("trials") or case["trials"])
            variant["priority"] = int(variant.get("priority") or 100)
        case["variants"] = sorted(variants, key=lambda item: (item["priority"], item["id"]))
        source_context(case)
        prepared.append(case)
    max_concurrency = int(raw.get("max_concurrency") or defaults.get("max_concurrency") or 6)
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    return {
        "cases": prepared,
        "max_concurrency": max_concurrency,
        "timeout_s": float(raw.get("timeout_s") or defaults.get("timeout_s") or 120),
        "template": str(Path(raw.get("template") or DEFAULT_TEMPLATE).expanduser().resolve()),
    }


def write_case_report(
    case: dict[str, Any],
    summaries: list[dict[str, Any]],
    incomplete: list[str],
    scheduling_notes: list[str],
    template: Path,
) -> None:
    report = Path(case["report_path"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_report(
            {
                "case": case,
                "experiments": summaries,
                "incomplete": incomplete,
                "scheduling_notes": scheduling_notes,
            },
            template,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--state", type=Path, default=None, help="Resume-state JSON path.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    config = prepare_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path)
    template = Path(config["template"])
    if not template.exists():
        raise SystemExit(f"Report template not found: {template}")
    if args.validate_only:
        print(json.dumps({"valid": True, "case_count": len(config["cases"])}, ensure_ascii=False))
        return 0
    state_path = (args.state or manifest_path.with_suffix(STATE_SUFFIX)).expanduser().resolve()
    store = StateStore(state_path)
    case_summaries: dict[str, list[dict[str, Any]]] = {case["id"]: [] for case in config["cases"]}
    case_incomplete: dict[str, list[str]] = {case["id"]: [] for case in config["cases"]}
    case_scheduling_notes: dict[str, list[str]] = {case["id"]: [] for case in config["cases"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=config["max_concurrency"]) as executor:
        baseline_tasks: list[TrialTask] = []
        baseline_defs: dict[str, dict[str, Any]] = {}
        for case in config["cases"]:
            baseline = {
                "id": "B0",
                "hypothesis": "原始 Prompt 基线",
                "only_changed_variable": "无",
                "trials": case["trials"],
            }
            baseline_defs[case["id"]] = baseline
            baseline_tasks.extend(build_task(case, baseline, trial) for trial in range(1, case["trials"] + 1))
        baseline_results = run_task_group(
            baseline_tasks, executor=executor, store=store, timeout=config["timeout_s"]
        )
        by_case: dict[str, list[dict[str, Any]]] = {case["id"]: [] for case in config["cases"]}
        cache_case = {task.cache_key: task.case["id"] for task in baseline_tasks}
        for item in baseline_results:
            by_case[cache_case[item["cache_key"]]].append(item)
        baselines: dict[str, dict[str, Any]] = {}
        for case in config["cases"]:
            summary = experiment_summary(baseline_defs[case["id"]], by_case[case["id"]], None)
            baselines[case["id"]] = summary
            case_summaries[case["id"]].append(summary)
            if summary["completed_trials"] < summary["planned_trials"] or summary["error_trials"]:
                case_incomplete[case["id"]].append("Baseline 未完整完成，后续变量实验已停止。")
            write_case_report(
                case,
                case_summaries[case["id"]],
                case_incomplete[case["id"]],
                case_scheduling_notes[case["id"]],
                template,
            )

        variant_queues: dict[str, list[dict[str, Any]]] = {}
        for case in config["cases"]:
            baseline = baselines[case["id"]]
            if baseline["completed_trials"] < baseline["planned_trials"] or baseline["error_trials"]:
                variant_queues[case["id"]] = []
                continue
            if baseline["reproduced_count"] == 0:
                selected = [v for v in case["variants"] if v.get("run_when_baseline_not_reproduced")]
                skipped = [v for v in case["variants"] if not v.get("run_when_baseline_not_reproduced")]
                case_scheduling_notes[case["id"]].extend(
                    f"{v['id']} 因 baseline 0/{baseline['completed_trials']} 未复现而由闸门跳过。"
                    for v in skipped
                )
            else:
                config_controls = [v for v in case["variants"] if not v.get("prompt_edit")]
                prompt_variants = [v for v in case["variants"] if v.get("prompt_edit")][
                    : case["max_prompt_variants"]
                ]
                selected = sorted(config_controls + prompt_variants, key=lambda item: (item["priority"], item["id"]))
                selected_ids = {item["id"] for item in selected}
                case_scheduling_notes[case["id"]].extend(
                    f"{v['id']} 超过 max_prompt_variants={case['max_prompt_variants']}，未调度。"
                    for v in case["variants"]
                    if v["id"] not in selected_ids
                )
            variant_queues[case["id"]] = selected
            write_case_report(
                case,
                case_summaries[case["id"]],
                case_incomplete[case["id"]],
                case_scheduling_notes[case["id"]],
                template,
            )

        while any(variant_queues.values()):
            wave: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for case in config["cases"]:
                queue = variant_queues[case["id"]]
                if queue:
                    wave.append((case, queue.pop(0)))
            tasks: list[TrialTask] = []
            task_case: dict[str, str] = {}
            task_experiment: dict[str, str] = {}
            executable_wave: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for case, experiment in wave:
                try:
                    experiment_tasks = [
                        build_task(case, experiment, trial)
                        for trial in range(1, int(experiment["trials"]) + 1)
                    ]
                except Exception as exc:
                    case_incomplete[case["id"]].append(
                        f"{experiment['id']} 构造失败：{type(exc).__name__}: {exc}"
                    )
                    variant_queues[case["id"]] = []
                    write_case_report(
                        case,
                        case_summaries[case["id"]],
                        case_incomplete[case["id"]],
                        case_scheduling_notes[case["id"]],
                        template,
                    )
                    continue
                executable_wave.append((case, experiment))
                for task in experiment_tasks:
                    tasks.append(task)
                    task_case[task.cache_key] = case["id"]
                    task_experiment[task.cache_key] = experiment["id"]
            if not tasks:
                continue
            results = run_task_group(tasks, executor=executor, store=store, timeout=config["timeout_s"])
            grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for item in results:
                key = (task_case[item["cache_key"]], task_experiment[item["cache_key"]])
                grouped.setdefault(key, []).append(item)
            for case, experiment in executable_wave:
                summary = experiment_summary(
                    experiment,
                    grouped.get((case["id"], experiment["id"]), []),
                    baselines[case["id"]],
                )
                case_summaries[case["id"]].append(summary)
                if summary["completed_trials"] < summary["planned_trials"] or summary["error_trials"]:
                    case_incomplete[case["id"]].append(
                        f"{experiment['id']} 未完整完成，后续变量实验已停止。"
                    )
                    variant_queues[case["id"]] = []
                elif summary["result"] == "supported" and case.get("stop_after_supported", True):
                    remaining = variant_queues[case["id"]]
                    derivative = [item for item in remaining if not item.get("independent", False)]
                    variant_queues[case["id"]] = [item for item in remaining if item.get("independent", False)]
                    case_scheduling_notes[case["id"]].extend(
                        f"{item['id']} 与已支持变量非独立，按提前停止规则未调度。"
                        for item in derivative
                    )
                write_case_report(
                    case,
                    case_summaries[case["id"]],
                    case_incomplete[case["id"]],
                    case_scheduling_notes[case["id"]],
                    template,
                )

    output = {
        "case_count": len(config["cases"]),
        "state": str(state_path),
        "reports": [case["report_path"] for case in config["cases"]],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
