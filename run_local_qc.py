#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch dialog QC via local OpenAI-compatible chat API (方案 A: 全量 LLM 质检)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    raise

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

WORKSPACE = Path("/mnt/zfs02/danny.li/workspace")
DEFAULT_INPUT = WORKSPACE / "cursor_test/check_dataset/daily_conversation_acceptance"
DEFAULT_OUTPUT = WORKSPACE / "cursor_test/check_dataset/check_result"
DEFAULT_RULES = WORKSPACE / "cursor_test/check_dataset/check_prompt.txt"
DEFAULT_URL = "http://192.168.101.15:9898"
DEFAULT_PROVIDER = "openai_api_like"

TREE_ID_RE = re.compile(r"(laep_dialogue_\d+-leaf-t\d+)")

JSON_OUTPUT_INSTRUCTION = """
---
【输出要求 — 必须严格遵守】
只输出一个 JSON 对象，不要 markdown 标题、不要代码块围栏、不要任何前后说明文字。
JSON 字段：
- id: string，与输入样本 id 一致
- tree_id: string，从 id 提取，如 "laep_dialogue_32049 / t14"
- verdict: "通过" | "基本通过" | "不通过"
- score: 0-100 整数
- summary: string，2-3 句话概括（流程跳转、语言风格、信息一致性、function 调用）
- issues: array，每项含 type, turn (int), description, fix；无问题则为 []

issue.type 仅用：场景角色 / 流程跳转 / 固定话术 / 格式轮次 / 语言风格 / Function
每个 issue 必须定位到具体 turn 号，description 须引用该 turn 的证据。
"""


def extract_annotator(fpath: Path) -> str:
    stem = fpath.stem
    for prefix in (
        "Daily_Conversation_Construction_filtered_",
        "reviewer_dialogs.",
    ):
        if stem.startswith(prefix):
            return stem.replace(prefix, "")
    return stem


def list_input_files(input_dir: Path, glob_pattern: str = "") -> List[Path]:
    if glob_pattern:
        return sorted(input_dir.glob(glob_pattern))
    files = sorted(input_dir.glob("Daily_Conversation_Construction_filtered_*.jsonl"))
    if not files:
        files = sorted(input_dir.glob("reviewer_dialogs.*"))
    return files


def parse_tree_id(sample_id: str) -> str:
    m = TREE_ID_RE.search(sample_id or "")
    if m:
        parts = m.group(1).split("-leaf-")
        return f"{parts[0]} / {parts[1]}"
    return sample_id or "unknown"


def compact_sample(raw: dict) -> dict:
    dialog = []
    for turn in raw.get("dialog", []):
        dialog.append({
            "turn_index": turn.get("turn_index"),
            "role": turn.get("role"),
            "content": turn.get("content", ""),
        })
    tools = raw.get("tools") or {}
    if isinstance(tools, dict) and not tools:
        tools = None
    return {
        "id": raw.get("id"),
        "type": raw.get("type"),
        "meta": raw.get("meta"),
        "tools": tools,
        "dialog": dialog,
    }


def extract_content(response: dict) -> str:
    if isinstance(response, str):
        return response
    if "content" in response and isinstance(response["content"], str):
        return response["content"]
    choices = response.get("choices")
    if choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            if "message" in c0 and isinstance(c0["message"], dict):
                return c0["message"].get("content") or ""
            if "text" in c0:
                return c0["text"] or ""
    return ""


def strip_thinking(text: str) -> str:
    """Remove model thinking blocks before JSON extraction."""
    text = text or ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # voyager / gemma channel format: <|channel>thought ... <channel|>answer
    if "<|channel>thought" in text:
        text = re.sub(
            r"<\|channel>thought[\s\S]*?(?:<channel\|>|<\|channel\|>)",
            "",
            text,
        )
    return text.strip()


def parse_model_json(text: str) -> dict:
    text = strip_thinking(text or "").strip()
    if not text:
        raise ValueError("empty model response")
    # strip markdown fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        end = text.rfind("}")
        if end <= start:
            raise ValueError(f"no JSON object in response: {text[:200]}")
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("parsed value is not object")
    return obj


def normalize_result(obj: dict, sample_id: str) -> dict:
    verdict = obj.get("verdict", "不通过")
    if verdict not in ("通过", "基本通过", "不通过"):
        verdict = "不通过"
    score = obj.get("score", 0)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))
    issues = obj.get("issues") or []
    if not isinstance(issues, list):
        issues = []
    clean_issues = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        clean_issues.append({
            "type": str(it.get("type", "其他")),
            "turn": it.get("turn", -1),
            "description": str(it.get("description", "")),
            "fix": str(it.get("fix", "")),
        })
    return {
        "id": obj.get("id") or sample_id,
        "tree_id": obj.get("tree_id") or parse_tree_id(sample_id),
        "verdict": verdict,
        "score": score,
        "summary": str(obj.get("summary", "")),
        "issues": clean_issues,
    }


class QCClient:
    def __init__(
        self,
        url: str,
        model: str,
        provider: str = DEFAULT_PROVIDER,
        temperature: float = 0.1,
        max_tokens: int = 20480,
        timeout: float = 300.0,
        reasoning_effort: Optional[str] = None,
    ):
        self.api_url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    def qc_one(self, rules_text: str, sample: dict) -> dict:
        user_payload = json.dumps(compact_sample(sample), ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": rules_text + JSON_OUTPUT_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    "请对以下 JSONL 样本做严格质检，依据 system 中的规则与评分标准。\n\n"
                    + user_payload
                ),
            },
        ]
        payload = {
            "messages": messages,
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
            if self.provider == "openai_api_like":
                payload["chat_template_kwargs"] = {"enable_thinking": True}
        resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        if isinstance(data, dict) and ("error" in data or "detail" in data):
            raise RuntimeError(str(data))
        content = extract_content(data)
        parsed = parse_model_json(content)
        return normalize_result(parsed, sample.get("id", ""))


def cache_path(cache_dir: Path, annotator: str, sample_id: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", sample_id)
    return cache_dir / annotator / f"{safe}.json"


def load_cache(path: Path) -> Optional[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_cache(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def format_sample_md(result: dict, detail: bool = True) -> str:
    lines = [
        f"### 样本 ID：{result['id']}",
        f"- Tree ID：{result['tree_id']}",
        f"- 判定：{result['verdict']}",
        f"- 总分：{result['score']} / 100",
        f"- 整体评价：{result['summary'] or '（无）'}",
    ]
    issues = result.get("issues") or []
    if issues and detail:
        lines.append("- 问题明细：")
        for it in issues:
            lines.append(
                f"  - [{it['type']}] turn {it['turn']}：{it['description']}"
            )
            if it.get("fix"):
                lines.append(f"    修改建议：{it['fix']}")
    elif detail:
        lines.append("- 问题明细：无")
    lines.append("")
    return "\n".join(lines)


def write_annotator_report(
    annotator: str,
    results: List[dict],
    output_dir: Path,
    source_name: str,
) -> None:
    pass_n = sum(1 for r in results if r["verdict"] == "通过")
    basic_n = sum(1 for r in results if r["verdict"] == "基本通过")
    fail_n = sum(1 for r in results if r["verdict"] == "不通过")
    problem_n = sum(1 for r in results if r.get("issues"))

    lines = [
        f"# 质检报告 - {annotator}",
        "",
        f"数据文件：`{source_name}`",
        "",
        "## 汇总",
        f"- 总样本数：{len(results)}",
        f"- 通过：{pass_n}",
        f"- 基本通过：{basic_n}",
        f"- 不通过：{fail_n}",
        f"- 有问题样本：{problem_n}",
        "",
    ]

    problem_results = [r for r in results if r.get("issues")]
    if problem_results:
        lines.append("## 有问题样本详情")
        lines.append("")
        for r in problem_results:
            lines.extend(format_sample_md(r, detail=True).splitlines())
            lines.append("")

    lines.append("## 全部样本一览")
    lines.append("")
    lines.append("| 样本 ID | Tree ID | 判定 | 总分 | 一句话评价 |")
    lines.append("|---------|---------|------|------|------------|")
    for r in results:
        summary = (r.get("summary") or "").replace("|", "\\|").replace("\n", " ")
        if len(summary) > 80:
            summary = summary[:77] + "..."
        lines.append(
            f"| {r['id']} | {r['tree_id']} | {r['verdict']} | {r['score']} | {summary} |"
        )
    lines.append("")

    out = output_dir / f"qc_{annotator}.md"
    out.write_text("\n".join(lines), encoding="utf-8")


def write_summary(all_stats: List[dict], output_dir: Path) -> None:
    lines = [
        "# 质检总览（本地模型批处理）",
        "",
        "| 标注人 | 样本数 | 通过 | 基本通过 | 不通过 | 通过率 | 基本+通过 | 有问题 | 系统错误 |",
        "|--------|--------|------|----------|--------|--------|-----------|--------|----------|",
    ]
    tot = {
        "total": 0, "pass": 0, "basic": 0, "fail": 0,
        "problem": 0, "errors": 0,
    }
    for s in all_stats:
        total = s["total"]
        pass_rate = f"{s['pass'] / total * 100:.1f}%" if total else "0%"
        ok_rate = f"{(s['pass'] + s['basic']) / total * 100:.1f}%" if total else "0%"
        lines.append(
            f"| {s['annotator']} | {total} | {s['pass']} | {s['basic']} | {s['fail']} "
            f"| {pass_rate} | {ok_rate} | {s['problem']} | {s.get('errors', 0)} |"
        )
        for k in tot:
            tot[k] += s[k]
    total = tot["total"]
    pass_rate = f"{tot['pass'] / total * 100:.1f}%" if total else "0%"
    ok_rate = f"{(tot['pass'] + tot['basic']) / total * 100:.1f}%" if total else "0%"
    lines.append(
        f"| **合计** | {total} | {tot['pass']} | {tot['basic']} | {tot['fail']} "
        f"| {pass_rate} | {ok_rate} | {tot['problem']} | {tot['errors']} |"
    )
    lines.append("")
    lines.append("由 `dialog-qc-local/run_local_qc.py` 生成。")
    lines.append("")
    (output_dir / "qc_summary.md").write_text("\n".join(lines), encoding="utf-8")


def process_file(
    fpath: Path,
    client: QCClient,
    rules_text: str,
    cache_dir: Path,
    force: bool,
) -> Tuple[str, List[dict], dict]:
    annotator = extract_annotator(fpath)
    results: List[dict] = []
    errors: List[str] = []

    with open(fpath, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    for sample in samples:
        sid = sample.get("id", "unknown")
        cpath = cache_path(cache_dir, annotator, sid)
        if not force:
            cached = load_cache(cpath)
            if cached:
                results.append(cached)
                continue
        try:
            result = client.qc_one(rules_text, sample)
            save_cache(cpath, result)
            results.append(result)
        except Exception as e:
            errors.append(f"{sid}: {e}")
            err_result = {
                "id": sid,
                "tree_id": parse_tree_id(sid),
                "verdict": "不通过",
                "score": 0,
                "summary": f"质检调用失败: {e}",
                "issues": [{
                    "type": "系统",
                    "turn": -1,
                    "description": str(e),
                    "fix": "检查模型服务、max_tokens 或重试 --force",
                }],
                "error": True,
            }
            results.append(err_result)

    stats = {
        "annotator": annotator,
        "total": len(results),
        "pass": sum(1 for r in results if r["verdict"] == "通过"),
        "basic": sum(1 for r in results if r["verdict"] == "基本通过"),
        "fail": sum(1 for r in results if r["verdict"] == "不通过"),
        "problem": sum(1 for r in results if r.get("issues")),
        "errors": len(errors),
    }
    return annotator, results, stats


def process_file_parallel_workers(
    fpath: Path,
    client: QCClient,
    rules_text: str,
    cache_dir: Path,
    force: bool,
    workers: int,
    retry_errors: bool = False,
) -> Tuple[str, List[dict], dict]:
    annotator = extract_annotator(fpath)
    with open(fpath, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    todo: List[Tuple[dict, Path]] = []
    results_map: Dict[str, dict] = {}

    for sample in samples:
        sid = sample.get("id", "unknown")
        cpath = cache_path(cache_dir, annotator, sid)
        if not force:
            cached = load_cache(cpath)
            if cached:
                if retry_errors and cached.get("error"):
                    pass
                else:
                    results_map[sid] = cached
                    continue
        todo.append((sample, cpath))

    def qc_sample(item: Tuple[dict, Path]) -> Tuple[str, dict]:
        sample, cpath = item
        sid = sample.get("id", "unknown")
        try:
            result = client.qc_one(rules_text, sample)
            save_cache(cpath, result)
            return sid, result
        except Exception as e:
            return sid, {
                "id": sid,
                "tree_id": parse_tree_id(sid),
                "verdict": "不通过",
                "score": 0,
                "summary": f"质检调用失败: {e}",
                "issues": [{
                    "type": "系统",
                    "turn": -1,
                    "description": str(e),
                    "fix": "检查模型服务、max_tokens 或重试 --force",
                }],
                "error": True,
            }

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(qc_sample, item) for item in todo]
            for fut in as_completed(futures):
                sid, result = fut.result()
                results_map[sid] = result

    # preserve file order
    results = [results_map[s.get("id", "unknown")] for s in samples]
    stats = {
        "annotator": annotator,
        "total": len(results),
        "pass": sum(1 for r in results if r["verdict"] == "通过"),
        "basic": sum(1 for r in results if r["verdict"] == "基本通过"),
        "fail": sum(1 for r in results if r["verdict"] == "不通过"),
        "problem": sum(1 for r in results if r.get("issues")),
        "errors": sum(1 for r in results if r.get("error")),
    }
    return annotator, results, stats


def main():
    parser = argparse.ArgumentParser(description="Local model batch dialog QC")
    parser.add_argument("-m", "--model", required=True, help="Model name served locally")
    parser.add_argument("-u", "--url", default=DEFAULT_URL, help="Chat API base URL")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Provider for chat-demo")
    parser.add_argument("-i", "--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--input-glob",
        default="",
        help="Input file glob under input-dir (default: auto-detect)",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Default: <output-dir>/.qc_cache")
    parser.add_argument("--annotators", type=str, default="", help="Comma-separated, e.g. 05,annie")
    parser.add_argument("--workers", type=int, default=4, help="Parallel samples per file")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort (open_router or thinking models)",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-run all")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-run only cached failures / missing cache; keep successful cache",
    )
    args = parser.parse_args()

    if not args.rules.is_file():
        print(f"Rules file not found: {args.rules}", file=sys.stderr)
        sys.exit(1)
    if not args.input_dir.is_dir():
        print(f"Input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    rules_text = args.rules.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / ".qc_cache")

    files = list_input_files(args.input_dir, args.input_glob)
    if args.annotators.strip():
        wanted = {x.strip() for x in args.annotators.split(",") if x.strip()}
        files = [f for f in files if extract_annotator(f) in wanted]
    if not files:
        print("No JSONL files to process.", file=sys.stderr)
        sys.exit(1)

    client = QCClient(
        url=args.url,
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
    )

    all_stats: List[dict] = []
    t0 = time.time()

    for fpath in files:
        annotator, results, stats = process_file_parallel_workers(
            fpath,
            client,
            rules_text,
            cache_dir,
            args.force,
            args.workers,
            retry_errors=args.retry_errors,
        )
        write_annotator_report(
            annotator, results, args.output_dir, fpath.name
        )
        all_stats.append(stats)
        msg = (
            f"{annotator}: total={stats['total']} pass={stats['pass']} "
            f"basic={stats['basic']} fail={stats['fail']} problem={stats['problem']} "
            f"errors={stats['errors']}"
        )
        if HAS_RICH:
            console.print(f"[green]✓[/] {msg}")
        else:
            print(msg)

    write_summary(all_stats, args.output_dir)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. Reports -> {args.output_dir}")


if __name__ == "__main__":
    main()
