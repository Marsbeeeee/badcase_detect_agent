#!/usr/bin/env python3
"""Extract callid samples per rm ai label from each Excel workbook.

Sampling quota is per source workbook:
- labels A/B/C: 30 rows each
- all other labels: 50 rows each
"""

from __future__ import annotations

import csv
import json
import posixpath
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


INPUT_FILES = [
    "/Users/zlshlt2501003/Downloads/2026-06-03-call-records.xlsx",
    "/Users/zlshlt2501003/Downloads/2026-06-03-call-records (1).xlsx",
    "/Users/zlshlt2501003/Downloads/2026-06-03-call-records (2).xlsx",
    "/Users/zlshlt2501003/Downloads/2026-06-03-call-records (3).xlsx",
    "/Users/zlshlt2501003/Downloads/2026-06-03-call-records (4).xlsx",
    "/Users/zlshlt2501003/Downloads/2026-06-03-call-records (5).xlsx",
    "/Users/zlshlt2501003/Downloads/2026-06-03-call-records (6).xlsx",
    "/Users/zlshlt2501003/Downloads/2026-06-03-call-records (7).xlsx",
    "/Users/zlshlt2501003/Downloads/cl_20260501-20260603_bot.xlsx",
]

OUTPUT_DIR = Path("outputs/rm_ai_label_samples")
FIELDNAMES = ["callid", "rm ai label"]
SPECIAL_LABEL_LIMITS = {"A": 30, "B": 30, "C": 30}
DEFAULT_LABEL_LIMIT = 50


def sample_limit(label: str) -> int:
    return SPECIAL_LABEL_LIMITS.get(label, DEFAULT_LABEL_LIMIT)


def normalize_header(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    if not letters:
        raise ValueError(f"Cell reference lacks column letters: {cell_ref}")

    idx = 0
    for char in letters.group(0):
        idx = idx * 26 + ord(char) - ord("A") + 1
    return idx - 1


def collect_text(element: ET.Element) -> str:
    parts = []
    for text_node in element.findall(f".//{{{MAIN_NS}}}t"):
        if text_node.text:
            parts.append(text_node.text)
    return "".join(parts)


def load_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [collect_text(si) for si in root.findall(f"{{{MAIN_NS}}}si")]


def load_sheet_paths(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{PKG_REL_NS}}}Relationship")
    }

    sheets = []
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get(f"{{{REL_NS}}}id")
        if not rel_id or rel_id not in rel_targets:
            continue

        target = rel_targets[rel_id]
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = posixpath.normpath(posixpath.join("xl", target))
        sheets.append((name, path))
    return sheets


def cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        return collect_text(cell)

    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return ""

    raw_value = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except (ValueError, IndexError):
            return raw_value
    return raw_value


def iter_sheet_rows(
    zf: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]
) -> Iterable[Dict[int, str]]:
    root = ET.fromstring(zf.read(sheet_path))
    for row in root.findall(f".//{{{MAIN_NS}}}row"):
        values: Dict[int, str] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            ref = cell.attrib.get("r")
            if not ref:
                continue
            values[column_index(ref)] = cell_value(cell, shared_strings)
        yield values


def find_columns(rows: List[Dict[int, str]]) -> Optional[Tuple[int, int, int, str]]:
    label_candidates = {
        "rmailabel": "rm ai label",
        "rmflowlab": "RMFlowLab",
        "intention": "Intention",
    }

    for row_idx, row in enumerate(rows):
        normalized = {idx: normalize_header(str(value).strip()) for idx, value in row.items()}
        callid_cols = [idx for idx, value in normalized.items() if value == "callid"]
        label_cols = [
            (idx, label_candidates[value])
            for idx, value in normalized.items()
            if value in label_candidates
        ]
        if callid_cols and label_cols:
            label_col, label_source = label_cols[0]
            return row_idx, callid_cols[0], label_col, label_source
    return None


def extract_samples(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    samples_by_label: Dict[str, List[Dict[str, str]]] = {}
    seen_by_label: Dict[str, set[str]] = {}
    sheets_seen = []
    missing_columns = []
    rows_scanned = 0
    label_sources = []

    with zipfile.ZipFile(path) as zf:
        shared_strings = load_shared_strings(zf)
        sheet_paths = load_sheet_paths(zf)

        for sheet_name, sheet_path in sheet_paths:
            rows = list(iter_sheet_rows(zf, sheet_path, shared_strings))
            column_info = find_columns(rows)
            if column_info is None:
                missing_columns.append(sheet_name)
                continue

            header_row_idx, callid_col, label_col, label_source = column_info
            sheets_seen.append(sheet_name)
            label_sources.append({"sheet": sheet_name, "column": label_source})

            for row in rows[header_row_idx + 1 :]:
                rows_scanned += 1
                label = str(row.get(label_col, "")).strip()
                if not label:
                    continue

                callid = str(row.get(callid_col, "")).strip()
                if not callid:
                    continue

                seen = seen_by_label.setdefault(label, set())
                if callid in seen:
                    continue

                label_samples = samples_by_label.setdefault(label, [])
                if len(label_samples) >= sample_limit(label):
                    continue

                label_samples.append(
                    {
                        "callid": callid,
                        "rm ai label": label,
                    }
                )
                seen.add(callid)

    samples = [
        sample
        for label in sorted(samples_by_label)
        for sample in samples_by_label[label]
    ]
    label_counts = {label: len(rows) for label, rows in sorted(samples_by_label.items())}
    shortfalls = {
        label: {"target": sample_limit(label), "actual": count}
        for label, count in label_counts.items()
        if count < sample_limit(label)
    }
    summary = {
        "source": str(path),
        "output": str(OUTPUT_DIR / f"{path.stem}_label_samples.csv"),
        "sheets_used": sheets_seen,
        "sheets_missing_columns": missing_columns,
        "label_sources": label_sources,
        "rows_scanned": rows_scanned,
        "sample_count": len(samples),
        "label_counts": label_counts,
        "shortfalls": shortfalls,
    }
    return samples, summary


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_samples: List[Dict[str, str]] = []

    for input_file in INPUT_FILES:
        path = Path(input_file)
        if not path.exists():
            summaries.append({"source": str(path), "error": "file not found"})
            continue

        samples, summary = extract_samples(path)
        write_csv(OUTPUT_DIR / f"{path.stem}_label_samples.csv", samples)
        all_samples.extend(samples)
        summaries.append(summary)

    write_csv(OUTPUT_DIR / "all_label_samples.csv", all_samples)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
