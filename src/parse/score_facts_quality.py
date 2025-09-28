#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                rows.append({"__parse_error__": str(e), "__line__": i})
    return pd.DataFrame(rows)


def _finite_ratio(a: int, b: int) -> float:
    return 0.0 if b <= 0 else round(100.0 * a / b, 2)


def score_one_file(path: Path) -> Dict[str, Any]:
    df = _read_jsonl(path)
    n = len(df)
    if n == 0:
        return {"file": str(path), "rows": 0, "score": 0.0}

    # Presence checks for required columns
    req_cols = ["concept", "period_label", "ticker", "form", "accno", "fy"]
    presence_ok = sum(1 for c in req_cols if c in df.columns)
    presence_rate = _finite_ratio(presence_ok, len(req_cols))

    # Non-null rates for key fields (only if columns exist)
    nonnull_rates: list[float] = []
    for c in req_cols:
        if c in df.columns:
            nonnull_rates.append(_finite_ratio(int(df[c].notna().sum()), n))

    nonnull_avg = (sum(nonnull_rates) / len(nonnull_rates)) if nonnull_rates else 0.0

    # FY numeric rate
    fy_num_rate = 0.0
    if "fy" in df.columns:
        try:
            fy_num_rate = _finite_ratio(int(pd.to_numeric(df["fy"], errors="coerce").notna().sum()), n)
        except Exception:
            fy_num_rate = 0.0

    # Numeric value quality: rows that look numeric vs clean numeric available
    looks_num = pd.Series([False] * n)
    for c in ("value_num", "value", "value_display", "value_raw"):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce") if c != "value_display" and c != "value_raw" else pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")
            looks_num = looks_num | s.notna()
    clean_ok = pd.Series([False] * n)
    if "value_num_clean" in df.columns:
        clean_ok = pd.to_numeric(df["value_num_clean"], errors="coerce").apply(lambda x: bool(np.isfinite(x)))
    numeric_rows = int(looks_num.sum())
    numeric_clean_rows = int((looks_num & clean_ok).sum())
    numeric_clean_rate = _finite_ratio(numeric_clean_rows, numeric_rows if numeric_rows > 0 else 1)

    # Duplicate rate by concept+period_label+accno(+unit)(+dims_signature)
    dup_rate = 0.0
    key_cols = [c for c in ["concept", "period_label", "accno", "unit", "dims_signature"] if c in df.columns]
    if key_cols:
        dups = int(df.duplicated(subset=key_cols).sum())
        dup_rate = _finite_ratio(dups, n)

    # Compose score (weights sum to 1.0). Penalize duplicates.
    # presence 0.15, nonnull 0.35, fy 0.10, numeric_clean 0.40, minus duplicates penalty up to 10 points
    raw_score = (
        0.15 * presence_rate
        + 0.35 * nonnull_avg
        + 0.10 * fy_num_rate
        + 0.40 * numeric_clean_rate
    )
    penalty = min(10.0, dup_rate * 0.2)  # if 5% dup -> penalty 1 point, cap 10
    score = max(0.0, min(100.0, raw_score - penalty))

    return {
        "file": str(path),
        "rows": n,
        "presence_rate": presence_rate,
        "nonnull_avg": round(nonnull_avg, 2),
        "fy_num_rate": fy_num_rate,
        "numeric_clean_rate": numeric_clean_rate,
        "dup_rate": dup_rate,
        "score": round(score, 2),
    }


def aggregate(scores: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not scores:
        return {"global_score": 0.0, "files": 0, "rows": 0}
    rows = 0
    for s in scores:
        rows += int(s.get("rows", 0))
    # Weighted by rows
    def wavg(key: str) -> float:
        num = 0.0
        den = 0
        for s in scores:
            r = int(s.get("rows", 0))
            num += float(s.get(key, 0.0)) * r
            den += r
        return 0.0 if den == 0 else round(num / den, 2)

    return {
        "files": len(scores),
        "rows": rows,
        "presence_rate": wavg("presence_rate"),
        "nonnull_avg": wavg("nonnull_avg"),
        "fy_num_rate": wavg("fy_num_rate"),
        "numeric_clean_rate": wavg("numeric_clean_rate"),
        "dup_rate": wavg("dup_rate"),
        "global_score": wavg("score"),
    }


def main():
    ap = argparse.ArgumentParser(description="Score fact.jsonl quality under a root directory")
    ap.add_argument("--root", default="data/processed", help="Root directory to scan")
    ap.add_argument("--min-files", type=int, default=50, help="Print first N files' scores for debugging")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = sorted(root.rglob("fact.jsonl"))
    scores = []
    for i, p in enumerate(files, 1):
        s = score_one_file(p)
        scores.append(s)
        if i <= args.min_files:
            print(f"[{i}] {s['score']:6.2f} | rows={s['rows']:5d} | dup={s['dup_rate']:5.2f}% | num_clean={s['numeric_clean_rate']:5.2f}% | {p}")

    agg = aggregate(scores)
    print("\n=== Aggregated ===")
    for k, v in agg.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


