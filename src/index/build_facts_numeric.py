#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import argparse
import json
import gzip
import math
import re
from typing import Iterable, List, Dict, Any, Optional

import pandas as pd

EXTENSIBLE_RE = re.compile(r"ExtensibleList$", re.I)

NUMERIC_UNIT_FAMILIES = {
    "money", "monetary", "shares", "count", "duration", "percent", "ratio"
}

# ---------- utils ----------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                # 忽略坏行
                continue
    return rows

def load_one_clean_fact_file(p: Path) -> Optional[pd.DataFrame]:
    """
    读取单个 clean/fact.* 文件，返回 DataFrame（若失败返回 None）
    兼容 parquet/jsonl(.gz)
    """
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower() == ".jsonl" or p.name.endswith(".jsonl.gz"):
            df = pd.DataFrame(read_jsonl(p))
        else:
            return None
        if df is None or df.empty:
            return None
        # 只取本脚本需要用到的字段（不存在的先填充）
        needed = [
            "ticker","form","accno",
            "fy_norm","fq_norm",
            "period_start","period_end","instant",
            "concept","label_text",
            "value_num","value_raw","value_display",
            "unit_normalized","unit_family","decimals","scale",
            "statement_hint","dims_signature","source_path","doc_date"
        ]
        for col in needed:
            if col not in df.columns:
                df[col] = None
        return df[needed].copy()
    except Exception:
        return None

def is_numeric_fact_row(row: pd.Series) -> bool:
    # value_num 必须是有效数值
    v = row.get("value_num")
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
    except Exception:
        return False

    # 概念过滤：剔除 ExtensibleList（这类是成员/分类，而非数值）
    concept = (row.get("concept") or "")
    if EXTENSIBLE_RE.search(concept):
        return False

    # 单位家族过滤：仅保留常见数值家族
    unit_family = (row.get("unit_family") or row.get("unit_normalized") or "")
    unit_family = str(unit_family).lower()
    if unit_family not in NUMERIC_UNIT_FAMILIES:
        # 有些清洗产物可能没填 unit_family，但 value_num 仍然是钱；这里可按需放宽
        # 保险起见：如果 unit_family 为空但有明显 money 线索，可以保留；否则丢弃
        return False

    return True

def normalize_value_std(value_num: float, unit_family: str, scale: Optional[float]) -> (float, str):
    """
    统一数值到标准单位：
    - money/shares/count/duration: 乘以 scale (若为空按 1)
    - percent/ratio: 若绝对值 >1.5 则视为百分数，除以 100，标准单位为 'pct'（小数）
    返回 (value_std, unit_std)
    """
    unit_family = (unit_family or "").lower()
    sc = scale if (isinstance(scale, (int, float)) and not math.isnan(scale)) else 1.0
    try:
        x = float(value_num)
    except Exception:
        return (None, None)

    if unit_family in {"percent", "ratio"}:
        # 标准化为小数（如 31.2% -> 0.312）
        if abs(x) > 1.5:
            x = x / 100.0
        return (x, "pct")

    if unit_family in {"money", "monetary", "shares", "count", "duration"}:
        return (x * sc, unit_family)

    # fallback（通常不会到这里）
    return (x, unit_family or "unknown")

def to_int_safe(v) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return int(v)
    except Exception:
        return None

# ---------- main ----------
def build_facts_numeric(clean_root: Path,
                        out_parquet: Path,
                        out_jsonl: Optional[Path],
                        jsonl_sample: int = 2000) -> None:
    files: List[Path] = []
    for form_dir in clean_root.rglob("*"):
        if form_dir.name.startswith("_"):
            continue
        if form_dir.is_file():
            name = form_dir.name.lower()
            if name == "fact.parquet" or name == "fact.jsonl" or name == "fact.jsonl.gz":
                files.append(form_dir)

    if not files:
        print(f"[WARN] no fact files found under {clean_root}")
        return

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = load_one_clean_fact_file(f)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        print(f"[WARN] loaded 0 rows from {len(files)} files")
        return

    all_df = pd.concat(dfs, ignore_index=True)

    # 类型修复
    all_df["fy_norm"] = all_df["fy_norm"].apply(to_int_safe)
    all_df["fq_norm"] = all_df["fq_norm"].apply(to_int_safe)

    # 仅保留数值行
    mask_numeric = all_df.apply(is_numeric_fact_row, axis=1)
    df_num = all_df[mask_numeric].copy()

    # 统一标准值
    std_vals = df_num.apply(
        lambda r: normalize_value_std(
            r.get("value_num"),
            r.get("unit_family") or r.get("unit_normalized"),
            r.get("scale")
        ), axis=1
    )
    df_num["value_std"] = [t[0] for t in std_vals]
    df_num["unit_std"]  = [t[1] for t in std_vals]

    # period_type 标注
    def _ptype(r):
        has_dur = (pd.notna(r.get("period_start")) and pd.notna(r.get("period_end")))
        return "duration" if has_dur else "instant"
    df_num["period_type"] = df_num.apply(_ptype, axis=1)

    # is_numeric 标注
    df_num["is_numeric"] = True

    # 选择输出字段（尽量稳定）
    out_cols = [
        "ticker","form","accno","fy_norm","fq_norm","period_type",
        "period_start","period_end","instant",
        "concept","label_text",
        "value_num","value_std","unit_family","unit_std","scale","decimals",
        "statement_hint","dims_signature",
        "source_path","doc_date"
    ]
    for c in out_cols:
        if c not in df_num.columns:
            df_num[c] = None

    out_df = df_num[out_cols].copy()

    # 写 parquet
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    print(f"[OK] facts_numeric.parquet -> {out_parquet}  rows={len(out_df)}")

    # 可选：写一份 jsonl sample（用于人工 spot check）
    if out_jsonl:
        n = min(jsonl_sample, len(out_df))
        sample_df = out_df.sample(n) if n < len(out_df) else out_df
        with open(out_jsonl, "w", encoding="utf-8") as w:
            for _, row in sample_df.iterrows():
                w.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        print(f"[OK] sample jsonl -> {out_jsonl}  rows={n}")

def main():
    ap = argparse.ArgumentParser(description="Build numeric facts view from clean/fact files.")
    ap.add_argument("--clean-root", type=str, default="data/clean")
    ap.add_argument("--out-parquet", type=str, default="data/clean/facts_numeric.parquet")
    ap.add_argument("--out-jsonl", type=str, default="data/clean/facts_numeric.sample.jsonl")
    ap.add_argument("--jsonl-sample", type=int, default=2000)
    args = ap.parse_args()

    build_facts_numeric(
        clean_root=Path(args.clean_root),
        out_parquet=Path(args.out_parquet),
        out_jsonl=Path(args.out_jsonl) if args.out_jsonl else None,
        jsonl_sample=int(args.jsonl_sample)
    )

if __name__ == "__main__":
    main()


#python src/index/build_facts_numeric.py --clean-root data/clean --out-parquet data/clean/facts_numeric.parquet
