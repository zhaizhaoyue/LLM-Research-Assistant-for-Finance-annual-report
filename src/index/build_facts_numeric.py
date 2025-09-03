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
from collections import Counter

import pandas as pd
import numpy as np

# -----------------------
# Config & Constants
# -----------------------
EXTENSIBLE_RE     = re.compile(r"ExtensibleList$", re.I)
TEXTBLOCK_RE      = re.compile(r"TextBlock$", re.I)
CURRENCY_HINT_RE  = re.compile(r"\b(?:USD|EUR|GBP|JPY|CNY|RMB|HKD|CAD|AUD|CHF|KRW|SGD|TWD)\b", re.I)
ISO4217_RE        = re.compile(r"iso4217", re.I)
REVENUE_RE        = re.compile(r"(revenue|revenues|sales)", re.I)
PERCENT_HINT_RE   = re.compile(r"(percent|percentage|pct|\d+\s*%)", re.I)

NUMERIC_UNIT_FAMILIES = {
    "money", "monetary", "currency",   # 允许 currency，统一后会映射到 money
    "shares", "count", "duration", "percent", "ratio"
}

ALLOWED_BASENAMES = {
    "fact.parquet", "fact.jsonl", "fact.jsonl.gz",
    "fact_like.jsonl", "fact_like.jsonl.gz",
}

NUM_RE = re.compile(r"[-+]?\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?|[-+]?\d+(?:\.\d+)?")

# -----------------------
# IO Helpers
# -----------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".gz") else open
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
    读取单个 clean/fact.* 或 fact_like.*，返回 DataFrame（若失败返回 None）
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

        # 只取本脚本需要用到的字段（不存在的先补 None）
        needed = [
            "ticker","form","accno",
            "fy_norm","fq_norm","fy","fq","year",
            "period_start","period_end","instant",
            "concept","label_text",
            "value_num","value_raw","value_display",
            "unit_normalized","unit_family","decimals","scale",
            "statement_hint","dims_signature","source_path","doc_date",
            "rag_text", "context_id", "context", "dimensions", "period_label",
        ]
        for col in needed:
            if col not in df.columns:
                df[col] = None

        # 统一类型中的 NaN → None（避免 json 序列化问题）
        df = df[needed].copy()
        df = df.where(pd.notna(df), None)
        return df
    except Exception:
        return None

# -----------------------
# Value & Unit Inference
# -----------------------
def to_int_safe(v) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return int(v)
    except Exception:
        return None

def value_num_with_fallback(row: pd.Series) -> Optional[float]:
    v = row.get("value_num")
    # pandas 可能是 numpy.nan
    if v is not None and not (isinstance(v, float) and math.isnan(v)):
        try:
            return float(v)
        except Exception:
            pass

    s = str(row.get("value_display") or row.get("value_raw") or "").strip()
    if not s:
        return None
    # 直接包含 % 的优先去掉 % 后再解析
    s_clean = s.replace("%", "")
    m = NUM_RE.search(s_clean)
    if not m:
        return None
    num = m.group(0).replace(",", "")
    neg = num.startswith("(") and num.endswith(")")
    num = num.strip("()")
    try:
        x = float(num)
        return -x if neg else x
    except Exception:
        return None

def normalize_unit_family_name(name: Optional[str]) -> str:
    name = (name or "").strip().lower()
    if name in ("currency", "monetary"):
        return "money"
    return name

def derive_unit_family(row: pd.Series) -> str:
    # 1) 现有 unit_family（含归一化）
    uf = normalize_unit_family_name(row.get("unit_family"))
    if uf in NUMERIC_UNIT_FAMILIES:
        return uf

    # 2) unit_normalized 的线索
    un = str(row.get("unit_normalized") or "").upper()
    if CURRENCY_HINT_RE.search(un) or ISO4217_RE.search(un) or un == "$":
        return "money"
    if un == "%" or "%" in str(row.get("value_display") or ""):
        return "percent"

    # 3) 概念/标签中特征词兜底（revenue/sales → money）
    concept = str(row.get("concept") or "")
    label   = str(row.get("label_text") or "")
    if REVENUE_RE.search(concept) or REVENUE_RE.search(label):
        return "money"

    # 4) 其它百分比线索
    if PERCENT_HINT_RE.search(label) or PERCENT_HINT_RE.search(concept):
        return "percent"

    return uf  # 可能是空串

def normalize_value_std(value_num: float, unit_family: str, scale: Optional[float]) -> (Optional[float], Optional[str]):
    """
    统一数值到标准单位：
    - money/shares/count/duration: 乘以 scale (若为空=1)
    - percent/ratio: 若 |x| > 1.5 则视为百分数，/100，标准单位为 'pct'（小数）
    返回 (value_std, unit_std)
    """
    unit_family = normalize_unit_family_name(unit_family)
    sc = scale if (isinstance(scale, (int, float)) and not (isinstance(scale, float) and math.isnan(scale))) else 1.0
    try:
        x = float(value_num)
    except Exception:
        return (None, None)

    if unit_family in {"percent", "ratio"}:
        # 统一为小数（例如 31.2% → 0.312）
        if abs(x) > 1.5:
            x = x / 100.0
        return (x, "pct")

    if unit_family in {"money", "shares", "count", "duration"}:
        return (x * sc, unit_family)

    # fallback
    return (x, unit_family or "unknown")

# -----------------------
# Row Filters
# -----------------------
def is_numeric_fact_row(row: pd.Series) -> bool:
    # 1) 先拿到可用的数值
    v = value_num_with_fallback(row)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False

    # 2) 概念黑名单（纯枚举/层级类）
    concept = (row.get("concept") or "")
    if EXTENSIBLE_RE.search(concept):
        return False

    # 3) TextBlock 通常是大段文字：允许解析到数字的极少数情况，但默认仍可保留
    #    这里不强行剔除 TextBlock，因为有些 textblock 会带结构化数值。

    # 4) 单位家族放宽 + 兜底推断
    uf = derive_unit_family(row)
    return uf in NUMERIC_UNIT_FAMILIES

# -----------------------
# Build Pipeline
# -----------------------
def build_facts_numeric(clean_root: Path,
                        out_parquet: Path,
                        out_jsonl: Optional[Path],
                        jsonl_sample: int = 2000) -> None:
    # 收集文件
    files: List[Path] = []
    for p in clean_root.rglob("*"):
        if p.name.startswith("_") or not p.is_file():
            continue
        if p.name.lower() in ALLOWED_BASENAMES:
            files.append(p)

    if not files:
        print(f"[WARN] no fact files found under {clean_root}")
        return

    # 读取
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
    for col in ("fy_norm", "fq_norm"):
        all_df[col] = all_df[col].apply(to_int_safe)

    # 丢弃原因统计
    drop_reason = Counter()

    def _why_drop(row):
        if value_num_with_fallback(row) is None:
            return "no_numeric_value"
        if EXTENSIBLE_RE.search(str(row.get("concept") or "")):
            return "extensible"
        uf = derive_unit_family(row)
        if uf not in NUMERIC_UNIT_FAMILIES:
            return f"unit_family={uf or 'EMPTY'}"
        return None

    reasons = []
    for _, r in all_df.iterrows():
        reason = _why_drop(r)
        reasons.append(reason)
        if reason is not None:
            drop_reason[reason] += 1

    print("[DROP STATS]", dict(drop_reason))
    mask_numeric = pd.Series([x is None for x in reasons], index=all_df.index)
    df_num = all_df[mask_numeric].copy()

    # 统一/派生单位家族
    df_num["unit_family"] = df_num.apply(derive_unit_family, axis=1)

    # 计算标准值
    std_vals = df_num.apply(
        lambda r: normalize_value_std(
            value_num_with_fallback(r),
            r.get("unit_family") or r.get("unit_normalized"),
            r.get("scale"),
        ), axis=1
    )
    df_num["value_std"] = [t[0] for t in std_vals]
    df_num["unit_std"]  = [t[1] for t in std_vals]

    # period_type
    def _ptype(r):
        has_dur = (pd.notna(r.get("period_start")) and pd.notna(r.get("period_end")))
        # 考虑我们上方 where(pd.notna)->None 的处理
        if r.get("period_start") is not None and r.get("period_end") is not None:
            return "duration"
        return "instant"
    df_num["period_type"] = df_num.apply(_ptype, axis=1)

    df_num["is_numeric"] = True

    # 输出字段（保留 unit_normalized 以便排查；period_label/rag_text 也保留）
    out_cols = [
        "ticker","form","accno","fy_norm","fq_norm","period_type",
        "period_start","period_end","instant",
        "concept","label_text",
        "value_num","value_std","unit_family","unit_std","unit_normalized","scale","decimals",
        "statement_hint","dims_signature",
        "source_path","doc_date",
        "period_label","rag_text"
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
        n = min(int(jsonl_sample), len(out_df))
        sample_df = out_df.sample(n) if n < len(out_df) else out_df
        with open(out_jsonl, "w", encoding="utf-8") as w:
            for _, row in sample_df.iterrows():
                # 转成 python 原生类型，避免 numpy 类型导致的非 JSON 可序列化
                rec = {k: (None if (isinstance(v, float) and (math.isnan(v))) else v) for k, v in row.to_dict().items()}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] sample jsonl -> {out_jsonl}  rows={n}")

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Build numeric facts view from clean/fact & fact_like files.")
    ap.add_argument("--clean-root", type=str, default="data/clean", help="Root dir of cleaned outputs")
    ap.add_argument("--out-parquet", type=str, default="data/index/facts_numeric.parquet")
    ap.add_argument("--out-jsonl", type=str, default="data/index/facts_numeric.sample.jsonl")
    ap.add_argument("--jsonl-sample", type=int, default=2000)
    args = ap.parse_args()

    build_facts_numeric(
        clean_root=Path(args.clean_root),
        out_parquet=Path(args.out_parquet),
        out_jsonl=Path(args.out_jsonl) if args.out_jsonl else None,
        jsonl_sample=int(args.jsonl_sample),
    )

if __name__ == "__main__":
    main()


'''
python -m src.index.build_facts_numeric `
  --clean-root data/clean `
  --out-parquet data/index/facts_numeric.parquet `
  --out-jsonl data/index/facts_numeric.sample.jsonl `
  --jsonl-sample 3000

'''