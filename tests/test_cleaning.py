#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean data validator (RAG-ready) — 2025-08
- 不再要求/检查 rag_text
- 兼容 fact.{parquet|jsonl}，遍历 data/clean 下所有含有 fact.* 的目录
- 侧重结构一致性、字段完整性、轻量对账（calculation_edges）与跨文件一致性
- 输出 CSV 报告：data/clean/_reports/validation_report.csv
"""

from __future__ import annotations
import json, re, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterable, Set

import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT   = PROJECT_ROOT / "data" / "clean"
REPORT_DIR   = CLEAN_ROOT / "_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_CSV   = REPORT_DIR / "validation_report.csv"

# -----------------------------
# Config (可按需调整)
# -----------------------------
# fact 的“强需求列”（rag_text 已移除）
REQUIRED_FACT_COLS: List[str] = [
    "concept", "period_label", "ticker", "form", "accno", "fy"
]
# fact 的“柔性列”（如果存在就校验；不存在不报错）
OPTIONAL_FACT_COLS: List[str] = [
    "fq", "value_num_clean", "value_display", "unit", "decimals", "scale"
]
# 容差：父=Σ(子*weight) 的允许误差（绝对 1 或 相对 0.5% 取大）
REL_TOL = 0.005
ABS_TOL = 1.0
# 计算抽样：每个目录最多抽多少个父概念做对账
MAX_PARENTS_SAMPLE = 20

ACCNO_PAT = re.compile(r"^\d{10}-\d{2}-\d{6}$")  # 典型 accno 格式（可能不是所有都匹配，匹配失败仅提示）

# -----------------------------
# Helpers
# -----------------------------
def pct(x: int, n: int) -> float:
    return 0.0 if n == 0 else round(100.0 * x / n, 2)

def read_jsonl(p: Path) -> pd.DataFrame:
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for i, l in enumerate(f, 1):
            s=l.strip()
            if not s: 
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                rows.append({"__parse_error__": str(e), "__line__": i})
    return pd.DataFrame(rows)

def maybe_read(dirpath: Path, stem: str) -> Optional[pd.DataFrame]:
    for name in (f"{stem}.parquet", f"{stem}.jsonl"):
        p = dirpath / name
        if p.exists():
            return (pd.read_parquet(p) if p.suffix==".parquet" else read_jsonl(p))
    return None

def iter_fact_dirs() -> Iterable[Path]:
    """枚举所有包含 fact.parquet 或 fact.jsonl 的目录"""
    seen: Set[Path] = set()
    for pat in ("**/fact.parquet", "**/fact.jsonl"):
        for f in CLEAN_ROOT.rglob(pat):
            seen.add(f.parent)
    # 排除 _reports 之类的内部目录
    return sorted(d for d in seen if "_reports" not in str(d))

def add(records: List[Dict[str, Any]], dirpath: Path, t: str, sev: str, msg: str) -> None:
    records.append({"dir": str(dirpath), "type": t, "severity": sev, "msg": msg})

def to_int_maybe(x) -> Optional[int]:
    if pd.isna(x): 
        return None
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else None

def is_fq_ok_for_form(fq, form: str) -> bool:
    # 10-K: fq 可为空或非数字；10-Q: fq 应为 1~4
    if form == "10-K":
        return True  # 不强制
    if form == "10-Q":
        v = to_int_maybe(fq)
        return v in {1, 2, 3, 4}
    return True

def finite_ratio(a: float, b: float) -> Optional[float]:
    if b == 0 or not np.isfinite(a) or not np.isfinite(b):
        return None
    return a / b

# -----------------------------
# Fact checks
# -----------------------------
def check_facts(dirpath: Path, recs: List[Dict[str, Any]]):
    facts = maybe_read(dirpath, "fact")
    if facts is None or facts.empty:
        add(recs, dirpath, "facts", "error", "missing or empty fact.{parquet|jsonl}")
        return

    n = len(facts)

    # 1) 必需列
    for c in REQUIRED_FACT_COLS:
        if c not in facts.columns:
            add(recs, dirpath, "facts", "error", f"missing column {c}")

    # 2) 全局一致性（单目录内的 meta 是否一致）
    for col in ("ticker", "form", "accno"):
        if col in facts.columns:
            uniq = pd.Series(facts[col].dropna().unique())
            if len(uniq) > 1:
                add(recs, dirpath, "facts", "warn", f"{col} not uniform in dir: {len(uniq)} distinct")
            if col == "accno" and len(uniq) >= 1:
                # 检查主 accno 是否匹配模式（只做提示）
                head = str(uniq.iloc[0])
                if head and not ACCNO_PAT.match(head):
                    add(recs, dirpath, "facts", "info", f"accno format unusual: {head}")

    # 3) fy/fq 解析与表单制约
    if "fy" in facts.columns:
        bad_fy = facts["fy"].map(to_int_maybe).isna().sum()
        if bad_fy > 0:
            add(recs, dirpath, "facts", "warn", f"fy non-numeric {bad_fy}/{n} ({pct(bad_fy,n)}%)")
    if "fq" in facts.columns and "form" in facts.columns:
        bad_fq = (~facts.apply(lambda r: is_fq_ok_for_form(r.get("fq"), str(r.get("form"))), axis=1)).sum()
        if bad_fq > 0:
            add(recs, dirpath, "facts", "warn", f"fq invalid given form {bad_fq}/{n} ({pct(bad_fq,n)}%)")

    # 4) value 列质量
    valcol = "value_num_clean" if "value_num_clean" in facts.columns else None
    if valcol:
        # 非数字/非有限值
        bad = (~facts[valcol].apply(lambda v: np.issubdtype(type(v), np.number) and np.isfinite(v))).sum()
        if bad > 0:
            add(recs, dirpath, "facts", "warn", f"{valcol} non-finite or non-numeric {bad}/{n} ({pct(bad,n)}%)")

        # 简单离群提示（IQR），仅作为 info
        try:
            v = pd.to_numeric(facts[valcol], errors="coerce").dropna()
            if len(v) >= 20:
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 3*iqr, q3 + 3*iqr
                outliers = ((v < lo) | (v > hi)).sum()
                if outliers > 0:
                    add(recs, dirpath, "facts", "info", f"{valcol} extreme values (3*IQR) count={outliers}")
        except Exception:
            pass

    # 5) 重复行（同 concept+period_label+accno[+unit]）
    keys = [c for c in ["concept", "period_label", "accno", "unit"] if c in facts.columns]
    if keys:
        dup = facts.duplicated(subset=keys).sum()
        if dup > 0:
            add(recs, dirpath, "facts", "warn", f"duplicates by {keys}: {dup} rows")

    # 6) 概念/标签覆盖（如存在 labels）
    labs = maybe_read(dirpath, "labels")
    if labs is not None and {"concept", "label_text"}.issubset(labs.columns):
        fact_concepts = set(facts["concept"].dropna().unique())
        label_concepts = set(labs["concept"].dropna().unique())
        covered = len(fact_concepts & label_concepts)
        if covered < len(fact_concepts):
            add(recs, dirpath, "labels", "info", f"label coverage {covered}/{len(fact_concepts)} concepts")

# -----------------------------
# Definition / Calculation checks
# -----------------------------
def check_definition(dirpath: Path, recs: List[Dict[str, Any]]):
    defs = maybe_read(dirpath, "definition_arcs")
    if defs is None:
        add(recs, dirpath, "def", "info", "definition_arcs not found (optional)")
        return
    for c in ["from_concept", "to_concept", "linkrole"]:
        if c not in defs.columns:
            add(recs, dirpath, "def", "warn", f"missing column {c} in definition_arcs")

def check_calculation(dirpath: Path, recs: List[Dict[str, Any]]):
    cal  = maybe_read(dirpath, "calculation_edges")
    facts= maybe_read(dirpath, "fact")
    if cal is None:
        add(recs, dirpath, "calc", "info", "calculation_edges not found (optional)")
        return
    # 1) 必需列
    need = ["parent_concept", "child_concept", "weight"]
    for c in need:
        if c not in cal.columns:
            add(recs, dirpath, "calc", "error", f"missing column {c} in calculation_edges")
    # 2) NaN 占比（关键列）
    for c in need:
        if c in cal.columns:
            r = pct(int(cal[c].isna().sum()), len(cal))
            if r > 0:
                add(recs, dirpath, "calc", "warn", f"{c} NaN {r}%")

    if facts is None or facts.empty:
        return

    # 3) child 概念是否存在于 facts
    if "concept" in facts.columns and "child_concept" in cal.columns:
        factset = set(facts["concept"].dropna().unique())
        miss = [c for c in pd.Series(cal["child_concept"].dropna().unique()) if c not in factset]
        if miss:
            add(recs, dirpath, "calc", "info", f"children not in facts: {len(miss)} concepts (sample)")

    # 4) 轻量父子和校验（基于 value_num_clean）
    if not {"concept", "period_label"}.issubset(facts.columns):
        return
    valcol = "value_num_clean" if "value_num_clean" in facts.columns else None
    if not valcol:
        add(recs, dirpath, "calc_check", "info", "skip sum check (no value_num_clean)")
        return

    # 只抽样最多 MAX_PARENTS_SAMPLE 个父概念
    parents = pd.Series(cal.get("parent_concept", pd.Series(dtype=object))).dropna().unique().tolist()
    parents = parents[:MAX_PARENTS_SAMPLE]

    if not parents:
        return

    # 以 period_label 为分组做对账
    facts_idx = (
        facts.set_index(["concept", "period_label"])
            .sort_index()[valcol]
    )

    for pc in parents:
        rows = cal[cal["parent_concept"] == pc]
        children = rows[["child_concept", "weight"]].dropna()
        if children.empty:
            continue

        # 找到该父概念在哪些期有值
        per_with_parent = facts.loc[facts["concept"] == pc, "period_label"].dropna().unique().tolist()
        for per in per_with_parent:
            try:
                parent_val = facts_idx.loc[(pc, per)]
            except KeyError:
                continue
            if pd.isna(parent_val) or not np.isfinite(parent_val):
                continue

            total = 0.0
            child_found_any = False
            for _, r in children.iterrows():
                cc, w = r["child_concept"], r["weight"]
                try:
                    v = facts_idx.loc[(cc, per)]
                except KeyError:
                    continue
                if pd.isna(v) or not np.isfinite(v):
                    continue
                child_found_any = True
                total += float(v) * float(w)

            # 如果一个子项都没找到，属于“信息不足”，只做 info 提示
            if not child_found_any:
                add(recs, dirpath, "calc_check", "info", f"{pc}@{per}: no child values to sum")
                continue

            parent = float(parent_val)
            tol = max(ABS_TOL, REL_TOL * abs(parent))
            if not np.isfinite(total) or not np.isfinite(parent):
                continue
            if abs(parent - total) > tol:
                add(recs, dirpath, "calc_check", "info",
                    f"sum mismatch {pc} @ {per}: parent={parent:.2f}, sum={total:.2f}, tol={tol:.2f}")

# -----------------------------
# Main
# -----------------------------
def main():
    records: List[Dict[str, Any]] = []
    dirs = list(iter_fact_dirs())
    if not dirs:
        print("[WARN] 未找到任何含 fact.{parquet|jsonl} 的目录。请确认 data/clean 是否已产出。")
        return

    for d in dirs:
        try:
            check_facts(d, records)
            check_definition(d, records)
            check_calculation(d, records)
        except Exception as e:
            add(records, d, "validator", "error", f"exception: {type(e).__name__}: {e}")

    if not records:
        print("[OK] 未发现问题项。")
        return

    df = pd.DataFrame(records)
    df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] 报告生成：{REPORT_CSV}  |  发现 {len(df)} 条记录")
    print(df.groupby(["type", "severity"]).size())

if __name__ == "__main__":
    main()
