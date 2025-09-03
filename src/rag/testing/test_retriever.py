#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic & E2E test for the retriever (multi-index).
- Tests both/fact/text retrieval with rich meta printing
- Optional numeric E2E (pick_two_period_values + compute_change + format_number_with_unit)
- Adds robust filtering:
  * form filter (e.g., 10-K)
  * year filter with annual-like period (FY/Q4/FQ4/4/Y or ~1y duration)
  * revenue concept whitelist (with regex fallback)
  * fallback ladder to avoid over-filtering to zero
"""
import pandas as pd
from collections import Counter
import os
import sys
import argparse
import traceback
from typing import List, Dict, Any, Optional
import re
from src.rag.numeric_cache import FactsNumericCache
from src.rag.retriever import REVENUE_CONCEPTS, REVENUE_CONCEPT_RE
# # ------- config: index dir (override if needed) -------
# DEFAULT_INDEX_DIR = os.path.abspath("data/index")

# # ------- revenue concept whitelist & regex fallback -------
# REVENUE_CONCEPTS = {
#     "us-gaap:SalesRevenueNet",
#     "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
#     "us-gaap:Revenues",
#     "ifrs-full:Revenue",
#     # issuer-specific (conservative):
#     "aapl:NetSales",
#     "aapl:Revenue",
# }
# REVENUE_CONCEPT_RE = re.compile(
#     r"^(?:us-gaap:(?:SalesRevenueNet|RevenueFromContractWithCustomer(?:ExcludingAssessedTax)?|Revenues)"
#     r"|ifrs-full:Revenue"
#     r"|[a-z0-9-]+:(?:NetSales|Revenue)s?)$",
#     re.IGNORECASE,
# )

# # ------- annual-like tokens & year regex -------
# _ANNUAL_FQ_TOKENS = {"FY", "Q4", "FQ4", "4", "Y"}
# _YEAR_RE = re.compile(r"(\d{4})")

# cache = FactsNumericCache("data/index/facts_numeric.parquet")
# # -------------------- Query alias expansion --------------------
# def expand_alias_en(q: str) -> str:
#     """Lightweight English alias expansion for revenue/net income/cash."""
#     ql = (q or "").lower()
#     extras = []
#     if any(k in ql for k in ["revenue", "sales", "net sales", "turnover"]):
#         extras.append("(revenue OR sales OR net sales OR total net sales OR turnover)")
#     if any(k in ql for k in ["net income", "profit", "earnings", "net earnings"]):
#         extras.append("(net income OR profit OR earnings OR net earnings)")
#     if "cash" in ql:
#         extras.append('(cash OR "cash and cash equivalents" OR "cash & cash eq")')
#     return q if not extras else (q + " " + " ".join(extras))


# # -------------------- Retriever import & index override --------------------
# def import_retriever(index_dir: str):
#     """Import retriever and override _INDEX_DIR if present."""
#     from src.rag import retriever as R
#     if hasattr(R, "_INDEX_DIR"):
#         R._INDEX_DIR = index_dir
#     from src.rag.retriever import hybrid_search
#     return hybrid_search


# # -------------------- Pretty printing --------------------
# def print_hits(title: str, hits, max_items: int = 8):
#     print(f"\n=== {title} | hits={len(hits)} ===")
#     if not hits:
#         print("[diag] no hits; check filters (ticker/year/form) and index path.")
#         return
#     for i, h in enumerate(hits[:max_items], 1):
#         m = h.get("meta", {}) or {}
#         src = m.get("_source_index") or m.get("file_type")
#         try:
#             sc = float(h.get("score") or 0.0)
#         except Exception:
#             sc = 0.0
#         print(f"{i:02d}. score={sc:.4f} "
#               f"fy={m.get('fy')} fq={m.get('fq')} form={m.get('form')} "
#               f"type={src} accno={m.get('accno')}")
#         extra = []
#         if m.get("page_no") is not None:
#             extra.append(f"p.{m.get('page_no')}")
#         if m.get("section"):
#             extra.append(f"[{m.get('section')}]")
#         if m.get("concept_primary"):
#             extra.append(f"concept={m.get('concept_primary')}")
#         if m.get("source_path"):
#             extra.append(os.path.basename(str(m.get("source_path"))))
#         if extra:
#             print("    ", " | ".join(extra))
#         snip = (h.get("snippet") or m.get("text_preview") or "").replace("\n", " ")[:160]
#         if snip:
#             print("    ", snip)


# def diag_hits(tag: str, hits: List[Dict[str, Any]]):
#     ys = [_to_year(h) for h in hits]
#     fqs = [str((h.get("meta") or {}).get("fq") or h.get("fq") or "").upper() for h in hits]
#     forms = [str((h.get("meta") or {}).get("form") or h.get("form") or "").upper() for h in hits]
#     concepts = []
#     for h in hits:
#         m = h.get("meta") or {}
#         c = m.get("concept_primary") or h.get("concept")
#         if c:
#             concepts.append(c)
#     print(f"[diag:{tag}] n={len(hits)} | year={Counter(ys)} | fq={Counter(fqs)} | form={Counter(forms)}")
#     if concepts:
#         topc = Counter(concepts).most_common(8)
#         print(f"[diag:{tag}] top concepts:", topc)


# # -------------------- Filters (self-contained) --------------------
# def _to_year(hit: Dict[str, Any]) -> Optional[int]:
#     """
#     Robust year extraction:
#     1) meta.fy / hit.fy (int or string like 'FY2023')
#     2) meta.years / hit.years (list/tuple with ints/strings)
#     3) meta.period_end/instant/period_start or same on hit
#     """
#     m = hit.get("meta") or {}

#     # 1) fy
#     for container in (m, hit):
#         fy = container.get("fy")
#         if fy is not None:
#             if isinstance(fy, int):
#                 return fy
#             if isinstance(fy, str):
#                 mm = _YEAR_RE.search(fy)
#                 if mm:
#                     try:
#                         return int(mm.group(1))
#                     except Exception:
#                         pass

#     # 2) years list
#     for container in (m, hit):
#         years = container.get("years")
#         if isinstance(years, (list, tuple)) and years:
#             for y in years:
#                 try:
#                     return int(y)
#                 except Exception:
#                     if isinstance(y, str):
#                         mm = _YEAR_RE.search(y)
#                         if mm:
#                             try:
#                                 return int(mm.group(1))
#                             except Exception:
#                                 pass

#     # 3) date-like fields
#     for container in (m, hit):
#         for k in ("period_end", "instant", "period_start"):
#             v = container.get(k)
#             if not v:
#                 continue
#             mm = _YEAR_RE.search(str(v))
#             if mm:
#                 try:
#                     return int(mm.group(1))
#                 except Exception:
#                     continue
#     return None


# def _is_annual_like(hit: Dict[str, Any]) -> bool:
#     m = hit.get("meta") or {}
#     fq = str(m.get("fq") or hit.get("fq") or "").upper()
#     if fq in _ANNUAL_FQ_TOKENS:
#         return True
#     # optional: duration heuristic
#     dur = m.get("duration_days") or hit.get("duration_days") or m.get("duration") or hit.get("duration")
#     try:
#         if dur and 330 <= int(dur) <= 400:
#             return True
#     except Exception:
#         pass
#     return False


# def filter_by_form(hits: List[Dict[str, Any]], form: Optional[str]) -> List[Dict[str, Any]]:
#     if not form:
#         return hits
#     f = form.upper()
#     out = []
#     for h in hits:
#         m = h.get("meta") or {}
#         hf = str(m.get("form") or h.get("form") or "").upper()
#         if hf == f:
#             out.append(h)
#     return out


# def filter_by_year_and_period(hits: List[Dict[str, Any]], target_year: int, require_annual_like: bool = True) -> List[Dict[str, Any]]:
#     out = []
#     for h in hits:
#         y = _to_year(h)
#         if y != target_year:
#             continue
#         if require_annual_like and not _is_annual_like(h):
#             continue
#         out.append(h)
#     return out


# def _get_concept(h: Dict[str, Any]) -> Optional[str]:
#     m = h.get("meta") or {}
#     return m.get("concept_primary") or h.get("concept")


# def filter_by_revenue_concepts(hits):
#     out = []
#     for h in hits:
#         m = h.get("meta") or {}
#         c = m.get("concept_primary") or h.get("concept")
#         if not c:
#             continue
#         if c == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" \
#            or REVENUE_CONCEPT_RE.match(c):
#             out.append(h)
#     return out


# def apply_numeric_filters_with_fallback(hits: List[Dict[str, Any]], year: int, form: str) -> List[Dict[str, Any]]:
#     """Laddered filtering for fact hits to avoid over-pruning to zero."""
#     # Step 1: strict (form → year+annual → revenue concept)
#     step1 = filter_by_form(hits, form=form)
#     step1 = filter_by_year_and_period(step1, target_year=year, require_annual_like=True)
#     step1 = filter_by_revenue_concepts(step1)
#     if step1:
#         return step1

#     # Step 2: relax annual-like constraint
#     step2 = filter_by_form(hits, form=form)
#     step2 = filter_by_year_and_period(step2, target_year=year, require_annual_like=False)
#     step2 = filter_by_revenue_concepts(step2)
#     if step2:
#         return step2

#     # Step 3: form + year only (no concept filter)
#     step3 = filter_by_form(hits, form=form)
#     step3 = filter_by_year_and_period(step3, target_year=year, require_annual_like=False)
#     if step3:
#         return step3

#     # Step 4: final fallback: form only
#     step4 = filter_by_form(hits, form=form)
#     return step4


# # -------------------- Numeric demo --------------------
# def numeric_demo(query: str, filters: dict, hits_fact):
#     try:
#         from src.qa.utils_parsing import (
#             parse_numeric_targets,
#             pick_two_period_values,
#             compute_change,
#             format_number_with_unit,
#         )
#         from src.rag.numeric_cache import FactsNumericCache
#         from src.rag.retriever import hybrid_search      # 用于文本兜底
#     except Exception:
#         print("[numeric-demo] imports failed — skipping.")
#         traceback.print_exc()
#         return

#     target, change_type = parse_numeric_targets(query)

#     v_cur = v_prev = unit = currency = None
#     citations = []

#     # --- A) parquet 优先通道 ---
#     try:
#         cache = FactsNumericCache("data/index/facts_numeric.parquet")
#         if cache.ok() and target == "revenue":
#             hit = cache.query_two_periods(
#                 ticker=filters["ticker"],
#                 fy=filters["year"],
#                 form=filters["form"],
#                 concepts=REVENUE_CONCEPTS,        # ← 只用这一个
#                 require_annual_like=True,         # ← FY/Q4/FQ4/4 都当年度
#             ) or cache.query_two_periods(
#                 ticker=filters["ticker"],
#                 fy=filters["year"],
#                 form=filters["form"],
#                 concepts=None,
#                 concepts_regex=r"(RevenueFromContractWithCustomer|SalesRevenueNet|Revenues|NetSales)",
#                 require_annual_like=True,
#             )
#             if hit:
#                 cur, prev = hit
#                 v_cur, v_prev = cur["value"], prev["value"]
#                 unit = cur.get("unit") or prev.get("unit")
#                 currency = cur.get("currency") or prev.get("currency")
#                 citations = [
#                     {"source_path": cur.get("source_path"), "accno": cur.get("accno")},
#                     {"source_path": prev.get("source_path"), "accno": prev.get("accno")},
#                 ]
#     except Exception:
#         print("[numeric-demo] parquet path/load issue — continuing without cache.")

#     # --- B) 用 fact 命中（如果有的话） ---
#     if v_cur is None or v_prev is None:
#         try:
#             v_cur, v_prev, unit, currency, citations = pick_two_period_values(hits_fact, target, filters)
#         except Exception:
#             pass

#     # --- C) 文本兜底：解析 MD&A 百分比（让 demo 不再“信息不足”） ---
#     if v_cur is None and v_prev is None:
#         try:
#             # 拉 10-K FY 的 text hits
#             text_filters = {"ticker": filters["ticker"], "year": filters["year"], "form": filters["form"]}
#             hits_text = hybrid_search(query, text_filters, top_k=20, target="text")
#             pct, abs_change = _parse_yoy_from_text(hits_text)
#             if pct is not None:
#                 # 仅百分比也能给出最终答案（单位跟随“百分比”）
#                 v_cur = v_prev = None
#                 unit = "%"
#                 currency = None
#                 # 用 compute_change 的接口：直接把 pct 当 delta
#                 delta = pct / 100.0
#                 final = f"{pct:.1f}%"
#                 print("\n[numeric-demo]")
#                 print(f"  parse_numeric_targets -> {target} {change_type}")
#                 print(f"  picked values -> {v_cur} {v_prev} {unit} {currency}")
#                 print(f"  compute_change -> {delta}")
#                 print(f"  final formatted -> {final}")
#                 # 给一条文本来源提示
#                 if hits_text:
#                     m = hits_text[0].get('meta') or {}
#                     print("  citations[0] ->", {"source_path": m.get("source_path"), "accno": m.get("accno")})
#                 return
#         except Exception:
#             pass

#     # --- 正常收尾：要么 parquet/fact 成功，要么仍然信息不足 ---
#     delta = compute_change(v_cur, v_prev, change_type)
#     final = format_number_with_unit(delta, unit, currency, change_type)

#     print("\n[numeric-demo]")
#     print(f"  parse_numeric_targets -> {target} {change_type}")
#     print(f"  picked values -> {v_cur} {v_prev} {unit} {currency}")
#     print(f"  compute_change -> {delta}")
#     print(f"  final formatted -> {final}")
#     if citations:
#         print("  citations[0] ->", citations[0])


# # --- 文本兜底解析器（放同文件里即可） ---
# import re
# _PCT_RE = re.compile(r"(increase|decrease|increased|decreased)\s+(\d+(?:\.\d+)?)\s*%", re.I)
# _USD_RE = re.compile(r"\$?\s*([\d,.]+)\s*(billion|million|thousand|bn|m|k)", re.I)

# def _parse_yoy_from_text(hits_text):
#     pct = None; abs_change = None
#     for h in hits_text[:20]:
#         snip = (h.get("snippet") or (h.get("meta") or {}).get("text_preview") or "")
#         if not snip:
#             continue
#         m = _PCT_RE.search(snip)
#         if m:
#             sign = -1 if m.group(1).lower().startswith("decreas") else 1
#             pct = sign * float(m.group(2))
#         m2 = _USD_RE.search(snip)
#         if m2:
#             val = float(m2.group(1).replace(",", ""))
#             unit = m2.group(2).lower()
#             scale = 1.0
#             if unit in ("billion","bn"): scale = 1e9
#             elif unit in ("million","m"): scale = 1e6
#             elif unit in ("thousand","k"): scale = 1e3
#             abs_change = val * scale
#         if pct is not None:
#             break
#     return pct, abs_change


# # -------------------- Main --------------------
# def main():
#     ap = argparse.ArgumentParser(description="Retriever diagnostic & numeric E2E.")
#     ap.add_argument("--index-dir", default=DEFAULT_INDEX_DIR, help="Path to data/index")
#     ap.add_argument("--q", default="AAPL 2023 revenue YoY", help="Query")
#     ap.add_argument("--ticker", default="AAPL")
#     ap.add_argument("--year", type=int, default=2023)
#     ap.add_argument("--form", default="10-K")
#     ap.add_argument("--topk", type=int, default=8)
#     ap.add_argument("--target", choices=["both", "fact", "text"], default="both")
#     ap.add_argument("--no-alias", action="store_true", help="Disable alias expansion")
#     ap.add_argument("--numeric-demo", action="store_true", help="Run numeric E2E on fact hits")
#     args = ap.parse_args()

#     print("[diag] CWD =", os.getcwd())
#     print("[diag] INDEX_DIR =", os.path.abspath(args.index_dir))

#     try:
#         hybrid_search = import_retriever(os.path.abspath(args.index_dir))
#     except Exception:
#         print("[diag] import failed:")
#         traceback.print_exc()
#         sys.exit(1)

#     # filters for retriever
#     filters = {"ticker": args.ticker}
#     if args.year:
#         filters["year"] = args.year
#     if args.form:
#         filters["form"] = args.form

#     # expanded query (optional)
#     q = args.q if args.no_alias else expand_alias_en(args.q)

#     try:
#         # route by target or run all three modes if target == both
#         if args.target == "both":
#             hits_both = hybrid_search(q, filters, topk=args.topk, target="both")
#             print_hits(f'both ({args.q})', hits_both)

#             hits_fact = hybrid_search(q, filters, topk=args.topk, target="fact")
#             diag_hits("raw-fact", hits_fact)
#             hits_fact = apply_numeric_filters_with_fallback(hits_fact, year=args.year, form=args.form or "10-K")
#             diag_hits("filtered-fact", hits_fact)
#             print_hits("fact only", hits_fact)

#             hits_text = hybrid_search(q, filters, topk=args.topk, target="text")
#             diag_hits("raw-text", hits_text)
#             # text：仅做 form + year + annual（用于 MD&A 兜底百分比）
#             hits_text = filter_by_form(hits_text, form=args.form or "10-K")
#             hits_text = filter_by_year_and_period(hits_text, target_year=args.year, require_annual_like=True)
#             diag_hits("filtered-text", hits_text)
#             print_hits("text only", hits_text)

#             if args.numeric_demo:
#                 # numeric E2E uses fact hits (take more candidates)
#                 hits_fact_more = hybrid_search(q, filters, topk=max(args.topk, 40), target="fact")
#                 hits_fact_more = apply_numeric_filters_with_fallback(hits_fact_more, year=args.year, form=args.form or "10-K")
#                 numeric_demo(args.q, filters, hits_fact_more)

#         else:
#             hits = hybrid_search(q, filters, topk=args.topk, target=args.target)

#             if args.target == "fact":
#                 diag_hits("raw-fact", hits)
#                 hits = apply_numeric_filters_with_fallback(hits, year=args.year, form=args.form or "10-K")
#                 diag_hits("filtered-fact", hits)
#             elif args.target == "text":
#                 diag_hits("raw-text", hits)
#                 hits = filter_by_form(hits, form=args.form or "10-K")
#                 hits = filter_by_year_and_period(hits, target_year=args.year, require_annual_like=True)
#                 diag_hits("filtered-text", hits)

#             print_hits(f'{args.target} ({args.q})', hits)

#             if args.numeric_demo and args.target in ("fact", "both"):
#                 hits_more = hybrid_search(q, filters, topk=max(args.topk, 40), target="fact")
#                 hits_more = apply_numeric_filters_with_fallback(hits_more, year=args.year, form=args.form or "10-K")
#                 numeric_demo(args.q, filters, hits_more)

#     except Exception:
#         print("[diag] runtime error:")
#         traceback.print_exc()
#         sys.exit(2)


# if __name__ == "__main__":
#     main()

# # Usage:
# # python -m src.rag.testing.test_retriever --numeric-demo


# '''
# python -m src.rag.testing.test_retriever --numeric-demo

# '''

# AAPL 2023 vs 2022
# df = pd.read_parquet("data/index/facts_numeric.parquet")
# mask = (
#     (df.ticker=="AAPL") & (df.form.str.upper()=="10-K") &
#     (df.fy_norm.isin([2023,2022])) &
#     (df.concept=="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
# )
# print(df[mask][["fy_norm","fq_norm","value_std","unit_std","accno","source_path"]])

import pandas as pd
import re
df = pd.read_parquet("data/index/facts_numeric.parquet")
print("rows:", len(df), "\ncols:", list(df.columns))

# 1) 看 period_type 分布
print(df["period_type"].value_counts(dropna=False).head())

# 2) 看 unit_family / unit_std
print(df["unit_family"].value_counts().head(5))
print(df["unit_std"].value_counts().head(5))

# 3) AAPL 2022/2023 的 revenue 是否存在（不加 unit/period 过滤，先看原始）
mask = (
    (df["ticker"]=="AAPL") &
    (df["form"].str.upper()=="10-K") &
    (df["fy_norm"].isin([2022, 2023])) &
    (df["concept"].str.fullmatch(
        r"us-gaap:RevenueFromContractWithCustomer(?:ExcludingAssessedTax)?",
        case=False
    ))
)
print(df[mask][["fy_norm","fq_norm","period_type","unit_family","unit_std","value_std","dims_signature"]])
