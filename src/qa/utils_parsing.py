# src/qa/utils_parsing.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import os
import re
import math
from collections import defaultdict
# =========================================
# 0. 别名与数字/工具
# =========================================
_NUM = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?")

ALIAS = {
    "revenue": {
        # 优先用概念匹配；小写
        "concepts": {"us-gaap:salesrevenuenet", "us-gaap:revenuefromcontractwithcustomersexcludingassessedtax"},
        # 其次用文本 token 模糊匹配
        "tokens": {"revenue", "net sales", "sales", "营收", "营业收入"},
    },
    "net_income": {
        "concepts": {"us-gaap:netincomeloss"},
        "tokens": {"net income", "profit", "净利润", "收益"},
    },
    "cash": {
        "concepts": {"us-gaap:cashandcashequivalentsatcarryingvalue"},
        "tokens": {"cash", "cash equivalents", "现金", "现金等价物"},
    },
}

FILETYPE_RANK = {"table": 0, "fact": 1, "cal": 2, "text_chunk": 3, "text": 3}

# ----------------------------- 
# 加强版目标匹配 & 文本数值抽取（支持 text fallback）
# -----------------------------
_NUM_MONEY = re.compile(
    r"(?P<prefix>\$)?\s*(?P<num>[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?P<unit>trillion|billion|million|thousand|bn|mn|k)?",
    re.IGNORECASE
)

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _unit_multiplier(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    u = unit.lower()
    if u in ("trillion",): return 1e12
    if u in ("billion", "bn"): return 1e9
    if u in ("million", "mn"): return 1e6
    if u in ("thousand", "k"): return 1e3
    return 1.0

def _looks_money_text(snippet: str, target: str) -> Optional[float]:
    """
    在正文 snippet 里，先看是否出现目标同义词，再抓“金额+可选单位词”，返回 float 金额（统一到绝对数，如 394,328,000,000）。
    仅作回退使用；优先 fact/meta。
    """
    s = (snippet or "")
    s_low = s.lower()

    # 目标同义词（仅英文）
    syns = ALIAS.get(target, {}).get("tokens", set()) if isinstance(ALIAS.get(target), dict) else []
    syns = set(syns) if syns else set()
    # 兼容你 utils_detect 里的英文别名
    if target == "revenue":
        syns |= {"revenue", "revenues", "net sales", "total net sales", "sales", "turnover"}
    elif target == "net_income":
        syns |= {"net income", "profit", "net profit", "earnings", "net earnings"}
    elif target == "cash":
        syns |= {"cash", "cash and cash equivalents", "cash equivalents", "cash & cash eq"}

    if not any(tok in s_low for tok in syns):
        return None

    # 抓金额
    m = None
    best = None
    for m in _NUM_MONEY.finditer(s):
        num_txt = m.group("num").replace(",", "")
        try:
            val = float(num_txt)
        except Exception:
            continue
        mul = _unit_multiplier(m.group("unit"))
        val = val * mul
        # 忽略明显太小的“比例/件数”（如 4、10 这类无单位的纯小数）
        if val < 1e6 and not m.group("prefix") and not m.group("unit"):
            continue
        best = val
        # 不贪心：取第一个匹配就好（正文里通常第一处就是该段主题数字）
        break
    return best

def _from_meta_value(h) -> Optional[float]:
    """尝试从 meta.value / value_display 中取数字。"""
    meta = h.get("meta", {}) or {}
    val = meta.get("value")
    if isinstance(val, (int, float)):
        return float(val)
    # value_display 可能是字符串
    vd = meta.get("value_display")
    if isinstance(vd, (int, float)):
        return float(vd)
    if isinstance(vd, str):
        m = _NUM.search(vd)
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except Exception:
                return None
    return None

def _edge_iter_values_for_target(h, target: str, prefer_year: Optional[int]) -> List[Tuple[Optional[int], float]]:
    """
    在 fact group 的 edges 里，针对目标指标（通过概念/同义词）收集 (fy, value) 候选。
    """
    meta = h.get("meta", {}) or {}
    edges = meta.get("edges") or []
    out: List[Tuple[Optional[int], float]] = []

    # 概念同义词：优先 us-gaap:SalesRevenueNet 等
    concept_hits = set()
    if isinstance(meta.get("edge_concepts"), list):
        concept_hits |= {str(c).lower() for c in meta["edge_concepts"]}
    cp = meta.get("concept_primary")
    if cp: concept_hits.add(str(cp).lower())
    # 目标概念关键词
    concept_keys = set()
    if target == "revenue":
        concept_keys = {"us-gaap:salesrevenuenet", "us-gaap:revenuefromcontractwithcustomerincludingsessales", "us-gaap:revenuefromcontractwithcustomerexcludingassessedtaxes"}
    elif target == "net_income":
        concept_keys = {"us-gaap:netincomeloss"}
    elif target == "cash":
        concept_keys = {"us-gaap:cashandcashequivalentsatcarryingvalue"}

    def _edge_val(e):
        v = e.get("value")
        if isinstance(v, (int, float)): return float(v)
        vd = e.get("value_display")
        if isinstance(vd, (int, float)): return float(vd)
        if isinstance(vd, str):
            m = _NUM.search(vd)
            if m:
                try: return float(m.group(0).replace(",", ""))
                except: return None
        return None

    # 先严格按目标概念筛选；再宽松用 label/snippet 匹配
    for e in edges:
        c = _norm(e.get("concept") or e.get("qname"))
        label = _norm(e.get("label") or e.get("row_label") or e.get("label_search_tokens"))
        snip = _norm(e.get("snippet") or "")

        # 是否匹配目标
        hit = False
        if c and (c in concept_keys):
            hit = True
        elif c and concept_hits and any(k in c for k in concept_keys):
            hit = True
        else:
            # 宽松 token（仅英文）
            if target == "revenue":
                hit = any(k in (label or "") for k in ["revenue", "net sales", "sales", "total net sales", "turnover"]) \
                      or any(k in (snip or "") for k in ["revenue", "net sales", "sales", "turnover"])
            elif target == "net_income":
                hit = any(k in (label or "") for k in ["net income", "profit", "earnings"]) \
                      or any(k in (snip or "") for k in ["net income", "profit", "earnings"])
            elif target == "cash":
                hit = any(k in (label or "") for k in ["cash", "cash equivalents"]) \
                      or any(k in (snip or "") for k in ["cash", "cash equivalents"])

        if not hit:
            continue

        v = _edge_val(e)
        if v is None:
            continue
        fy = e.get("fy_norm", e.get("fy"))
        try:
            fy = int(fy) if fy is not None else None
        except Exception:
            pass
        out.append((fy, v))

    # 如果没有 edge 命中，再尝试 meta.value
    if not out:
        mv = _from_meta_value(h)
        if mv is not None:
            fy = meta.get("fy")
            try:
                fy = int(fy) if fy is not None else None
            except Exception:
                pass
            out.append((fy, mv))

    # 简单排序：优先 prefer_year，再按 fy 降序
    def _score(item):
        fy, _ = item
        if prefer_year is not None and fy == prefer_year:
            return (2, fy or -1)
        if prefer_year is not None and fy == prefer_year - 1:
            return (1, fy or -1)
        return (0, fy or -1)
    out.sort(key=_score, reverse=True)
    return out

def _is_target_hit(h, target: str) -> bool:
    """
    更强的“是否目标”判断：考虑 concept_primary / edge_concepts / label_search_tokens / snippet
    """
    meta = h.get("meta", {}) or {}
    concept = _norm(meta.get("concept") or meta.get("concept_primary") or meta.get("meta_concept"))
    label = _norm(meta.get("label_search_tokens") or meta.get("row_label"))
    snip  = _norm(h.get("snippet") or meta.get("text_preview"))

    # 旧版别名
    a = ALIAS.get(target, {"concepts": set(), "tokens": set()})
    concepts = {c.lower() for c in a.get("concepts", set())}
    tokens   = {t.lower() for t in a.get("tokens", set())}

    # 目标概念集合（含 edges 聚合）
    edge_concepts = set()
    if isinstance(meta.get("edge_concepts"), list):
        edge_concepts |= {str(c).lower() for c in meta["edge_concepts"]}
    if concept:
        edge_concepts.add(concept)

    if concepts and (edge_concepts & concepts):
        return True

    # 纯 token 匹配（英文）
    if tokens and any(t in (label or "") or t in (snip or "") for t in tokens):
        return True

    # 兜底：根据 target 再做一次英文关键词匹配
    if target == "revenue":
        kws = ["revenue", "net sales", "sales", "total net sales", "turnover"]
    elif target == "net_income":
        kws = ["net income", "profit", "earnings", "net earnings"]
    elif target == "cash":
        kws = ["cash", "cash equivalents", "cash and cash equivalents"]
    else:
        kws = []
    return any(k in (label or "") or k in (snip or "") for k in kws)
    

def pick_two_period_values(hits, target, filters):
    """
    增强版：
    1) 仍然优先从 fact 命中（meta/edges）抓取两期数值；
    2) 若缺失，则回退到 text 命中（支持一次性从正文抓到两期）；
    3) 严格使用 filters.year 判定“当前期/上一期”；若 year 缺失，则取不同 FY 的前两条。
    """
    FILETYPE_RANK = {"fact": 0, "table": 0, "cal": 1, "text_chunk": 2, "text": 2}
    v_cur = v_prev = unit = currency = None
    citations = []
    year = filters.get("year")

    def _ft(h):
        ft = _norm(h.get("meta", {}).get("file_type") or "")
        return FILETYPE_RANK.get(ft, 9)

    # 先按 file_type 排序（fact 在前）
    hits_sorted = sorted(hits, key=_ft)

    # ---------- Pass 1：优先 fact ----------
    for h in hits_sorted:
        meta = h.get("meta", {}) or {}
        if _norm(meta.get("file_type")) != "fact":
            continue
        if not _is_target_hit(h, target):
            continue

        # 收集候选 (fy, value)
        cands = _edge_iter_values_for_target(h, target, year)
        if not cands:
            continue

        # 记录单位/币种
        unit = unit or meta.get("unit")
        currency = currency or meta.get("currency")

        # —— 关键改进：先按 FY 分桶，再配对 —— #
        by_fy = defaultdict(list)
        for fy, val in cands:
            by_fy[fy].append(float(val))

        def _first(xs):  # 取该年的第一个值即可
            return xs[0] if xs else None

        if year is not None:
            v_y   = _first(by_fy.get(year, []))
            v_y_1 = _first(by_fy.get(year - 1, []))
            if (v_cur is None) and (v_y is not None):
                v_cur = v_y; citations.append(make_cite_from_meta(h))
            if (v_prev is None) and (v_y_1 is not None):
                v_prev = v_y_1; citations.append(make_cite_from_meta(h))
            if (v_cur is not None) and (v_prev is not None):
                return v_cur, v_prev, unit, currency, citations
        else:
            # 无 year：选择两个“不同 FY”的值
            # 先取最大 FY，其次取另一个不同 FY
            fys_sorted = sorted([fy for fy in by_fy.keys() if fy is not None], reverse=True)
            if fys_sorted:
                fy0 = fys_sorted[0]
                v0 = _first(by_fy.get(fy0, []))
                if (v_cur is None) and (v0 is not None):
                    v_cur = v0; citations.append(make_cite_from_meta(h))
                # 找另一个不同 FY
                for fy1 in fys_sorted[1:]:
                    v1 = _first(by_fy.get(fy1, []))
                    if v1 is not None:
                        v_prev = v1; citations.append(make_cite_from_meta(h))
                        break
                if (v_cur is not None) and (v_prev is not None):
                    return v_cur, v_prev, unit, currency, citations

    # ---------- Pass 2：回退到 text ----------
    for h in hits_sorted:
        meta = h.get("meta", {}) or {}
        if _norm(meta.get("file_type")) != "text":
            continue
        if not _is_target_hit(h, target):
            continue

        snippet = h.get("snippet") or meta.get("text_preview") or ""
        s = snippet

        # 尝试一次性抓两年两数（典型 “during 2023 compared to 2022 … 394,328 … 365,817 …”）
        if year is not None:
            years_in_snip = re.findall(r"\b(20\d{2})\b", s)
            years_in_snip = [int(y) for y in years_in_snip if 2000 <= int(y) <= 2100]
            if {year, year - 1}.issubset(set(years_in_snip)):
                # 抓两个“大额”金额（带逗号/单位），按出现顺序假定：第1个=当年，第2个=上年
                nums = []
                for m in _NUM_MONEY.finditer(s):
                    num_txt = m.group("num").replace(",", "")
                    try:
                        val = float(num_txt) * _unit_multiplier(m.group("unit"))
                    except Exception:
                        continue
                    if val < 1e6 and not m.group("prefix") and not m.group("unit"):
                        continue
                    nums.append(val)
                    if len(nums) >= 3:  # 控制噪音
                        break
                if len(nums) >= 2:
                    if v_cur is None:
                        v_cur = float(nums[0]); citations.append(make_cite_from_meta(h))
                    if v_prev is None:
                        v_prev = float(nums[1]); citations.append(make_cite_from_meta(h))
                    # 文本中常见美元：默认 USD
                    currency = currency or ("USD" if "$" in s else meta.get("currency"))
                    unit = unit or meta.get("unit")
                    if v_cur is not None and v_prev is not None:
                        return v_cur, v_prev, unit, currency, citations

        # 若未命中两期成对，再按“单值”回退（原有逻辑）
        val = _looks_money_text(snippet, target)
        if val is None:
            continue

        currency = currency or ("USD" if "$" in snippet else meta.get("currency"))
        unit = unit or meta.get("unit")

        fy = meta.get("fy")
        try:
            fy = int(fy) if fy is not None else None
        except Exception:
            fy = None

        if year is not None:
            if fy == year and v_cur is None:
                v_cur = float(val); citations.append(make_cite_from_meta(h))
            elif fy == year - 1 and v_prev is None:
                v_prev = float(val); citations.append(make_cite_from_meta(h))
            if v_cur is not None and v_prev is not None:
                return v_cur, v_prev, unit, currency, citations
        else:
            if v_cur is None:
                v_cur = float(val); citations.append(make_cite_from_meta(h))
            elif v_prev is None and fy is not None:
                v_prev = float(val); citations.append(make_cite_from_meta(h))
                return v_cur, v_prev, unit, currency, citations

    # 两轮都没拿齐，返回已有（可能只有 v_cur）
    return v_cur, v_prev, unit, currency, citations

def _maybe_number_from_snippet(h: Dict[str, Any]) -> Optional[float]:
    snip = (h.get("snippet") or "").lower()
    m = _NUM.search(snip)
    if not m:
        return None
    txt = m.group(0).replace(",", "")
    try:
        return float(txt)
    except Exception:
        return None





def make_cite_from_meta(hit: Dict[str, Any]) -> dict:
    meta = hit.get("meta", {}) or {}
    return {
        "source_path": meta.get("source_path"),
        "accno": meta.get("accno"),
        "ticker": meta.get("ticker"),
        "form": meta.get("form"),
        "fy": meta.get("fy"),
        "fq": meta.get("fq"),
        "section": meta.get("section") or meta.get("item"),
        "page": meta.get("page_no"),
        "chunk_id": hit.get("chunk_id"),
    }


# =========================================
# 1) 解析 query 的目标指标与运算类型
# =========================================
def parse_numeric_targets(query: str) -> Tuple[str, str]:
    q = (query or "").lower()
    q_norm = q.replace("-", " ").replace("/", " ")

    # 指标
    if any(k in q_norm for k in ["revenue", "sales", "营收", "营业收入"]):
        target = "revenue"
    elif any(k in q_norm for k in ["net income", "profit", "净利润", "收益"]):
        target = "net_income"
    elif any(k in q_norm for k in ["cash", "现金", "现金等价物"]):
        target = "cash"
    else:
        target = "unknown"

    # 变化类型
    if ("同比" in q_norm) or ("yoy" in q_norm) or ("year over year" in q_norm) or ("y y" in q_norm):
        change_type = "YoY"
    elif ("环比" in q_norm) or ("qoq" in q_norm) or ("quarter over quarter" in q_norm) or ("q q" in q_norm):
        change_type = "QoQ"
    elif any(k in q_norm for k in ["差额", "变化", "变动", "difference", "diff", "change"]):
        change_type = "diff"
    else:
        change_type = "raw"

    return target, change_type


# =========================================
# 2) 从 hits 抽取两期数值；不足则回退到 facts_numeric.parquet
# =========================================
def _rank_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _score(h):
        ft = (h.get("meta", {}).get("file_type") or "").lower()
        return FILETYPE_RANK.get(ft, 9)
    return sorted(hits, key=_score)


def _extract_from_hits(
    hits: List[Dict[str, Any]],
    target: str,
    year: Optional[int],
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str], List[dict]]:
    v_cur = v_prev = unit = currency = None
    cites: List[dict] = []

    for h in _rank_hits(hits):
        if not _is_target_hit(h, target):
            continue

        meta = h.get("meta", {}) or {}
        val = meta.get("value")
        if val is None:
            val = _maybe_number_from_snippet(h)
        if val is None:
            continue

        unit = unit or meta.get("unit")
        currency = currency or meta.get("currency")

        fy = meta.get("fy")
        if year is not None and fy == year and v_cur is None:
            v_cur = float(val)
            cites.append(make_cite_from_meta(h))
        elif year is not None and fy == year - 1 and v_prev is None:
            v_prev = float(val)
            cites.append(make_cite_from_meta(h))
        elif year is None and v_cur is None:
            v_cur = float(val)
            cites.append(make_cite_from_meta(h))

        if v_cur is not None and v_prev is not None:
            break

    return v_cur, v_prev, unit, currency, cites


# ---------- facts_numeric.parquet 回退 ----------
# 允许不同清洗版本的列名差异，统一做“尽力取值”

_df_cache = None
_df_path_cache = None


def _load_facts_numeric(path: Optional[str] = None):
    global _df_cache, _df_path_cache
    if path is None:
        path = os.environ.get("FACTS_NUMERIC_PATH", "data/index/facts_numeric.parquet")
    if _df_cache is not None and _df_path_cache == path:
        return _df_cache

    try:
        import pandas as pd
    except Exception:
        return None  # 没装 pandas 就不做回退

    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    # 统一小写列名，方便后续兼容
    df.columns = [str(c).lower() for c in df.columns]
    _df_cache = df
    _df_path_cache = path
    return df


def _col(df, name: str, alt: List[str] = []) -> Optional[str]:
    name = name.lower()
    if name in df.columns:
        return name
    for a in alt:
        if a.lower() in df.columns:
            return a.lower()
    return None


def _extract_from_numeric_index(
    target: str,
    filters: Dict[str, Any],
    index_path: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str], List[dict]]:
    df = _load_facts_numeric(index_path)
    if df is None:
        return None, None, None, None, []

    ticker = (filters.get("ticker") or "").upper() or None
    form = (str(filters.get("form") or "") or "").upper() or None
    year = filters.get("year")

    # 列名映射（尽量容错）
    c_ticker = _col(df, "ticker", ["symbol"])
    c_form = _col(df, "form")
    c_fy = _col(df, "fy", ["year"])
    c_concept = _col(df, "concept")
    c_label = _col(df, "label_search_tokens", ["label", "row_label", "label_tokens"])
    c_value = _col(df, "value", ["amount", "val"])
    c_unit = _col(df, "unit")
    c_curr = _col(df, "currency", ["curr", "iso_currency"])
    c_accno = _col(df, "accno")
    c_src = _col(df, "source_path", ["source", "path"])
    c_fq = _col(df, "fq")
    c_section = _col(df, "section", ["item"])
    c_page = _col(df, "page_no", ["page"])

    q = df

    if c_ticker and ticker:
        q = q[q[c_ticker].str.upper() == ticker]
    if c_form and form:
        q = q[q[c_form].str.upper() == form]

    # 概念/label 过滤：优先概念，其次 token
    alias = ALIAS.get(target, {"concepts": set(), "tokens": set()})
    if c_concept and alias["concepts"]:
        q = q[q[c_concept].str.lower().isin({x.lower() for x in alias["concepts"]})]
    elif c_label and alias["tokens"]:
        # 用 contains 任一 token
        tok_re = "|".join([re.escape(t) for t in alias["tokens"]])
        q = q[q[c_label].str.lower().str.contains(tok_re, na=False)]

    # 只保留有数值的
    if c_value:
        q = q[q[c_value].notna()]
    else:
        return None, None, None, None, []

    # 取当前/前一年
    v_cur = v_prev = unit = currency = None
    cites: List[dict] = []

    def _first_row(xdf):
        # 返回最靠近“总表”的行：优先 FY、其次 fq 排序、再按 value 非空靠前
        if c_fq in xdf.columns:
            order = ["FY", "Q4", "Q3", "Q2", "Q1"]
            xdf = xdf.assign(_fq_ord=xdf[c_fq].map({v: i for i, v in enumerate(order)})).sort_values(
                by=["_fq_ord"], na_position="last"
            )
        return xdf.iloc[0]

    try:
        import pandas as pd
    except Exception:
        pd = None

    if year is not None and c_fy:
        q_cur = q[q[c_fy] == year]
        if len(q_cur) > 0:
            r = _first_row(q_cur)
            v_cur = float(r[c_value])
            unit = (r[c_unit] if c_unit else None) or unit
            currency = (r[c_curr] if c_curr else None) or currency
            cites.append({
                "source_path": r[c_src] if c_src else None,
                "accno": r[c_accno] if c_accno else None,
                "ticker": ticker,
                "form": form,
                "fy": int(r[c_fy]) if c_fy else None,
                "fq": (r[c_fq] if c_fq else None),
                "section": (r[c_section] if c_section else None),
                "page": (int(r[c_page]) if (c_page and pd is not None and pd.notna(r[c_page])) else None),
                "chunk_id": None,
            })

        q_prev = q[q[c_fy] == (year - 1)]
        if len(q_prev) > 0:
            r = _first_row(q_prev)
            v_prev = float(r[c_value])
            unit = (r[c_unit] if c_unit else None) or unit
            currency = (r[c_curr] if c_curr else None) or currency
            cites.append({
                "source_path": r[c_src] if c_src else None,
                "accno": r[c_accno] if c_accno else None,
                "ticker": ticker,
                "form": form,
                "fy": int(r[c_fy]) if c_fy else None,
                "fq": (r[c_fq] if c_fq else None),
                "section": (r[c_section] if c_section else None),
                "page": (int(r[c_page]) if (c_page and pd is not None and pd.notna(r[c_page])) else None),
                "chunk_id": None,
            })
    else:
        # 未指定年份：挑最近两期（按 FY desc, FQ 排序）
        if c_fy:
            q = q.sort_values(by=[c_fy], ascending=False)
        rows = q.head(2).to_dict("records")
        if rows:
            r0 = rows[0]
            try:
                v_cur = float(r0[c_value]); unit = (r0.get(c_unit) or unit); currency = (r0.get(c_curr) or currency)
            except Exception:
                pass
            cites.append({
                "source_path": r0.get(c_src),
                "accno": r0.get(c_accno),
                "ticker": r0.get(c_ticker).upper() if c_ticker else None,
                "form": r0.get(c_form),
                "fy": r0.get(c_fy),
                "fq": r0.get(c_fq),
                "section": r0.get(c_section),
                "page": r0.get(c_page),
                "chunk_id": None,
            })
        if len(rows) >= 2:
            r1 = rows[1]
            try:
                v_prev = float(r1[c_value]); unit = (r1.get(c_unit) or unit); currency = (r1.get(c_curr) or currency)
            except Exception:
                pass
            cites.append({
                "source_path": r1.get(c_src),
                "accno": r1.get(c_accno),
                "ticker": r1.get(c_ticker).upper() if c_ticker else None,
                "form": r1.get(c_form),
                "fy": r1.get(c_fy),
                "fq": r1.get(c_fq),
                "section": r1.get(c_section),
                "page": r1.get(c_page),
                "chunk_id": None,
            })

    return v_cur, v_prev, unit, currency, cites





# =========================================
# 3) 计算变化值
# =========================================
def compute_change(
    v_cur: Optional[float],
    v_prev: Optional[float],
    change_type: str,
) -> Optional[float]:
    if v_cur is None:
        return None
    if change_type == "raw":
        return v_cur
    if v_prev is None:
        return None

    try:
        if change_type in ("YoY", "QoQ"):
            if v_prev == 0:
                return None
            return (v_cur - v_prev) / v_prev
        elif change_type == "diff":
            return v_cur - v_prev
    except Exception:
        return None
    return None


# =========================================
# 4) 格式化输出
# =========================================
def format_number_with_unit(
    value: Optional[float],
    unit: Optional[str] = None,
    currency: Optional[str] = None,
    change_type: str = "raw",
) -> str:
    if value is None:
        return "信息不足"

    if change_type in ("YoY", "QoQ"):
        return f"{value*100:.2f}%"

    abs_v = abs(value)
    if abs_v >= 1e9:
        val_str = f"{value/1e9:.2f}B"
    elif abs_v >= 1e6:
        val_str = f"{value/1e6:.2f}M"
    elif abs_v >= 1e3:
        val_str = f"{value/1e3:.2f}K"
    else:
        val_str = f"{value:.2f}"

    prefix = "$" if (currency or "").upper() in {"USD", "$"} else ""
    suffix = f" {unit}" if unit else ""
    return prefix + val_str + suffix
