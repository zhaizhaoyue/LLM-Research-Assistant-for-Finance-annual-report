# src/qa/utils_parsing.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import re
import math

# -----------------------------
# 1. 解析 query 的目标指标与运算类型
# -----------------------------
def parse_numeric_targets(query: str) -> Tuple[str, str]:
    q = (query or "").lower()
    # 标准化：把连字符/斜杠变成空格，便于匹配
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

    # 变化类型（支持 year-over-year, y/y, q/q）
    if ("同比" in q_norm) or ("yoy" in q_norm) or ("year over year" in q_norm) or ("y y" in q_norm):
        change_type = "YoY"
    elif ("环比" in q_norm) or ("qoq" in q_norm) or ("quarter over quarter" in q_norm) or ("q q" in q_norm):
        change_type = "QoQ"
    elif any(k in q_norm for k in ["差额", "变化", "变动", "difference", "diff", "change"]):
        change_type = "diff"
    else:
        change_type = "raw"

    return target, change_type



# -----------------------------
# 2. 从 hits 中抽取两期数值 + 单位/币种
# -----------------------------
import re

_NUM = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?")
ALIAS = {
    "revenue": {
        "concepts": {"us-gaap:salesrevenuenet"},
        "tokens": {"revenue", "net sales", "sales"}
    },
    "net_income": {
        "concepts": {"us-gaap:netincomeloss"},
        "tokens": {"net income", "profit"}
    },
    "cash": {
        "concepts": {"us-gaap:cashandcashequivalentsatcarryingvalue"},
        "tokens": {"cash", "cash equivalents"}
    }
}

def _is_target_hit(h, target: str) -> bool:
    meta = h.get("meta", {}) or {}
    concept = (meta.get("concept") or "").lower()
    label = (meta.get("label_search_tokens") or meta.get("row_label") or "").lower()
    snip  = (h.get("snippet") or "").lower()
    a = ALIAS.get(target, {"concepts": set(), "tokens": set()})
    if concept in a["concepts"]:
        return True
    return any(tok in label or tok in snip for tok in a["tokens"])


def pick_two_period_values(hits, target, filters):
    FILETYPE_RANK = {"table": 0, "fact": 1, "cal": 2, "text_chunk": 3, "text": 3}
    v_cur = v_prev = unit = currency = None
    citations = []
    year = filters.get("year")

    # 先把更可能是表格的命中排前
    def _score(h):
        ft = (h.get("meta", {}).get("file_type") or "").lower()
        return FILETYPE_RANK.get(ft, 9)

    hits_sorted = sorted(hits, key=_score)

    def _maybe_number_from_snippet(h):
        """从 snippet 回退抓一个'像金额'的数字，并转 float；失败返回 None"""
        snip = (h.get("snippet") or "").lower()
        m = _NUM.search(snip)
        if not m: return None
        txt = m.group(0).replace(",", "")
        try: return float(txt)
        except: return None

    for h in hits_sorted:
        meta = h.get("meta", {}) or {}
        concept = (meta.get("concept") or "").lower()
        label = (meta.get("label_search_tokens") or "").lower()
        if not _is_target_hit(h, target):
            # 目标不匹配就跳过（很松的判定）
            continue

        val = meta.get("value")
        if val is None:
            val = _maybe_number_from_snippet(h)  # 回退：从正文抓数
        if val is None:
            continue

        # 单位/币种（尽早记录一次）
        unit = unit or meta.get("unit")
        currency = currency or meta.get("currency")

        fy = meta.get("fy")
        if year is not None and fy == year and v_cur is None:
            v_cur = float(val)
            citations.append(make_cite_from_meta(h))
        elif year is not None and fy == year - 1 and v_prev is None:
            v_prev = float(val)
            citations.append(make_cite_from_meta(h))
        elif year is None and v_cur is None:
            # 没给 year：先把遇到的第一条当当前期
            v_cur = float(val)
            citations.append(make_cite_from_meta(h))

        if v_cur is not None and v_prev is not None:
            break

    return v_cur, v_prev, unit, currency, citations



def make_cite_from_meta(hit: Dict[str, Any]) -> dict:
    meta = hit.get("meta", {})
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


# -----------------------------
# 3. 计算变化值
# -----------------------------
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
        if change_type == "YoY" or change_type == "QoQ":
            if v_prev == 0:
                return None
            return (v_cur - v_prev) / v_prev
        elif change_type == "diff":
            return v_cur - v_prev
    except Exception:
        return None
    return None


# -----------------------------
# 4. 格式化输出
# -----------------------------
def format_number_with_unit(
    value: Optional[float],
    unit: Optional[str] = None,
    currency: Optional[str] = None,
    change_type: str = "raw",
) -> str:
    if value is None:
        return "信息不足"

    if change_type in ("YoY", "QoQ"):
        # 转百分比
        return f"{value*100:.2f}%"
    else:
        # 数字缩写（千/百万/十亿）
        abs_v = abs(value)
        if abs_v >= 1e9:
            val_str = f"{value/1e9:.2f}B"
        elif abs_v >= 1e6:
            val_str = f"{value/1e6:.2f}M"
        elif abs_v >= 1e3:
            val_str = f"{value/1e3:.2f}K"
        else:
            val_str = f"{value:.2f}"

        prefix = "$" if currency in ("USD", "usd", "$") else ""
        suffix = f" {unit}" if unit else ""
        return prefix + val_str + suffix
