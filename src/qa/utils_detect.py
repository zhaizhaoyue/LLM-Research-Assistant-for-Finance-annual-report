# src/qa/utils_detect.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import re
from datetime import datetime, timedelta
import calendar

# -----------------------------
# 0) Global alias table (extend as needed)
# -----------------------------
ALIAS: Dict[str, List[str]] = {
    "revenue": [
        "revenue", "revenues", "sales", "net sales", "total net sales", "turnover"
    ],
    "net_income": [
        "net income", "profit", "net profit", "earnings", "net earnings"
    ],
    "cash": [
        "cash", "cash and cash equivalents", "cash & cash equivalents", "cash equivalents"
    ],
    # add more: gross margin, operating income, opex, capex, etc.
}

# Signal keywords
KW_REASON = [
    "why", "because", "due to", "drivers", "driver", "explain", "explanation",
    "factors", "reason", "reasons", "impact", "contributed", "attributed"
]

# Patterns indicating numeric intent / math-y questions
KW_NUMERIC_HARD = [
    "yoy", "qoq", "year over year", "quarter over quarter", "%"
]
KW_NUMERIC_SOFT = [
    "increase", "decrease", "growth", "decline", "change", "difference", "diff", "rate",
    "percent", "percentage", "ratio"
]

RE_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?")

# -----------------------------
# 1) Query type detection: numeric / textual / mixed
# -----------------------------
def detect_query_type(query: str, hits=None) -> str:
    q = (query or "").lower()

    explicit_math = any(k in q for k in KW_NUMERIC_HARD) or bool(RE_NUMBER.search(q))
    reason_kw     = any(k in q for k in KW_REASON)
    soft_numeric  = any(k in q for k in KW_NUMERIC_SOFT)

    if reason_kw and not explicit_math:
        return "textual"          # explanatory without explicit math
    if explicit_math and reason_kw:
        return "mixed"
    if explicit_math:
        return "numeric"

    # No explicit math; soft numeric wording only → safer to treat as textual
    if soft_numeric:
        return "textual"

    # Heuristic from hits (optional)
    if hits:
        numeric_like = sum(
            1 for h in hits[:5]
            if (h.get("meta", {}).get("file_type") or "") in ("fact", "cal", "table")
        )
        if numeric_like >= 3:
            return "numeric"

    return "textual"

# -----------------------------
# 2) Query expansion with aliases (before retrieval)
# -----------------------------
def expand_with_alias(query: str, targets: Optional[List[str]] = None) -> str:
    """
    Expand the query by appending OR-clauses of synonyms for the target metrics.
    Example:
      "AAPL 2023 revenue YoY" ->
      "AAPL 2023 revenue YoY (revenues OR sales OR net sales OR turnover)"
    """
    if not targets:
        targets = guess_targets_from_query(query)

    expansions: List[str] = []
    for t in targets:
        syns = ALIAS.get(t, [])
        if not syns:
            continue
        clause = "(" + " OR ".join(sorted(set(syns))) + ")"
        expansions.append(clause)

    q = (query or "").strip()
    if expansions:
        q = q + " " + " ".join(expansions)
    return q

def guess_targets_from_query(query: str) -> List[str]:
    q = (query or "").lower()
    tgs: List[str] = []
    for k, syns in ALIAS.items():
        if any(s.lower() in q for s in syns):
            tgs.append(k)
    return sorted(set(tgs)) or ["revenue"]  # default to a common target

# -----------------------------
# 3) Hit filtering: keep hits aligned with a target metric
# -----------------------------
def filter_hits_by_target(hits: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    """
    Filter hits whose meta/snippet bag contains target aliases.
    Preserves original order.
    """
    syns = set([target] + ALIAS.get(target, []))
    out: List[Dict[str, Any]] = []
    for h in hits:
        meta = h.get("meta", {}) or {}
        bag = " ".join([
            str(meta.get("concept") or ""),
            str(meta.get("label_search_tokens") or ""),
            str(h.get("snippet") or "")
        ]).lower()
        if any(s.lower() in bag for s in syns):
            out.append(h)
    return out or hits

# -----------------------------
# 4) Previous-period locator (YoY / QoQ)
# -----------------------------
def locate_previous_period(
    cur_meta: Dict[str, Any],
    prefer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Input: meta with {fy, fq, period_end or instant}
    Output: dict with previous period hints: {"fy", "fq", "period_end"} where possible.

    prefer: "YoY" | "QoQ" | None (auto: if fq exists → QoQ else YoY)
    """
    fy = _to_int(cur_meta.get("fy"))
    fq = _fq_norm(cur_meta.get("fq"))
    pend = _parse_date(cur_meta.get("period_end") or cur_meta.get("instant"))

    mode = prefer or ("QoQ" if fq else "YoY")

    if mode == "QoQ" and fy and fq:
        fy_prev, fq_prev = _prev_quarter(fy, fq)
        pend_prev = _prev_quarter_date(pend) if pend else None
        return _pack_prev(fy_prev, fq_prev, pend_prev)

    # YoY: subtract one fiscal year; keep quarter if present
    fy_prev = (fy - 1) if fy else None
    pend_prev = _shift_year(pend, years=-1) if pend else None
    return _pack_prev(fy_prev, fq, pend_prev)

# Helpers
def _pack_prev(fy_prev: Optional[int], fq_prev: Optional[str], pend_prev: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if fy_prev is not None:
        out["fy"] = fy_prev
    if fq_prev:
        out["fq"] = fq_prev
    if pend_prev:
        out["period_end"] = pend_prev
    return out

def _to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _fq_norm(fq: Any) -> Optional[str]:
    if not fq:
        return None
    s = str(fq).upper().strip()
    if s in {"Q1", "Q2", "Q3", "Q4"}:
        return s
    m = re.match(r".*Q([1-4]).*", s)
    if m:
        return f"Q{m.group(1)}"
    return None

def _prev_quarter(fy: int, fq: str) -> Tuple[int, str]:
    order = ["Q1", "Q2", "Q3", "Q4"]
    if fq not in order:
        return fy - 1, "Q4"
    idx = order.index(fq)
    if idx == 0:
        return fy - 1, "Q4"
    return fy, order[idx - 1]

def _parse_date(s: Any) -> Optional[datetime]:
    if not s:
        return None
    txt = str(s)
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(txt, fmt)
        except Exception:
            pass
    return None

def _fmt_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def _shift_year(d: datetime, years: int = -1) -> str:
    if not d:
        return ""
    try:
        return _fmt_date(d.replace(year=d.year + years))
    except ValueError:
        # e.g., Feb 29: back off by one day until valid (up to 3 tries)
        nd = d
        for _ in range(3):
            nd = nd - timedelta(days=1)
            try:
                return _fmt_date(nd.replace(year=nd.year + years))
            except ValueError:
                continue
        return _fmt_date(nd)

def _quarter_end(year: int, month: int) -> str:
    q = ((month - 1) // 3) + 1   # 1..4
    end_m = q * 3                # 3/6/9/12
    last = calendar.monthrange(year, end_m)[1]
    return f"{year:04d}-{end_m:02d}-{last:02d}"

def _prev_quarter_date(d: Optional[datetime]) -> Optional[str]:
    if not d:
        return None
    cur_q = ((d.month - 1) // 3) + 1
    if cur_q == 1:
        year, q = d.year - 1, 4
    else:
        year, q = d.year, cur_q - 1
    end_m = q * 3
    last = calendar.monthrange(year, end_m)[1]
    return f"{year:04d}-{end_m:02d}-{last:02d}"

# -----------------------------
# 5) Infer target from hits (optional)
# -----------------------------
def guess_target_from_hits(hits: List[Dict[str, Any]]) -> Optional[str]:
    score: Dict[str, int] = {}
    for h in hits[:8]:
        meta = h.get("meta", {}) or {}
        bag = " ".join([
            str(meta.get("concept") or ""),
            str(meta.get("label_search_tokens") or "")
        ]).lower()
        for t, syns in ALIAS.items():
            if any(s.lower() in bag for s in syns):
                score[t] = score.get(t, 0) + 1
    if not score:
        return None
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))[0][0]
