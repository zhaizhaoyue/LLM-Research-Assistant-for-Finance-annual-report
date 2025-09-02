# src/qa/utils_detect.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import re
from datetime import datetime, timedelta
import calendar
# -----------------------------
# 0) 全局同义词/别名表（可按需扩展）
# -----------------------------
ALIAS: Dict[str, List[str]] = {
    "revenue": [
        "revenue", "revenues", "sales", "net sales", "total net sales",
        "turnover", "营业收入", "营收"
    ],
    "net_income": [
        "net income", "profit", "net profit", "earnings", "net earnings",
        "净利润", "利润", "收益"
    ],
    "cash": [
        "cash", "cash and cash equivalents", "cash & cash eq",
        "现金", "现金等价物"
    ],
    # 你可以继续加：gross margin, operating income, opex, capex, ...
}

# 变化类型关键词（中文/英文）
KW_NUMERIC = [
    "同比", "环比", "增长", "下降", "变动", "变化", "增幅", "减少", "qoq", "yoy",
    "year over year", "quarter over quarter", "increase", "decrease",
    "difference", "diff", "change", "rate", "%"
]
KW_REASON = [
    "原因", "驱动", "影响因素", "why", "because", "due to", "drivers", "explain",
    "主要", "导致", "贡献度"
]
KW_AMOUNT = [
    "多少", "金额", "数值", "比例", "百分比", "rate", "ratio", "percent", "%"
]

RE_NUMBER = re.compile(r"[-+]?\d+([,]\d{3})*(\.\d+)?\s*%?")  # 包含百分号的数字


# -----------------------------
# 1) 意图判定：numeric / textual / mixed
# -----------------------------
def detect_query_type(query: str, hits=None) -> str:
    q = (query or "").lower()

    explicit_math = any(k in q for k in ["yoy","同比","qoq","环比","year over year","quarter over quarter","%"]) or bool(RE_NUMBER.search(q))
    reason_kw     = any(k in q for k in KW_REASON)
    soft_numeric  = any(k in q for k in ["increase","decrease","growth","变化","变动","增幅"])

    if reason_kw and not explicit_math:
        return "textual"               # 解释为主、没有明确数学指令 → textual
    if explicit_math and reason_kw:
        return "mixed"
    if explicit_math:
        return "numeric"

    # 没有明确数学，仅有“increase/decrease”这类软信号：
    if soft_numeric:
        return "textual"               # 更保守：没有数字/百分号就当 textual

    # 命中类型启发（可留）
    if hits:
        numeric_like = sum(1 for h in hits[:5] if (h.get("meta", {}).get("file_type") or "") in ("fact","cal","table"))
        if numeric_like >= 3:
            return "numeric"
    return "textual"



# -----------------------------
# 2) Query 扩展：加入别名（检索前可用）
# -----------------------------
def expand_with_alias(query: str, targets: Optional[List[str]] = None) -> str:
    """
    将 query 扩展为包含主要目标的同义词，便于 BM25/向量检索召回。
    例：
      "AAPL 2023 revenue YoY" →
      "AAPL 2023 revenue YoY (sales OR net sales OR turnover OR 营收 OR 营业收入)"
    """
    if not targets:
        targets = guess_targets_from_query(query)

    expansions: List[str] = []
    for t in targets:
        syns = ALIAS.get(t, [])
        if not syns:
            continue
        # 构造一个 OR 串
        clause = "(" + " OR ".join(sorted(set(syns))) + ")"
        expansions.append(clause)

    q = query.strip()
    if expansions:
        q = q + " " + " ".join(expansions)
    return q


def guess_targets_from_query(query: str) -> List[str]:
    q = (query or "").lower()
    tgs: List[str] = []
    for k, syns in ALIAS.items():
        for s in syns:
            if s.lower() in q:
                tgs.append(k)
                break
    return sorted(set(tgs)) or ["revenue"]  # 无匹配时给个常见缺省


# -----------------------------
# 3) 命中过滤：按目标指标筛 hits
# -----------------------------
def filter_hits_by_target(hits: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    """
    依据 meta.concept / meta.label_search_tokens / snippet 与别名匹配，筛出更可能相关的命中。
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
    # 保持原顺序（假设原先已按 score 排好）
    return out or hits


# -----------------------------
# 4) 跨期定位：上一年/上一季度
# -----------------------------
def locate_previous_period(
    cur_meta: Dict[str, Any],
    prefer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    输入：包含 {fy, fq, period_end 或 instant} 的 meta（尽量提供）
    输出：{"fy": fy_prev, "fq": fq_prev, "period_end": "..."} （尽可能补全）
    - prefer: "YoY"|"QoQ"|None；None 时自动推断（若 fq 存在 → QoQ，否则 YoY）
    """
    fy = _to_int(cur_meta.get("fy"))
    fq = _fq_norm(cur_meta.get("fq"))
    pend = _parse_date(cur_meta.get("period_end") or cur_meta.get("instant"))

    # 自动模式：有 fq → QoQ；否则 YoY
    mode = prefer or ("QoQ" if fq else "YoY")

    if mode == "QoQ" and fy and fq:
        fy_prev, fq_prev = _prev_quarter(fy, fq)
        pend_prev = _prev_quarter_date(pend) if pend else None
        return _pack_prev(fy_prev, fq_prev, pend_prev)

    # YoY：季度或年报都 -1 年
    if fy:
        fy_prev = fy - 1
    else:
        fy_prev = None
    pend_prev = _shift_year(pend, years=-1) if pend else None
    # 季度保持不变（若有）
    return _pack_prev(fy_prev, fq, pend_prev)


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
    # 常见写法：Q1/Q2/Q3/Q4、FY、FQ4 等
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
    # 处理 2/29 等边界：若新日期非法，则回退到该月最后一天
    if not d:
        return ""
    try:
        return _fmt_date(d.replace(year=d.year + years))
    except ValueError:
        # 简单兜底：向前回退 1 天直到合法（最多 3 次）
        nd = d
        for _ in range(3):
            nd = nd - timedelta(days=1)
            try:
                return _fmt_date(nd.replace(year=nd.year + years))
            except ValueError:
                continue
        return _fmt_date(nd)




def _quarter_end(year: int, month: int) -> str:
    # 返回 year, month 所在季度的季度末（3/6/9/12）的 yyyy-mm-dd
    q = ((month - 1) // 3) + 1               # 1..4
    end_m = q * 3                            # 3/6/9/12
    last = calendar.monthrange(year, end_m)[1]
    return f"{year:04d}-{end_m:02d}-{last:02d}"

def _prev_quarter_date(d: Optional[datetime]) -> Optional[str]:
    if not d:
        return None
    # 求当前季度
    cur_q = ((d.month - 1) // 3) + 1         # 1..4
    # 上一季度
    if cur_q == 1:
        year, q = d.year - 1, 4
    else:
        year, q = d.year, cur_q - 1
    end_m = q * 3                            # 3/6/9/12
    last = calendar.monthrange(year, end_m)[1]
    return f"{year:04d}-{end_m:02d}-{last:02d}"



# -----------------------------
# 5) 辅助：从 hits 猜目标（可选）
# -----------------------------
def guess_target_from_hits(hits: List[Dict[str, Any]]) -> Optional[str]:
    """
    若 query 不明确目标，可从命中概念/label 中猜测一个最常见目标。
    """
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
