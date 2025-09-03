# src/qa/utils_citation.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import re

# --------- Types ---------
Citation = Dict[str, Any]
Hit = Dict[str, Any]

# --------- Config ---------
_REQUIRED_META_KEYS = ["source_path", "accno", "ticker", "form", "fy"]
_OPTIONAL_META_KEYS = ["section", "item", "page_no", "lines", "concept", "fq", "page"]

_FQ_ORDER = {"FY": 5, "Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}

# =========================================
# Utils
# =========================================
def _norm_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    try:
        return os.path.normpath(p)
    except Exception:
        return p

def _norm_ticker(x: Optional[str]) -> Optional[str]:
    return str(x).upper() if x else x

def _norm_form(x: Optional[str]) -> Optional[str]:
    return str(x).upper() if x else x

def _norm_concept(x: Optional[str]) -> Optional[str]:
    return str(x).lower() if x else x

def _to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _to_int_or_none(x) -> Optional[int]:
    try:
        v = int(x)
        return v
    except Exception:
        try:
            # 有时 page 会是 "123.0"
            if isinstance(x, str) and re.fullmatch(r"\d+(?:\.0+)?", x):
                return int(float(x))
        except Exception:
            pass
    return None

def _norm_lines(lines: Any) -> Optional[List[int]]:
    if not lines:
        return None
    if isinstance(lines, (list, tuple)):
        out = []
        for v in lines:
            iv = _to_int(v)
            if iv is not None:
                out.append(iv)
        return out or None
    # 单个数字
    iv = _to_int(lines)
    return [iv] if iv is not None else None

def _fq_norm(s: Any) -> Optional[str]:
    if not s:
        return None
    txt = str(s).upper()
    if txt in _FQ_ORDER:
        return txt
    m = re.search(r"Q([1-4])", txt)
    return f"Q{m.group(1)}" if m else ("FY" if "FY" in txt else None)

def _prefer(a, b):
    """选第一个非空；若 a 为空返回 b。"""
    return a if a not in (None, "", []) else b

# =========================================
# Core builders
# =========================================
def make_citation(hit: Hit) -> Citation:
    """
    统一从检索命中构造 Citation，包含必要归一化。
    期望 hit = {"chunk_id", "meta": {...}}；meta 尽量包含 _REQUIRED_META_KEYS。
    """
    meta = hit.get("meta", {}) or {}

    # 多来源字段兜底：page/page_no；section/item
    page_raw = _prefer(meta.get("page_no"), meta.get("page"))
    section_raw = _prefer(meta.get("section"), meta.get("item"))

    c: Citation = {
        "source_path": _norm_path(_prefer(meta.get("source_path"), meta.get("path"))),
        "accno": meta.get("accno"),
        "ticker": _norm_ticker(meta.get("ticker")),
        "form": _norm_form(meta.get("form")),
        "fy": _to_int(meta.get("fy")),
        "fq": _fq_norm(meta.get("fq")),
        "section": section_raw,
        "page": _to_int_or_none(page_raw),
        "chunk_id": hit.get("chunk_id"),
        "lines": _norm_lines(meta.get("lines")),
        "concept": _norm_concept(meta.get("concept")),
    }
    return c

def normalize_citation(cite: Citation) -> Citation:
    """对外来/历史 citation 做一次就地归一化与兜底。"""
    if not cite:
        return cite
    cite = dict(cite)  # 复制一份
    cite["source_path"] = _norm_path(_prefer(cite.get("source_path"), cite.get("path")))
    cite["ticker"] = _norm_ticker(cite.get("ticker"))
    cite["form"] = _norm_form(cite.get("form"))
    cite["fy"] = _to_int(_prefer(cite.get("fy"), cite.get("year")))
    cite["fq"] = _fq_norm(cite.get("fq"))
    # 页面/行号
    page_raw = _prefer(cite.get("page"), cite.get("page_no"))
    cite["page"] = _to_int_or_none(page_raw)
    cite.pop("page_no", None)
    cite["lines"] = _norm_lines(cite.get("lines"))
    # section/item 归一
    cite["section"] = _prefer(cite.get("section"), cite.get("item"))
    # concept 大小写统一
    cite["concept"] = _norm_concept(cite.get("concept"))
    return cite

# =========================================
# Validation & ranking
# =========================================
def validate_citation(cite: Citation) -> Tuple[bool, List[str], List[str]]:
    """
    轻量校验：确保关键字段存在；对可能影响前端跳转的字段给 warnings。
    返回: (ok, missing, warnings)
    """
    c = normalize_citation(cite)
    missing = [k for k in _REQUIRED_META_KEYS if not c.get(k)]
    warnings: List[str] = []

    if not c.get("page"):
        warnings.append("page missing")
    if not c.get("section"):
        warnings.append("section missing")
    if not c.get("fq"):
        warnings.append("fq missing")
    return (len(missing) == 0), missing, warnings

def _sig_for_dedupe(c: Citation):
    """去重签名：优先 chunk_id；否则 (accno, page, lines)；再兜底 (source_path, page)。"""
    c = normalize_citation(c)
    if c.get("chunk_id"):
        return ("chunk", c["chunk_id"])
    if c.get("accno") and c.get("page") is not None:
        return ("accno+page+lines", c["accno"], c["page"], tuple(c.get("lines") or []))
    return ("path+page", c.get("source_path"), c.get("page"))

def dedupe_citations(citations: List[Citation]) -> List[Citation]:
    seen = set()
    out: List[Citation] = []
    for c in citations:
        sig = _sig_for_dedupe(c)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(normalize_citation(c))
    return out

def _rank_key(c: Citation):
    """排序：FY desc → FQ 优先级 desc → page asc（无 page 放后）"""
    c = normalize_citation(c)
    fy = c.get("fy") or -1
    fq_ord = _FQ_ORDER.get((c.get("fq") or "FY"), 0)
    page = c.get("page")
    page_key = page if page is not None else 1_000_000
    # 负号用于降序 FY / FQ
    return (-fy, -fq_ord, page_key)

def sort_citations(citations: List[Citation]) -> List[Citation]:
    return sorted((normalize_citation(c) for c in citations), key=_rank_key)

# =========================================
# Builders & helpers
# =========================================
def ensure_citations(citations: List[Citation], hits: List[Hit]) -> List[Citation]:
    """
    兜底策略：
    1) 若已有引用 → 归一化、去重、排序；
    2) 若无引用但有命中 → 用第一条命中生成引用；
    3) 否则 → 返回空列表。
    """
    if citations:
        return sort_citations(dedupe_citations([normalize_citation(c) for c in citations]))
    if hits:
        return [make_citation(hits[0])]
    return []

def top_citations_from_hits(hits: List[Hit], k: int = 3) -> List[Citation]:
    """
    从命中里直接挑前 k 条作为引用（常用于 textual/mixed 回答）。
    要求 hits 按 score 降序。
    """
    cites = [make_citation(h) for h in hits[:k]]
    return sort_citations(dedupe_citations(cites))

def compress_lines(lines: Optional[List[int]], max_span: int = 8) -> Optional[List[int]]:
    """
    行号压缩（例如 [2061,2062,2063,2068] → [2061,2063,2068]），
    过长则首尾保留并使用省略号。
    """
    if not lines:
        return lines
    try:
        seq = sorted(set(int(x) for x in lines))
    except Exception:
        return lines

    if len(seq) <= 2:
        return seq

    out: List[Any] = [seq[0]]
    for i in range(1, len(seq) - 1):
        prev_, cur, nxt = seq[i - 1], seq[i], seq[i + 1]
        if (cur - prev_ <= 1) and (nxt - cur <= 1):
            continue
        out.append(cur)
    out.append(seq[-1])

    if len(out) > max_span:
        half = max_span // 2
        out = out[:half] + ["…"] + out[-half:]
    return out

def merge_citations(*lists: List[Citation]) -> List[Citation]:
    """合并多来源引用（如表格+正文），并去重、排序。"""
    merged: List[Citation] = []
    for lst in lists:
        if not lst:
            continue
        merged.extend(lst)
    return sort_citations(dedupe_citations(merged))

def citation_to_display(cite: Citation) -> str:
    """
    简短展示：AAPL 10-K FY2023 p.123 [Item 7]
    """
    c = normalize_citation(cite)
    parts = []
    if c.get("ticker"):
        parts.append(str(c["ticker"]))
    if c.get("form"):
        parts.append(str(c["form"]))
    if c.get("fy"):
        parts.append(f"FY{c['fy']}")
    if c.get("fq") and c["fq"] != "FY":
        parts.append(c["fq"])
    if c.get("page") is not None:
        parts.append(f"p.{c['page']}")
    if c.get("section"):
        parts.append(f"[{c['section']}]")
    return " ".join(parts) or (c.get("source_path") or "citation")

def citation_to_brief(cite: Citation) -> str:
    """
    更短：AAPL FY2023 p.26
    """
    c = normalize_citation(cite)
    parts = []
    if c.get("ticker"):
        parts.append(str(c["ticker"]))
    if c.get("fy"):
        parts.append(f"FY{c['fy']}")
    if c.get("page") is not None:
        parts.append(f"p.{c['page']}")
    return " ".join(parts) or "citation"

def citation_to_logline(cite: Citation) -> str:
    """
    机器日志友好：AAPL|10-K|2023|p26|accno=...|chunk=...
    """
    c = normalize_citation(cite)
    segs = [
        str(_prefer(c.get("ticker"), "")),
        str(_prefer(c.get("form"), "")),
        str(_prefer(c.get("fy"), "")),
        f"p{c['page']}" if c.get("page") is not None else "",
        f"accno={c.get('accno') or ''}",
        f"chunk={c.get('chunk_id') or ''}",
    ]
    return "|".join(segs).strip("|")

def shorten_source_path(cite: Citation, project_root: Optional[str] = None) -> str:
    """
    将绝对路径裁剪为相对项目根的短路径，便于前端展示。
    """
    c = normalize_citation(cite)
    p = c.get("source_path") or ""
    if not p:
        return ""
    try:
        if project_root and os.path.isabs(p) and p.startswith(project_root):
            return os.path.relpath(p, start=project_root)
    except Exception:
        pass
    # 兜底仅保留尾部 3 层
    parts = p.replace("\\", "/").split("/")
    return "/".join(parts[-3:])
