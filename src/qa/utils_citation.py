# src/qa/utils_citation.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os

_REQUIRED_META_KEYS = [
    "source_path", "accno", "ticker", "form", "fy",
]
_OPTIONAL_META_KEYS = [
    "section", "item", "page_no", "lines", "concept",
]

Citation = Dict[str, Any]
Hit = Dict[str, Any]


def _norm_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    # 统一分隔符，避免 Windows/Unix 差异导致的“同源不同键”
    return os.path.normpath(p)


def make_citation(hit: Hit) -> Citation:
    """
    将 retrieval 命中 (hit) 统一转换为引用结构。
    期望 hit = {"chunk_id", "meta": {...}}；meta 尽量包含 _REQUIRED_META_KEYS。
    返回示例：
    {
      "source_path": ".../US_AAPL_2023_10-K_....html",
      "accno": "0000...",
      "ticker": "AAPL",
      "form": "10-K",
      "fy": 2023, "fq": "Q4",
      "section": "Item 7",
      "page": 123,
      "chunk_id": "AAPL::text::group::42",
      "lines": [2061, 2068],
      "concept": "us-gaap:Revenues"
    }
    """
    meta = hit.get("meta", {}) or {}
    c: Citation = {
        "source_path": _norm_path(meta.get("source_path")),
        "accno": meta.get("accno"),
        "ticker": meta.get("ticker"),
        "form": meta.get("form"),
        "fy": meta.get("fy"),
        "fq": meta.get("fq"),
        "section": meta.get("section") or meta.get("item"),
        "page": meta.get("page_no"),
        "chunk_id": hit.get("chunk_id"),
        "lines": meta.get("lines"),
        "concept": meta.get("concept"),
    }
    return c


def validate_citation(cite: Citation) -> Tuple[bool, List[str]]:
    """
    轻量校验：确保关键字段存在，便于导出/前端跳转。
    """
    missing = []
    for k in _REQUIRED_META_KEYS:
        if not cite.get(k):
            missing.append(k)
    ok = len(missing) == 0
    return ok, missing


def dedupe_citations(citations: List[Citation]) -> List[Citation]:
    """
    引用去重：优先按 chunk_id；若无 chunk_id，则按 (source_path, page, lines) 去重。
    """
    seen = set()
    out: List[Citation] = []
    for c in citations:
        key = c.get("chunk_id")
        if key:
            sig = ("chunk", key)
        else:
            sig = ("pos", c.get("source_path"), c.get("page"), tuple(c.get("lines") or []))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def ensure_citations(citations: List[Citation], hits: List[Hit]) -> List[Citation]:
    """
    兜底策略：
    1) 若已有引用 → 去重并返回；
    2) 若无引用但有命中 → 用第一条命中生成引用；
    3) 否则 → 返回空列表。
    """
    if citations:
        return dedupe_citations(citations)
    if hits:
        return [make_citation(hits[0])]
    return []


def top_citations_from_hits(hits: List[Hit], k: int = 3) -> List[Citation]:
    """
    从命中里直接挑前 k 条作为引用（常用于 textual/mixed 回答）。
    要求 hits 按 score 降序。
    """
    cites = [make_citation(h) for h in hits[:k]]
    return dedupe_citations(cites)


def compress_lines(lines: Optional[List[int]], max_span: int = 8) -> Optional[List[int]]:
    """
    可选：行号压缩（例如 [2061,2062,2063,2068] → [2061,2063,2068]），
    以减少前端展示占用。你可以在 make_citation 前先处理 meta['lines']。
    """
    if not lines:
        return lines
    if len(lines) <= 2:
        return lines
    lines_sorted = sorted(set(lines))
    out = [lines_sorted[0]]
    for i in range(1, len(lines_sorted) - 1):
        prev_, cur, nxt = lines_sorted[i - 1], lines_sorted[i], lines_sorted[i + 1]
        # 如果 cur 与 prev/nxt 都相邻，压缩掉中间点，仅保留区段端点
        if (cur - prev_ <= 1) and (nxt - cur <= 1):
            continue
        out.append(cur)
    out.append(lines_sorted[-1])
    # 若区段过长则裁剪（可根据需要调整策略）
    if len(out) > max_span:
        out = out[: max_span // 2] + ["…"] + out[-max_span // 2 :]
    return out


def merge_citations(*lists: List[Citation]) -> List[Citation]:
    """
    合并多处来源的引用（如表格+正文），并去重。
    """
    merged: List[Citation] = []
    for lst in lists:
        if not lst:
            continue
        merged.extend(lst)
    return dedupe_citations(merged)


def citation_to_display(cite: Citation) -> str:
    """
    将引用转为简短展示字符串，便于日志/CLI 打印。
    例：AAPL 10-K FY2023 p.123 [Item 7]
    """
    parts = []
    if cite.get("ticker"):
        parts.append(str(cite["ticker"]))
    if cite.get("form"):
        parts.append(str(cite["form"]))
    if cite.get("fy"):
        parts.append(f"FY{cite['fy']}")
    page = f"p.{cite['page']}" if cite.get("page") is not None else None
    section = f"[{cite['section']}]" if cite.get("section") else None
    if page:
        parts.append(page)
    if section:
        parts.append(section)
    return " ".join(parts) or (cite.get("source_path") or "citation")
