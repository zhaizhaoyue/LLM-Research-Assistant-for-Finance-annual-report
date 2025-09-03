# src/qa/retriever_hybrid.py  (你也可以仍叫原文件名)
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime
import faiss
import numpy as np

# Optional BM25
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None  # noqa: N816

from sentence_transformers import SentenceTransformer

REVENUE_CONCEPTS = {
    # US GAAP 常见
    "us-gaap:SalesRevenueNet",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:Revenues",
    # 发行人自定义里偶见的 “NetSales” 等（保守加入）
    "aapl:NetSales",
    "aapl:Revenue",  # 只是兜底，真命中不多
    # IFRS 兼容
    "ifrs-full:Revenue",
}

REVENUE_CONCEPT_RE = re.compile(
    r"^(?:us-gaap:(?:SalesRevenueNet|RevenueFromContractWithCustomer(?:ExcludingAssessedTax)?|Revenues)"
    r"|ifrs-full:Revenue"
    r"|[a-z0-9-]+:(?:NetSales|Revenue)s?)$",
    re.IGNORECASE,
)

ANNUAL_FQ_TOKENS = {"FY", "Q4", "FQ4", "4", "Y"}
_YEAR_RE = re.compile(r"(\d{4})")


#-----------------------------时间过滤--------------------------------------------------------------
def apply_numeric_filters_with_fallback(hits: List[Dict[str, Any]], year: int, form: str) -> List[Dict[str, Any]]:
    # Step 1: 严格三步（form → 年份+年度口径 → 概念白名单）
    step1 = filter_by_form(hits, form=form)
    step1 = filter_by_year_and_period(step1, target_year=year, require_annual_like=True)
    step1 = filter_by_revenue_concepts(step1)
    if step1:
        return step1

    # Step 2: 放宽年度口径（FY/Q4/FQ4/4 失败时，允许没有 fq 或 duration 异常）
    step2 = filter_by_form(hits, form=form)
    step2 = filter_by_year_and_period(step2, target_year=year, require_annual_like=False)
    step2 = filter_by_revenue_concepts(step2)
    if step2:
        return step2

    # Step 3: 仅按 form+year（不做概念白名单），让后续“文本/表格兜底”还能接住
    step3 = filter_by_form(hits, form=form)
    step3 = filter_by_year_and_period(step3, target_year=year, require_annual_like=False)
    if step3:
        return step3

    # Step 4（最终兜底）：只按 form，保留给 numeric_demo 的 fallback
    step4 = filter_by_form(hits, form=form)
    return step4


def _to_year(hit: Dict[str, Any]) -> Optional[int]:
    # 1) 先看 fy
    fy = hit.get("fy")
    if fy is not None:
        # int 直接返回
        if isinstance(fy, int):
            return fy
        # 字符串里若含4位数字（如 "2023"、"FY2023"）抽出来
        if isinstance(fy, str):
            m = _YEAR_RE.search(fy)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass

    # 2) 看 years 列表（你们的 fact 常见这个）
    years = hit.get("years")
    if isinstance(years, (list, tuple)) and years:
        # 取第一个能转成int的
        for y in years:
            try:
                return int(y)
            except Exception:
                # y 可能是 "2023" 字符串
                if isinstance(y, str):
                    m = _YEAR_RE.search(y)
                    if m:
                        try:
                            return int(m.group(1))
                        except Exception:
                            pass

    # 3) 从日期字段推断
    for k in ("period_end", "instant", "period_start"):
        v = hit.get(k)
        if not v:
            continue
        m = _YEAR_RE.search(str(v))
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue

    # 都没有则 None
    return None


def _is_annual_like(hit: Dict[str, Any]) -> bool:
    """把 FY、Q4、FQ4、4 统一视作年度口径；其他情况尽量再看 duration（如果你们有时长字段可以补上）。"""
    fq = str(hit.get("fq") or "").upper()
    if fq in ANNUAL_FQ_TOKENS:
        return True
    # 可选：若你们有 duration（天数）字段，可把 ~360±30 天视作年度
    dur_days = hit.get("duration_days") or hit.get("duration")
    try:
        if dur_days and 330 <= int(dur_days) <= 400:
            return True
    except Exception:
        pass
    return False

def filter_by_year_and_period(hits: List[Dict[str, Any]], target_year: int, require_annual_like: bool = True) -> List[Dict[str, Any]]:
    out = []
    for h in hits:
        y = _to_year(h)
        if y != target_year:
            continue
        if require_annual_like and not _is_annual_like(h):
            continue
        out.append(h)
    return out

def filter_by_form(hits: List[Dict[str, Any]], form: Optional[str]) -> List[Dict[str, Any]]:
    if not form:
        return hits
    f = form.upper()
    return [h for h in hits if str(h.get("form") or "").upper() == f]

def filter_by_revenue_concepts(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for h in hits:
        c = h.get("concept")
        if not c:
            continue
        if c in REVENUE_CONCEPTS or REVENUE_CONCEPT_RE.match(c):
            out.append(h)
    return out

#----------------------------------------------------------------------------------------------------
def _tokenize_en(s: str) -> List[str]:
    """English/number tokenization only."""
    return re.findall(r"[a-z0-9$%\.]+", (s or "").lower())


class _OneIndex:
    """A single FAISS index + meta + optional BM25 built on the same corpus."""
    def __init__(self, name: str, faiss_path: Path, meta_path: Path, enc: SentenceTransformer):
        self.name = name
        self.index = faiss.read_index(str(faiss_path))
        with meta_path.open("r", encoding="utf-8") as f:
            self.metas: List[Dict[str, Any]] = [json.loads(line) for line in f]
        if self.index.ntotal != len(self.metas):
            raise ValueError(f"[{name}] index.ntotal={self.index.ntotal} != metas={len(self.metas)}")

        self.metric_type = getattr(self.index, "metric_type", faiss.METRIC_INNER_PRODUCT)
        self.enc = enc

        # BM25 build (optional)
        self._bm25 = None
        self._bm25_meta_ids: List[int] = []
        if BM25Okapi is not None:
            docs: List[List[str]] = []
            for i, m in enumerate(self.metas):
                t = m.get("text") or m.get("raw_text") or m.get("text_preview") or ""
                toks = _tokenize_en(t)
                if not toks:
                    continue
                self._bm25_meta_ids.append(i)
                docs.append(toks)
            if docs:
                self._bm25 = BM25Okapi(docs)

    def _encode(self, q: str) -> np.ndarray:
        v = self.enc.encode([q], normalize_embeddings=True, show_progress_bar=False)
        return v.astype("float32")

    def _score_from_dist(self, dist: float) -> float:
        # For IP, faiss returns similarity; for L2 use 1/(1+dist)
        if self.metric_type == faiss.METRIC_INNER_PRODUCT:
            return float(dist)
        return 1.0 / (1.0 + float(dist))

    def _passes_filter(self, m: Dict[str, Any], ticker: Optional[str], year: Optional[int], form: Optional[str]) -> bool:
        if ticker and str(m.get("ticker", "")).upper() != ticker.upper():
            return False
        if year is not None:
            fy = m.get("fy")
            if fy is None or int(fy) != int(year):
                return False
        if form and str(m.get("form", "")).upper() != form.upper():
            return False
        return True

    def _to_item(self, meta_idx: int, score: float) -> Dict[str, Any]:
        m = self.metas[meta_idx]
        text = m.get("text") or m.get("raw_text") or m.get("text_preview") or ""
        return {
            "chunk_id": m.get("chunk_id") or f"{m.get('accno','NA')}::{m.get('file_type','NA')}::idx::{meta_idx}",
            "score": float(score),
            "snippet": (text or "")[:600],
            "meta": m,
            "faiss_id": meta_idx,
        }

    def search_dense(self, query: str, top_k: int, ticker: Optional[str], year: Optional[int], form: Optional[str]) -> List[Dict[str, Any]]:
        qv = self._encode(query)
        D, I = self.index.search(qv, max(top_k * 4, top_k))
        out: List[Dict[str, Any]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = self.metas[idx]
            if not self._passes_filter(m, ticker, year, form):
                continue
            out.append(self._to_item(idx, self._score_from_dist(dist)))
            if len(out) >= top_k:
                break
        return out

    def search_bm25(self, query: str, top_k: int, ticker: Optional[str], year: Optional[int], form: Optional[str]) -> List[Dict[str, Any]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize_en(query))  # ndarray (len == len(_bm25_meta_ids))
        if np.all(scores == 0):
            return []
        order = np.argsort(scores)[::-1]
        out: List[Dict[str, Any]] = []
        for local_i in order[: max(top_k * 4, top_k)]:
            meta_i = self._bm25_meta_ids[local_i]
            m = self.metas[meta_i]
            if not self._passes_filter(m, ticker, year, form):
                continue
            out.append(self._to_item(meta_i, float(scores[local_i])))
            if len(out) >= top_k:
                break
        return out


class HybridRetriever:
    """
    Multi-index hybrid retriever:
      - Loads two indices: ip_bge_fact.faiss + ip_bge_fact.meta.jsonl, ip_bge_text.faiss + ip_bge_text.meta.jsonl
      - Dense + (optional) BM25 per index
      - RRF fusion across dense & BM25, and across indexes if target="both"
    """
    def __init__(
        self,
        index_dir: str | Path,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda",
    ):
        self.index_dir = Path(index_dir)

        # encoder (auto-fallback to CPU if device not available)
        try:
            self.enc = SentenceTransformer(model_name, device=device)
        except Exception:
            self.enc = SentenceTransformer(model_name, device="cpu")

        self.dim = self.enc.get_sentence_embedding_dimension()

        self.idxs: Dict[str, _OneIndex] = {}
        for name in ("fact", "text"):
            fpath = self.index_dir / f"ip_bge_{name}.faiss"
            mpath = self.index_dir / f"ip_bge_{name}.meta.jsonl"
            if fpath.exists() and mpath.exists():
                self.idxs[name] = _OneIndex(name, fpath, mpath, self.enc)

        if not self.idxs:
            raise FileNotFoundError(f"No index loaded under {self.index_dir}")

        # optional: global meta (not required)
        self.global_meta: List[Dict[str, Any]] = []
        gmeta = self.index_dir / "ip_bge.meta.jsonl"
        if gmeta.exists():
            with gmeta.open("r", encoding="utf-8") as f:
                self.global_meta = [json.loads(line) for line in f]

    # ----------------- Public API -----------------
    def search_hybrid(
        self,
        query: str,
        top_k: int = 50,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        form: Optional[str] = None,
        rr_k_dense: int = 100,
        rr_k_bm25: int = 100,
        target: str = "both",  # "fact" | "text" | "both"
    ) -> List[Dict[str, Any]]:
        """Return unified hits: [{chunk_id, score, snippet, meta, faiss_id}, ...]"""
        targets = [target] if target in ("fact", "text") else [k for k in ("fact", "text") if k in self.idxs]

        # collect candidates per target
        fused_all: List[Dict[str, Any]] = []
        for t in targets:
            idx = self.idxs[t]
            dense = idx.search_dense(query, rr_k_dense, ticker, year, form)
            bm25 = idx.search_bm25(query, rr_k_bm25, ticker, year, form)

            # RRF within this target
            fused = self._rrf_fuse(dense, bm25)
            # 标注来源，便于调试
            for it in fused:
                it.setdefault("meta", {}).setdefault("_source_index", t)
            fused_all.extend(fused)

        # if both, fuse again across targets
        if len(targets) > 1:
            fused_all = self._rrf_fuse(fused_all, [])  # 单次 RRF 以 rank 融合
        # sort by score desc and trim
        fused_all.sort(key=lambda x: x["score"], reverse=True)
        return fused_all[:top_k]

    # ----------------- Internal -----------------
    @staticmethod
    def _rrf_fuse(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]], c: float = 60.0) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion on two ranked lists.
        """
        def _key(item: Dict[str, Any], fallback_rank: int):
            m = item.get("meta", {}) or {}
            return item.get("faiss_id") or m.get("_id") or item.get("chunk_id") or fallback_rank

        rank_a: Dict[Any, int] = { _key(it, i+1): i+1 for i, it in enumerate(primary) }
        rank_b: Dict[Any, int] = { _key(it, i+1): i+1 for i, it in enumerate(secondary) }
        ids = set(rank_a.keys()) | set(rank_b.keys())

        # need a way to map key back to item; keep best one we saw
        by_key: Dict[Any, Dict[str, Any]] = {}
        for i, it in enumerate(primary):
            by_key[_key(it, i+1)] = it
        for i, it in enumerate(secondary):
            by_key.setdefault(_key(it, i+1), it)

        fused: List[Dict[str, Any]] = []
        for k in ids:
            score = 0.0
            if k in rank_a:
                score += 1.0 / (c + rank_a[k])
            if k in rank_b:
                score += 1.0 / (c + rank_b[k])
            x = dict(by_key[k])
            x["score"] = float(score)
            fused.append(x)

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused

# ------------------------------
# Singleton wrapper
# ------------------------------
_INDEX_DIR = "data/index"
_retriever_singleton: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = HybridRetriever(index_dir=_INDEX_DIR)
    return _retriever_singleton

def hybrid_search(query: str, filters: Dict[str, Any], topk: int = 8, target: str = "both") -> List[Dict[str, Any]]:
    r = get_retriever()
    return r.search_hybrid(
        query=query,
        top_k=topk,
        ticker=filters.get("ticker"),
        year=filters.get("year"),
        form=str(filters.get("form") or "").upper() if filters.get("form") else None,
        target=target,
    )


