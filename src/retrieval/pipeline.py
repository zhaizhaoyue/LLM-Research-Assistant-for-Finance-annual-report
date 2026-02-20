
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.retrieval.retriever.bm25_text import BM25TextConfig
from src.retrieval.retriever.dense import DenseRetriever
from src.retrieval.retriever.hybrid import HybridRetrieverRRF, CrossEncoderReranker
from src.retrieval.retriever.answer_api import LLMClient, answer_with_llm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query-side NER: extract entities from user query for filter/boost
# ---------------------------------------------------------------------------
_query_ner = None


def _extract_query_entities(query: str) -> dict:
    """Extract entities from user query for filter/boost.

    Returns dict with keys: orgs, dates, gpes, tickers, fin_metrics
    """
    global _query_ner
    result: dict = {"orgs": [], "dates": [], "gpes": [], "tickers": [], "fin_metrics": []}

    entities = []
    try:
        if _query_ner is None:
            from src.ner.recognizer import NERRecognizer
            _query_ner = NERRecognizer(device=-1, batch_size=1)
        entities = _query_ner.recognize(query)
    except Exception as e:
        log.debug("Query NER model unavailable: %s", e)

    # Also extract custom finance entities
    try:
        from src.ner.fin_entities import extract_custom_entities
        entities.extend(extract_custom_entities(query))
    except Exception:
        pass

    for ent in entities:
        label = ent.get("label", "")
        text = ent.get("text", "")
        if label == "ORG":
            result["orgs"].append(text)
        elif label == "DATE":
            result["dates"].append(text)
        elif label == "GPE":
            result["gpes"].append(text)
        elif label == "TICKER":
            result["tickers"].append(text)
        elif label == "FIN_METRIC":
            result["fin_metrics"].append(text)
    return result


@dataclass
class QueryRequest:
    query: str
    index_dir: Path
    content_dir: Optional[Path] = None
    bm25_meta: Optional[Path] = None
    dense_model: str = "BAAI/bge-base-en-v1.5"
    dense_device: str = "cpu"
    topk: int = 8
    bm25_topk: int = 200
    dense_topk: int = 200
    ce_candidates: int = 256
    rrf_k: float = 60.0
    rrf_w_bm25: float = 2.0
    rrf_w_dense: float = 2.0
    ce_weight: float = 0.5
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: Optional[str] = None
    ticker: Optional[str] = None
    form: Optional[str] = None
    year: Optional[int] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    max_context_tokens: int = 2400
    strict_filters: bool = True


@dataclass
class QueryResult:
    query: str
    records: List[Dict[str, Any]]
    answer: Optional[Dict[str, Any]]
    query_entities: Optional[Dict[str, list]] = None


def _build_hybrid(req: QueryRequest) -> HybridRetrieverRRF:
    bm25_cfg = BM25TextConfig(
        index_dir=str(req.index_dir),
        meta=str(req.bm25_meta) if req.bm25_meta else None,
        content_dir=str(req.content_dir) if req.content_dir else None,
        content_path=str(req.content_dir) if req.content_dir and req.content_dir.is_file() else None,
    )

    dense = DenseRetriever(
        index_dir=str(req.index_dir),
        model=req.dense_model,
        device=req.dense_device,
    )

    reranker: Optional[CrossEncoderReranker] = None
    if req.ce_weight > 1e-6:
        reranker = CrossEncoderReranker(
            model_name=req.rerank_model,
            device=req.rerank_device,
        )

    return HybridRetrieverRRF(
        bm25_cfg=bm25_cfg,
        dense=dense,
        reranker=reranker,
        k=req.rrf_k,
        w_bm25=req.rrf_w_bm25,
        w_dense=req.rrf_w_dense,
        ce_weight=req.ce_weight,
    )


def run_query(req: QueryRequest) -> QueryResult:
    index_dir = req.index_dir
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    content_dir = req.content_dir
    if content_dir is not None and not content_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {content_dir}")

    # Extract entities from user query
    query_entities = _extract_query_entities(req.query)

    hybrid = _build_hybrid(req)

    filters: Dict[str, Any] = {}
    if req.ticker:
        filters["ticker"] = req.ticker
    if req.form:
        filters["form"] = req.form
    if req.year:
        filters["year"] = req.year

    records = hybrid.search(
        req.query,
        topk=req.topk,
        content_dir=str(content_dir) if content_dir else None,
        bm25_topk=req.bm25_topk,
        dense_topk=req.dense_topk,
        ce_candidates=req.ce_candidates,
        strict_filters=req.strict_filters,
        **filters,
    )

    # Apply entity overlap boost to records
    if query_entities and any(query_entities.values()):
        for rec in records:
            meta = rec.get("meta", {}) or {}
            boost = _entity_overlap_boost(query_entities, meta)
            if boost > 0 and "final_score" in rec:
                rec["final_score"] = rec["final_score"] * (1.0 + boost)

    answer: Optional[Dict[str, Any]] = None
    if req.llm_base_url and req.llm_model and req.llm_api_key:
        llm = LLMClient(
            base_url=req.llm_base_url,
            model=req.llm_model,
            api_key=req.llm_api_key,
        )
        answer = answer_with_llm(
            req.query,
            records,
            llm,
            max_ctx_tokens=req.max_context_tokens,
        )

    return QueryResult(query=req.query, records=records, answer=answer,
                       query_entities=query_entities)


def _entity_overlap_boost(query_entities: dict, chunk_meta: dict) -> float:
    """Boost score based on entity overlap between query and chunk."""
    boost = 0.0
    query_orgs = {o.upper() for o in query_entities.get("orgs", [])}
    chunk_orgs = set(chunk_meta.get("entities_org", []))
    if query_orgs & chunk_orgs:
        boost += 0.15  # org match +15%

    query_gpes = {g.upper() for g in query_entities.get("gpes", [])}
    chunk_gpes = set(chunk_meta.get("entities_gpe", []))
    if query_gpes & chunk_gpes:
        boost += 0.05  # geo match +5%

    query_tickers = {t.upper() for t in query_entities.get("tickers", [])}
    chunk_tickers = set(chunk_meta.get("entities_ticker", []))
    if query_tickers & chunk_tickers:
        boost += 0.10  # ticker match +10%

    return boost
