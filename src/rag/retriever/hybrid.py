# retriever/hybrid.py
# -*- coding: utf-8 -*-
"""
Hybrid retriever: BM25 + Dense fusion (z-score), returns unified top-K.
- Uses BM25Retriever from bm25.py
- Uses DenseRetriever from dense.py if available; otherwise can accept a custom dense callable.

CLI (example):
python -m src.rag.retriever.hybrid \
  --q "What are Apple’s main sources of revenue in its 2023 annual report?" \
  --ticker AAPL --form 10-K --year 2023 \
  --topk 10 \
  --index-dir data/index \
  --content-dir data/chunked \
  --alpha 0.6 \
  --normalize
"""
from __future__ import annotations
import argparse, math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from .bm25 import BM25Retriever, BM25Config

# 可选导入 DenseRetriever；如果不存在，Hybrid 也能只跑 BM25
try:
    from .dense import DenseRetriever  # 你自己的 dense 检索类（若存在）
except Exception:
    DenseRetriever = None  # type: ignore

def zscore(xs: List[float]) -> List[float]:
    if not xs: return xs
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
    sd = math.sqrt(var) or 1.0
    return [(x - m) / sd for x in xs]

@dataclass
class HybridConfig:
    # BM25
    index_dir: str = "data/index"
    meta: Optional[str] = None
    content_path: Optional[str] = None
    content_dir: Optional[str] = None
    k1: float = 1.5
    b: float = 0.75
    min_token_len: int = 1
    title_weight: float = 3.0
    # Dense (与 dense.py 一致的参数名；若 DenseRetriever 不存在会忽略)
    faiss: Optional[str] = None
    model: str = "BAAI/bge-base-en-v1.5"
    device: str = "cuda"
    normalize: bool = False
    # Fusion
    alpha: float = 0.6      # 权重：alpha*BM25 + (1-alpha)*Dense
    bm25_k: int = 64        # 单独各取多少候选再融合
    dense_k: int = 64

class HybridRetriever:
    def __init__(self, cfg: HybridConfig):
        self.cfg = cfg
        self.bm25 = BM25Retriever(BM25Config(
            index_dir=cfg.index_dir, meta=cfg.meta,
            content_path=cfg.content_path, content_dir=cfg.content_dir,
            k1=cfg.k1, b=cfg.b, min_token_len=cfg.min_token_len,
            title_weight=cfg.title_weight
        ))
        self.dense = None
        if DenseRetriever is not None:
            # 这里假定你的 dense.py 暴露 DenseRetriever(init with paths)
            self.dense = DenseRetriever(
                index_dir=cfg.index_dir, faiss_path=cfg.faiss, meta_path=cfg.meta,
                model=cfg.model, device=cfg.device, normalize=cfg.normalize
            )

    def _dense_search(self, query: str, topk: int, ticker: Optional[str], form: Optional[str], year: Optional[int]) -> List[Dict[str, Any]]:
        if self.dense is None:
            return []
        # 这里假设 DenseRetriever 有 search(...) 返回 list[dict]，字段包含 chunk_id/meta/score/snippet
        try:
            return self.dense.search(query, topk=topk, ticker=ticker, form=form, year=year)
        except TypeError:
            # 如果你的 DenseRetriever 接口不同，可在这里适配
            return self.dense.search(query, topk)  # type: ignore

    def search(self, query: str, topk: int = 10,
               ticker: Optional[str] = None, form: Optional[str] = None, year: Optional[int] = None) -> List[Dict[str, Any]]:
        bm_hits = self.bm25.search(query, topk=max(topk, self.cfg.bm25_k), ticker=ticker, form=form, year=year)
        dn_hits = self._dense_search(query, topk=max(topk, self.cfg.dense_k), ticker=ticker, form=form, year=year)

        # 合并 by chunk_id
        by_id: Dict[str, Dict[str, Any]] = {}
        for h in bm_hits:
            cid = h["chunk_id"]
            d = by_id.setdefault(cid, dict(h))
            d["_bm25"] = float(h["score"])
        for h in dn_hits:
            cid = h["chunk_id"]
            d = by_id.setdefault(cid, dict(h))
            d["_dense"] = float(h["score"])

        ids = list(by_id.keys())
        bm = [by_id[i].get("_bm25", float("nan")) for i in ids]
        dn = [by_id[i].get("_dense", float("nan")) for i in ids]
        bm = [0.0 if math.isnan(x) else x for x in bm]
        dn = [0.0 if math.isnan(x) else x for x in dn]
        bm_z, dn_z = zscore(bm), zscore(dn)

        fused: List[Tuple[float, Dict[str, Any]]] = []
        for i, cid in enumerate(ids):
            score = self.cfg.alpha * bm_z[i] + (1 - self.cfg.alpha) * dn_z[i]
            d = by_id[cid]; d["fused_score"] = float(score)
            fused.append((score, d))

        fused.sort(key=lambda t: t[0], reverse=True)
        out = []
        for rank, (_, d) in enumerate(fused[:topk], 1):
            d["rank"] = rank
            out.append(d)
        return out

# ---------------- CLI ----------------
def _cli():
    ap = argparse.ArgumentParser()
    # common
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--ticker"); ap.add_argument("--form"); ap.add_argument("--year", type=int)

    # bm25
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--content-path", default=None)
    ap.add_argument("--content-dir", default=None)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--min-token-len", type=int, default=1)
    ap.add_argument("--title-weight", type=float, default=3.0)

    # dense
    ap.add_argument("--faiss", default=None)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normalize", action="store_true")

    # fusion
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--bm25-k", type=int, default=64)
    ap.add_argument("--dense-k", type=int, default=64)

    args = ap.parse_args()

    cfg = HybridConfig(
        index_dir=args.index_dir, meta=args.meta, content_path=args.content_path, content_dir=args.content_dir,
        k1=args.k1, b=args.b, min_token_len=args.min_token_len, title_weight=args.title_weight,
        faiss=args.faiss, model=args.model, device=args.device, normalize=args.normalize,
        alpha=args.alpha, bm25_k=args.bm25_k, dense_k=args.dense_k
    )
    retr = HybridRetriever(cfg)
    hits = retr.search(args.query, topk=args.topk, ticker=args.ticker, form=args.form, year=args.year)
    if not hits:
        print("[INFO] No hits.")
        return

    print(f"Query: {args.query}\n" + "="*80)
    for h in hits:
        m = h.get("meta", {})
        snippet = h.get("snippet") or (m.get("title") or "")
        print(f"[{h['rank']:02d}] fused={h['fused_score']:.4f} "
              f"(bm25={h.get('_bm25',0):.4f}, dense={h.get('_dense',0):.4f}) "
              f"| {m.get('ticker')} {m.get('fy')} {m.get('form')} | id={h.get('chunk_id')}")
        if snippet:
            print("     ", str(snippet)[:240].replace("\n"," "))
        print("-"*80)

if __name__ == "__main__":
    _cli()
