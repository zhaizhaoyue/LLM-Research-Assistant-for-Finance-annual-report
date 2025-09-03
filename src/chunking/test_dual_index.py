#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for dual indices (text + fact).
"""

import faiss, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

INDEX_TEXT = "data/index/ip_bge_text.faiss"
META_TEXT  = "data/index/ip_bge_text.meta.jsonl"
INDEX_FACT = "data/index/ip_bge_fact.faiss"
META_FACT  = "data/index/ip_bge_fact.meta.jsonl"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


# ---- load index + meta ----
def load_index_and_meta(index_path, meta_path):
    index = faiss.read_index(index_path)
    metas = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
    assert len(metas) == index.ntotal
    return index, metas


index_text, metas_text = load_index_and_meta(INDEX_TEXT, META_TEXT)
index_fact, metas_fact = load_index_and_meta(INDEX_FACT, META_FACT)


# ---- embedder (mean pooling + L2 norm) ----
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
mdl = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
mdl.eval()

def embed(texts):
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
        tok_emb = out.last_hidden_state
        attn = enc["attention_mask"].unsqueeze(-1).float()
        vec = (tok_emb * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
        vec = vec.cpu().numpy().astype("float32", copy=False)
    faiss.normalize_L2(vec)
    return vec


# ---- hybrid search ----
def search(query, topk_text=5, topk_fact=5):
    qv = embed([query])
    st, it = index_text.search(qv, topk_text)
    sf, if_ = index_fact.search(qv, topk_fact)

    print("=== Text hits ===")
    for s, i in zip(st[0], it[0]):
        m = metas_text[i]
        print(f"{s:.3f}", m.get("ticker"), m.get("form"), m.get("year"), m.get("source_path"))

    print("\n=== Fact hits ===")
    for s, i in zip(sf[0], if_[0]):
        m = metas_fact[i]
        print(f"{s:.3f}", m.get("ticker"), m.get("form"), m.get("year"),
              m.get("chunk_id"), m.get("accno"))


if __name__ == "__main__":
    q = "How did foreign exchange rates impact revenue in 2023?"
    search(q, topk_text=5, topk_fact=5)
