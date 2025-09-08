# src/rag/retriever/bm25f.py
# -*- coding: utf-8 -*-
"""
BM25F retriever with finance-friendly tokenization, fielded scoring (title/body),
inverted index acceleration, meta filters, soft prior boosts (section/year/phrases),
and better snippets.

CLI (example):
python -m src.rag.retriever.bm25 `
  --q "What are Apple’s main sources of revenue in its 2023 annual report?" `
  --ticker AAPL --form 10-K --year 2023 `
  --topk 8 `
  --index-dir data/index `
  --content-dir data/chunked `
  --w-title 0.5 --w-body 1.0
"""
from __future__ import annotations
import argparse, json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable, Tuple, Set
from collections import Counter, defaultdict

# ----------------- I/O utils -----------------
def load_metas(meta_path: Path) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            if isinstance(raw.get("meta"), dict):
                flat = dict(raw)
                inner = flat.pop("meta") or {}
                for k, v in inner.items():
                    flat.setdefault(k, v)
            else:
                flat = raw
            flat.setdefault("chunk_id", flat.get("chunk_id") or flat.get("id"))
            flat.setdefault("title", flat.get("title") or "")
            metas.append(flat)
    return metas

def _iter_jsonl_files(base: Path) -> List[Path]:
    if base.is_file():
        return [base]
    return [p for p in base.rglob("*.jsonl")]

def fetch_contents(content_path: Optional[Path], chunk_ids: Set[str]) -> Dict[str, str]:
    if not content_path:
        return {}
    found: Dict[str, str] = {}
    files = _iter_jsonl_files(content_path)
    targets = set([cid for cid in chunk_ids if cid])
    if not targets or not files:
        return {}
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    cid = obj.get("chunk_id") or obj.get("id") or obj.get("chunkId")
                    if cid in targets:
                        txt = obj.get("content") or obj.get("text") or obj.get("raw_text") or ""
                        if txt:
                            found[cid] = txt
                            targets.remove(cid)
                            if not targets:
                                return found
        except Exception:
            continue
    return found

# ----------------- tokenization -----------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9%$€¥£\.-]+", re.UNICODE)
STOPWORDS = {
    "the","a","an","and","or","of","in","to","for","on","by","as","at","is","are",
    "with","its","this","that","from","which","be","we","our","their","it","was",
    "were","has","have","had","but","not","no","can","could","would","should",
    "may","might","will","shall","into","than","then","there","here","also"
}

def tokenize(text: str, lower: bool = True, min_len: int = 2) -> List[str]:
    if not text:
        return []
    if lower:
        text = text.lower()
    toks = _TOKEN_RE.findall(text)
    toks = [t for t in toks if len(t) >= min_len and t not in STOPWORDS]
    return toks

# ----------------- snippets -----------------
def best_snippet(raw: str, q_terms: Set[str], win_chars: int = 700) -> str:
    txt = (raw or "").replace("\n", " ").strip()
    if len(txt) <= win_chars:
        return txt
    words = txt.split()
    # positions of query terms
    qset = {t.lower().strip(".,;:()") for t in q_terms if t}
    positions = [i for i, w in enumerate(words) if w.lower().strip(".,;:()") in qset]
    if not positions:
        return txt[:win_chars]
    # pick the densest 40-word window around a hit
    best_score, best_l = -1, 0
    for p in positions:
        L = max(0, p - 20); R = min(len(words), p + 20)
        score = sum(1 for w in words[L:R] if w.lower().strip(".,;:()") in qset)
        if score > best_score:
            best_score, best_l = score, L
    snippet = " ".join(words[best_l:best_l + 80])
    return snippet[:win_chars]

# ----------------- BM25F with inverted index -----------------
class BM25FIndex:
    """
    Fielded BM25 (title/body) with per-term postings to avoid full scans.
    docs: list of (title_tokens, body_tokens)
    """
    def __init__(self,
                 docs: List[Tuple[List[str], List[str]]],
                 k1: float = 1.5,
                 b_title: float = 0.20,
                 b_body: float = 0.75,
                 w_title: float = 0.5,
                 w_body: float = 1.0):
        self.k1 = float(k1)
        self.b_title = float(b_title)
        self.b_body = float(b_body)
        self.w_title = float(w_title)
        self.w_body = float(w_body)
        self.N = len(docs)

        self.title_len = [len(t) for t, _ in docs]
        self.body_len  = [len(b) for _, b in docs]
        self.avg_t = (sum(self.title_len) / self.N) if self.N else 0.0
        self.avg_b = (sum(self.body_len) / self.N) if self.N else 0.0

        # postings
        self.inv_t: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.inv_b: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # df over union of fields
        df = defaultdict(int)

        for i, (toks_t, toks_b) in enumerate(docs):
            ct, cb = Counter(toks_t), Counter(toks_b)
            for term, tf in ct.items():
                self.inv_t[term].append((i, tf))
            for term, tf in cb.items():
                self.inv_b[term].append((i, tf))
            # union for df
            terms_union = set(ct.keys()) | set(cb.keys())
            for term in terms_union:
                df[term] += 1

        # idf
        self.idf: Dict[str, float] = {
            term: math.log((self.N - dfi + 0.5) / (dfi + 0.5) + 1.0)
            for term, dfi in df.items()
        }

    def _norm_title(self, i: int) -> float:
        return self.k1 * (1 - self.b_title + self.b_title * (self.title_len[i] / (self.avg_t + 1e-9)))

    def _norm_body(self, i: int) -> float:
        return self.k1 * (1 - self.b_body + self.b_body * (self.body_len[i] / (self.avg_b + 1e-9)))

    def scores(self, q_tokens: List[str]) -> List[float]:
        if not q_tokens or not self.N:
            return [0.0] * self.N
        acc: Dict[int, float] = defaultdict(float)
        seen_terms: Set[str] = set()

        for term in q_tokens:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            idf = self.idf.get(term)
            if idf is None:
                continue

            # title postings
            for i, ft in self.inv_t.get(term, []):
                denom_t = ft + self._norm_title(i)
                contrib_t = self.w_title * (ft * (self.k1 + 1.0)) / (denom_t + 1e-12)
                acc[i] += idf * contrib_t

            # body postings
            for i, fb in self.inv_b.get(term, []):
                denom_b = fb + self._norm_body(i)
                contrib_b = self.w_body * (fb * (self.k1 + 1.0)) / (denom_b + 1e-12)
                acc[i] += idf * contrib_b

        # dense to list
        out = [0.0] * self.N
        for i, v in acc.items():
            out[i] = float(v)
        return out

# ----------------- retriever heuristics -----------------
INCLUDE_RE = re.compile(
    r"(net\s+sales|revenue|sources\s+of\s+revenue|segment\s+information|"
    r"geographic\s+data|by\s+product|by\s+region|item\s+7\b|md&a|note\s+1?\d\b)",
    re.I
)
EXCLUDE_RE = re.compile(
    r"(report\s+of\s+independent|internal\s+control|certification|exhibit|"
    r"signatures?\b|cover\s+page|documents\s+incorporated)", re.I
)
REV_BY_PRODUCT_HINT = re.compile(
    r"(sources?\s+of\s+revenue|by\s+product|product\s+categor(?:y|ies))",
    re.I
)

def intent_prior_bonus(m: Dict[str, Any], query: str) -> float:
    """
    当查询更像“主要收入来源/按产品”时：
      - 提升 Note 2 / Revenue 相关标题
      - 轻微压低纯 Segment/Geographic 相关标题
    """
    head = " ".join(str(m.get(k, "")) for k in ("heading", "title", "section")).lower()
    if REV_BY_PRODUCT_HINT.search(query):
        if ("note 2" in head) or (" revenue" in head):   # 空格避免匹配“total revenue”一类噪点，可按需放宽
            return 0.25
        if ("segment information" in head) or ("geographic" in head):
            return -0.10
    return 0.0

def extract_phrases(q: str) -> List[str]:
    # things inside double quotes are treated as phrases
    return [m.group(1).strip().lower() for m in re.finditer(r'"([^"]+)"', q)]

def phrase_bonus(text: str, phrases: List[str], per_hit: float = 0.15, cap: float = 0.45) -> float:
    t = (text or "").lower()
    b = 0.0
    for p in phrases:
        if p and p in t:
            b += per_hit
    return min(cap, b)

def year_proximity_bonus(m: Dict[str, Any], target_year: Optional[int]) -> float:
    try:
        if not target_year:
            return 0.0
        # prefer doc_date year if present, else fy
        d = str(m.get("doc_date", ""))[:4]
        year = int(d) if len(d) == 4 and d.isdigit() else int(m.get("fy"))
        gap = abs(year - int(target_year))
        return 0.20 if gap == 0 else (0.10 if gap == 1 else 0.0)
    except Exception:
        return 0.0

def soft_section_boost(m: Dict[str, Any], body_text: str) -> float:
    head = " ".join(str(m.get(k, "")) for k in ("heading", "title", "section")).lower()
    bonus = 0.0
    if "segment information" in head or re.search(r"\bnote\s+1?\d\b", head):
        bonus += 0.25
    if "item 7" in head or "md&a" in head:
        bonus += 0.15
    if "net sales" in head or "revenue" in head:
        bonus += 0.15
    t = (body_text or "").lower()
    if re.search(r"\bnet\s+sales\b", t):
        bonus += 0.10
    if sum(bool(re.search(p, t)) for p in [r"\biphone\b", r"\bmac\b", r"\bipad\b", r"\bwearables\b", r"\bservices\b"]) >= 2:
        bonus += 0.10
    return min(0.60, bonus)  # cap

# ----------------- config & retriever -----------------
@dataclass
class BM25FConfig:
    index_dir: str = "data/index"
    meta: Optional[str] = None
    content_path: Optional[str] = None
    content_dir: Optional[str] = None
    # tokenization
    min_token_len: int = 2
    # BM25F params
    k1: float = 1.5
    b_title: float = 0.20
    b_body: float = 0.75
    w_title: float = 2.5
    w_body: float = 1.0

class BM25FRetriever:
    def __init__(self, cfg: BM25FConfig):
        self.cfg = cfg
        meta_path = Path(cfg.meta) if cfg.meta else Path(cfg.index_dir) / "meta.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.jsonl not found at {meta_path}")
        self.metas = load_metas(meta_path)

        if cfg.content_path:
            self.content_root = Path(cfg.content_path)
        elif cfg.content_dir:
            self.content_root = Path(cfg.content_dir)
        else:
            self.content_root = None

    # helpers
    def _apply_meta_filters(self, ticker: Optional[str], form: Optional[str], year: Optional[int]) -> List[Dict[str, Any]]:
        tick = ticker.upper() if ticker else None
        frm = form.upper() if form else None
        yr = year
        out: List[Dict[str, Any]] = []
        for m in self.metas:
            if tick and str(m.get("ticker", "")).upper() != tick:
                continue
            if frm and str(m.get("form", "")).upper() != frm:
                continue
            if yr is not None:
                try:
                    if int(m.get("fy")) != int(yr):
                        continue
                except Exception:
                    # tolerate missing/invalid fy when year is provided
                    pass
            if not m.get("chunk_id"):
                continue
            out.append(m)
        return out

    def _expand_query(self, q: str, ticker: Optional[str]) -> str:
        ql = q.lower(); extra: List[str] = []
        if ("revenue" in ql) or ("sources of revenue" in ql) or ("source of revenue" in ql):
            extra += ["net sales", "by product", "by region", "segment information", "geographic data"]
        if ticker and ticker.upper() == "AAPL":
            extra += ["iphone", "mac", "ipad", "wearables", "services"]
        return q + " " + " ".join(extra)

    # public API
    def search(self, query: str, topk: int = 8,
               ticker: Optional[str] = None, form: Optional[str] = None, year: Optional[int] = None) -> List[Dict[str, Any]]:
        candidates = self._apply_meta_filters(ticker, form, year)
        if not candidates:
            return []

        # optional heading prefilter
        pre = [m for m in candidates
               if INCLUDE_RE.search(" ".join(str(m.get(k, "")) for k in ("heading", "title", "section")))
               and not EXCLUDE_RE.search(" ".join(str(m.get(k, "")) for k in ("heading", "title", "section")))]
        if pre:
            candidates = pre

        if not self.content_root:
            raise ValueError("BM25F needs text: set BM25FConfig.content_dir or content_path.")

        need_ids = {m["chunk_id"] for m in candidates}
        id2txt = fetch_contents(self.content_root, need_ids)

        # build docs
        doc_metas: List[Dict[str, Any]] = []
        docs_tokens: List[Tuple[List[str], List[str]]] = []
        raw_texts: List[str] = []
        titles_raw: List[str] = []

        for m in candidates:
            cid = m["chunk_id"]
            raw = id2txt.get(cid, "")
            if not raw:
                continue
            title_str = " ".join(str(m.get(k, "")) for k in ("heading", "title", "section")).strip()
            t_tokens = tokenize(title_str, lower=True, min_len=self.cfg.min_token_len)
            b_tokens = tokenize(raw, lower=True, min_len=self.cfg.min_token_len)

            if not (t_tokens or b_tokens):
                continue

            doc_metas.append(m)
            docs_tokens.append((t_tokens, b_tokens))
            raw_texts.append(raw)
            titles_raw.append(title_str)

        if not docs_tokens:
            return []

        # build BM25F index
        bm25f = BM25FIndex(
            docs_tokens,
            k1=self.cfg.k1,
            b_title=self.cfg.b_title,
            b_body=self.cfg.b_body,
            w_title=self.cfg.w_title,
            w_body=self.cfg.w_body,
        )

        # query tokens + phrases
        q_expanded = self._expand_query(query, ticker)
        q_tokens = tokenize(q_expanded, lower=True, min_len=self.cfg.min_token_len)
        phrases = extract_phrases(query)

        base_scores = bm25f.scores(q_tokens)

        # soft boosts (section/year/phrases)
        results: List[Tuple[int, float]] = []  # (idx, score)
        q_terms_for_snippet = set(q_tokens)
        for i, base in enumerate(base_scores):
            if base <= 0.0:
                continue
            m = doc_metas[i]
            body_text = raw_texts[i]
            bonus = 0.0
            bonus += soft_section_boost(m, body_text)
            bonus += phrase_bonus(body_text, phrases, per_hit=0.15, cap=0.45)
            bonus += year_proximity_bonus(m, year)
            bonus += intent_prior_bonus(m, query) 
            bonus = min(0.80, bonus)
            score = base * (1.0 + bonus)
            results.append((i, score))

        if not results:
            # fall back to base scores if boosts removed everything
            results = [(i, s) for i, s in enumerate(base_scores) if s > 0.0]
            if not results:
                return []

        results.sort(key=lambda x: x[1], reverse=True)
        unique = {}
        for i, s in results:  # results 已按分数倒序
            cid = doc_metas[i].get("chunk_id")
            if cid not in unique:   # 第一次出现的就是最高分
                unique[cid] = (i, s)
        results = list(unique.values())[:max(1, topk)]
        results = results[:max(1, topk)]

        out: List[Dict[str, Any]] = []
        for rank, (i, s) in enumerate(results, 1):
            m = doc_metas[i]
            raw = raw_texts[i]
            snippet = best_snippet(raw, q_terms_for_snippet, win_chars=500)
            out.append({
                "rank": rank,
                "score": float(s),
                "chunk_id": m.get("chunk_id"),
                "meta": m,
                "snippet": snippet,
                "source": "bm25f",
            })
        return out

# ----------------- CLI -----------------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--content-path", default=None)
    ap.add_argument("--content-dir", default=None)
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ticker"); ap.add_argument("--form"); ap.add_argument("--year", type=int)
    # tokenization
    ap.add_argument("--min-token-len", type=int, default=2)
    # BM25F params
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b-title", type=float, default=0.20)
    ap.add_argument("--b-body", type=float, default=0.75)
    ap.add_argument("--w-title", type=float, default=2.5)
    ap.add_argument("--w-body", type=float, default=1.0)

    args = ap.parse_args()

    cfg = BM25FConfig(
        index_dir=args.index_dir, meta=args.meta,
        content_path=args.content_path, content_dir=args.content_dir,
        min_token_len=args.min_token_len,
        k1=args.k1, b_title=args.b_title, b_body=args.b_body,
        w_title=args.w_title, w_body=args.w_body
    )
    retr = BM25FRetriever(cfg)
    hits = retr.search(args.query, topk=args.topk, ticker=args.ticker, form=args.form, year=args.year)

    if not hits:
        print("[INFO] No hits.")
        return
    print(f"Query: {args.query}\n" + "="*80)
    for r in hits:
        m = r["meta"]
        heading = " ".join(str(m.get(k,"")) for k in ("heading","title","section"))
        print(f"[{r['rank']:02d}] score={r['score']:.6f} | {m.get('ticker')} {m.get('fy')} {m.get('form')} "
              f"| chunk={m.get('chunk_index')} | id={r['chunk_id']}")
        print(f"     heading: {heading[:120]}")
        print(f"     {r['snippet']}")
        print("-"*80)

if __name__ == "__main__":
    _cli()
