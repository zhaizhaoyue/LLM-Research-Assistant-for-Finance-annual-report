#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunk text/fact corpora and build FAISS (IP/cosine) indices for BGE embeddings.

Key features:
- Text chunking: preserve source metadata (ticker/accno/form/fy/fq/page_no/section/source_path/...),
  add text_preview, emb_text.
- Fact grouping: normalize edges, bubble representative fields to top-level (fy/fq/section/page_no/source_path/accno/form/period_*),
  keep edges, aggregate fys/fqs.
- Build index: write rich meta.jsonl, including aggregated edge_* tags and representative fields,
  so retrievers can filter by year/fy/fq/form and citations can jump by page/section.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
from collections import Counter

import faiss  # index IO for building phase


# ============== Basic IO ==============

def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n

def strip_html(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = (s.replace("&nbsp;", " ")
           .replace("&amp;", "&")
           .replace("&lt;", "<")
           .replace("&gt;", ">")
           .replace("&quot;", '"')
           .replace("&#39;", "'"))
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============== Tokenizer (BGE) ==============

def load_bge_tokenizer(model_name: str = "BAAI/bge-base-en-v1.5"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)

def count_tokens(tok, text: str) -> int:
    if not text:
        return 0
    return len(tok.encode(text, add_special_tokens=False))

def split_by_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n\s*\n", t)
    out = []
    for p in paras:
        p = p.strip()
        if p:
            out.append(p)
    return out

def merge_paras_to_windows(paras: List[str], tok, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Pack paragraphs into token-budgeted windows, keeping a small overlap."""
    if not paras:
        return []

    windows: List[str] = []
    cur_buf: List[str] = []
    cur_tok = 0

    def push_cur():
        nonlocal windows, cur_buf, cur_tok
        if cur_buf:
            windows.append("\n\n".join(cur_buf))
            cur_buf, cur_tok = [], 0

    for p in paras:
        ptok = count_tokens(tok, p)
        # Hard long paragraph: slice by words approximately
        if ptok > max_tokens:
            push_cur()
            words = p.split(" ")
            chunk, chunk_tok = [], 0
            for w in words:
                wtok = count_tokens(tok, (w if not chunk else " " + w))
                if chunk_tok + wtok <= max_tokens:
                    chunk.append(w)
                    chunk_tok += wtok
                else:
                    if chunk:
                        windows.append(" ".join(chunk))
                    # start new chunk
                    chunk, chunk_tok = [w], count_tokens(tok, w)
            if chunk:
                windows.append(" ".join(chunk))
            continue

        add_tok = count_tokens(tok, ("\n\n" if cur_buf else "") + p)
        if cur_tok + add_tok <= max_tokens:
            cur_buf.append(p)
            cur_tok += add_tok
        else:
            # emit current window
            push_cur()
            # overlap: bring tail of last paragraph as seed
            # (approximate by last 'overlap_tokens' worth of tokens)
            if overlap_tokens > 0 and windows:
                tail = windows[-1].split()  # rough approx by words
                keep: List[str] = []
                ttok = 0
                for w in reversed(tail):
                    wtok = count_tokens(tok, (" " + w if keep else w))
                    if ttok + wtok <= overlap_tokens:
                        keep.append(w); ttok += wtok
                    else:
                        break
                keep.reverse()
                if keep:
                    cur_buf = [" ".join(keep)]
                    cur_tok = count_tokens(tok, cur_buf[0])
            # start with current paragraph
            cur_buf.append(p)
            cur_tok += count_tokens(tok, p)

    if cur_buf:
        windows.append("\n\n".join(cur_buf))
    return windows


# ============== Text chunking ==============

TEXT_FIELDS = ("text", "content", "text_clean", "text_raw", "paragraph")

def chunk_text_file(in_file: Path, out_file: Path, tok, *, max_tokens: int, overlap_tokens: int) -> int:
    paras: List[str] = []
    first_meta: Optional[Dict[str, Any]] = None
    accno_fallback: Optional[str] = None

    for row in read_jsonl(in_file):
        if first_meta is None:
            first_meta = dict(row)
        if not accno_fallback and row.get("accno"):
            accno_fallback = row.get("accno")

        txt = None
        for k in TEXT_FIELDS:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                txt = v
                break
        if not txt:
            continue

        clean = strip_html(txt)
        for p in split_by_paragraphs(clean):
            paras.append(p)

    if not paras:
        return write_jsonl(out_file, [])

    windows = merge_paras_to_windows(paras, tok, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    base = accno_fallback or in_file.stem
    out_rows: List[Dict[str, Any]] = []

    # carry through as many useful source keys as possible
    CARRY_KEYS = {
        "ticker","accno","cik","company_name",
        "form","filing_date","report_date",
        "fy","fq","year","period_start","period_end","instant",
        "source_path","source_file","page_no","section","item","linkrole",
        "concept","qname","language","schema_version"
    }

    for i, body in enumerate(windows):
        o = dict(first_meta or {})
        # ensure keys are present on top-level if available
        for k in list(CARRY_KEYS):
            if k not in o and first_meta and k in first_meta:
                o[k] = first_meta[k]

        o["schema_version"] = o.get("schema_version") or "0.3.0"
        o["language"] = o.get("language") or "en"

        o["chunk_id"] = f"{base}::text::chunk-{i}"
        o["chunk_index"] = i
        o["chunk_count"] = len(windows)

        o["text"] = body
        o["emb_text"] = body
        o["text_preview"] = body[:480]
        o["file_type"] = "text"
        o["source_file"] = o.get("source_path") or str(in_file)

        out_rows.append(o)

    return write_jsonl(out_file, out_rows)


# ============== Fact grouping ==============

def normalize_fact_record(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)

    per = ((r.get("context") or {}).get("period") or {})
    for src, dst in (("start_date", "period_start"),
                     ("end_date", "period_end"),
                     ("instant", "instant")):
        if per.get(src) is not None:
            out[dst] = per.get(src)

    fy = out.get("fy_norm") or out.get("fy")
    fq = out.get("fq_norm") or out.get("fq")
    if fy is not None:
        try:
            out["fy_norm"] = int(fy)
        except Exception:
            out["fy_norm"] = fy
    if fq is not None:
        fq_str = str(fq).upper()
        m = re.search(r"Q([1-4])", fq_str)
        out["fq_norm"] = (f"Q{m.group(1)}" if m else ("FY" if "FY" in fq_str else fq_str))

    concept = (out.get("concept") or out.get("qname") or "").strip()
    out["concept"] = concept

    # TextBlock -> keep clean text in value_display
    if concept.lower().endswith("textblock"):
        val = out.get("value") or out.get("value_display") or ""
        out["value_display"] = strip_html(val)[:5000] if isinstance(val, str) else str(val)

    # original fy/fq fallback
    if "fy" in r and out.get("fy") is None:
        try:
            out["fy"] = int(r.get("fy"))
        except Exception:
            out["fy"] = r.get("fy")
    if "fq" in r and out.get("fq") is None:
        out["fq"] = r.get("fq")

    # rag_text for embedding
    if not out.get("rag_text"):
        parts = [
            out.get("concept") or "",
            str(out.get("value_display") or out.get("value") or out.get("value_raw") or ""),
            f"(FY{out.get('fy_norm') or out.get('fy')}{' ' + str(out.get('fq_norm') or out.get('fq')) if (out.get('fq_norm') or out.get('fq')) else ''})",
            (f"instant {out.get('instant')}" if out.get("instant") else f"{out.get('period_start')}→{out.get('period_end')}"),
        ]
        out["rag_text"] = " ".join([p for p in parts if p]).strip()

    return out

def estimate_fact_tokens(tok, r: Dict[str, Any]) -> int:
    s = r.get("rag_text")
    if not s:
        s = "|".join([
            str(r.get("concept") or r.get("qname") or ""),
            str(r.get("value_display") or r.get("value_raw") or r.get("value") or ""),
            str(r.get("period_start") or "") + "–" + str(r.get("period_end") or r.get("instant") or ""),
        ])
    return max(8, count_tokens(tok, s))

def _mode(values: List[Any]) -> Any:
    vals = [v for v in values if v not in (None, "", [], {})]
    return Counter(vals).most_common(1)[0][0] if vals else None

def fact_group_chunks(in_file: Path, out_file: Path, tok, *, group_size: int, token_budget: int) -> int:
    rows = [normalize_fact_record(r) for r in read_jsonl(in_file)]
    if not rows:
        return write_jsonl(out_file, [])

    # bucket by linkrole
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        lr = r.get("linkrole") or "__NA__"
        buckets.setdefault(lr, []).append(r)

    out_rows: List[Dict[str, Any]] = []
    groups_all: List[List[Dict[str, Any]]] = []

    # pack groups by token budget
    for _, bucket in buckets.items():
        cur, cur_tok = [], 0
        for r in bucket:
            etok = estimate_fact_tokens(tok, r)
            if cur and (cur_tok + etok > token_budget or len(cur) >= group_size):
                groups_all.append(cur)
                cur, cur_tok = [], 0
            cur.append(r); cur_tok += etok
        if cur:
            groups_all.append(cur)

    total = len(groups_all)
    base = (rows[0].get("accno") or in_file.stem)

    for gidx, g in enumerate(groups_all):
        # headline
        tickers = sorted({r.get("ticker") for r in g if r.get("ticker")})
        forms   = sorted({r.get("form") for r in g if r.get("form")})
        years   = sorted({r.get("year") for r in g if r.get("year") is not None})
        lr      = g[0].get("linkrole") or "__NA__"
        top_concepts = []
        for r in g[:10]:
            c = r.get("concept") or r.get("qname")
            if c: top_concepts.append(str(c))
        title = f"[FACT GROUP] {','.join(tickers) or 'NA'} {','.join(forms) or 'NA'} {','.join(map(str,years)) or 'NA'} | {lr} | {', '.join(top_concepts[:6])}"

        # emb_text
        lines = [title]
        for r in g[:40]:
            t = r.get("rag_text") or ""
            if t: lines.append(t)
        emb_text = "\n".join(lines)

        # aggregate/bubble representative fields
        source_paths = [r.get("source_path") for r in g if r.get("source_path")]
        page_nos     = [r.get("page_no") for r in g if r.get("page_no") is not None]
        sections     = [r.get("section") or r.get("item") for r in g if (r.get("section") or r.get("item"))]
        concepts_all = [r.get("concept") or r.get("qname") for r in g if (r.get("concept") or r.get("qname"))]
        units        = [r.get("unit") for r in g if r.get("unit")]
        currs        = [r.get("currency") for r in g if r.get("currency")]
        period_starts= [r.get("period_start") for r in g if r.get("period_start")]
        period_ends  = [r.get("period_end") for r in g if r.get("period_end")]
        instants     = [r.get("instant") for r in g if r.get("instant")]
        forms_list   = [r.get("form") for r in g if r.get("form")]
        accnos_list  = [r.get("accno") for r in g if r.get("accno")]
        filing_dates = [r.get("filing_date") or r.get("report_date") for r in g if (r.get("filing_date") or r.get("report_date"))]

        # years/quarters
        fys = sorted({ (rr.get("fy_norm") if rr.get("fy_norm") is not None else rr.get("fy"))
                       for rr in g if (rr.get("fy_norm") is not None or rr.get("fy") is not None) })
        fqs = sorted({ (str(rr.get("fq_norm") or rr.get("fq")).upper())
                       for rr in g if (rr.get("fq_norm") or rr.get("fq")) })

        rep = {
            "source_path": source_paths[0] if source_paths else None,
            "page_no": min(page_nos) if page_nos else None,
            "section": _mode(sections),
            "concept_primary": concepts_all[0] if concepts_all else None,
            "unit_primary": _mode(units),
            "currency_primary": _mode(currs),
            "period_start": _mode(period_starts),
            "period_end": _mode(period_ends),
            "instant": _mode(instants),
            "form_primary": _mode(forms_list),
            "accno": accnos_list[0] if accnos_list else None,
            "filing_date": _mode(filing_dates),
            "fy": (fys[0] if fys else None),
            "fq": (fqs[0] if fqs else None),
        }

        out = {
            "chunk_id": f"{base}::fact::group::{gidx}",
            "chunk_index": gidx,
            "chunk_count": total,
            "file_type": "fact",
            "source_file": str(in_file),

            # bubbled representative fields
            "source_path": rep["source_path"],
            "page_no": rep["page_no"],
            "section": rep["section"],
            "accno": rep["accno"],
            "form": rep["form_primary"],
            "filing_date": rep["filing_date"],
            "fy": rep["fy"],
            "fq": rep["fq"],
            "period_start": rep["period_start"],
            "period_end": rep["period_end"],
            "instant": rep["instant"],

            "edges": g,                        # keep all normalized records
            "summary_text": title,             # title/headline
            "emb_text": emb_text,              # embedding text

            "tickers": sorted({r.get("ticker") for r in g if r.get("ticker")}),
            "forms":   sorted({r.get("form") for r in g if r.get("form")}),
            "years":   sorted({r.get("year") for r in g if r.get("year") is not None}),
            "fys":     [x for x in fys if x is not None],
            "fqs":     [x for x in fqs if x],
            "accnos":  sorted(set(accnos_list)),
            "linkroles": [lr],
            "edge_count": len(g),

            # helpful for BM25/filter/debug
            "concept_primary": rep["concept_primary"],
            "concepts_top": concepts_all[:6],
            "unit_primary": rep["unit_primary"],
            "currency_primary": rep["currency_primary"],

            "schema_version": "0.3.0",
            "language": g[0].get("language") or "en",
        }
        out_rows.append(out)

    return write_jsonl(out_file, out_rows)


# ============== Directory traversal & mirroring ==============

FILE_POLICIES = {
    "text_corpus.jsonl": ("text", "text_chunks.jsonl"),
    "text.jsonl":        ("text", "text_chunks.jsonl"),
    "fact.jsonl":        ("fact", "fact_chunks.jsonl"),
    "calculation_edges.jsonl": ("copy", "calc_chunks.jsonl"),
    "definition_arcs.jsonl":   ("copy", "def_chunks.jsonl"),
    "labels.jsonl":      ("copy", "labels.jsonl"),
}

def chunk_command(args) -> None:
    inp = Path(args.input_root).resolve()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    tok = load_bge_tokenizer(args.model_name)

    stats = {"text_files":0, "fact_files":0, "copied":0, "unknown":0, "chunks_written":0}
    for f in inp.rglob("*.jsonl"):
        if not f.is_file():
            continue
        rel_dir = f.relative_to(inp).parent
        out_dir = (out / rel_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        kind, out_name = FILE_POLICIES.get(f.name, (None, None))
        if kind == "text":
            n = chunk_text_file(
                f, out_dir / out_name, tok,
                max_tokens=args.text_max_tokens,
                overlap_tokens=args.text_overlap
            )
            stats["text_files"] += 1
            stats["chunks_written"] += n
        elif kind == "fact":
            n = fact_group_chunks(
                f, out_dir / out_name, tok,
                group_size=args.fact_group_size,
                token_budget=args.fact_token_budget
            )
            stats["fact_files"] += 1
            stats["chunks_written"] += n
        elif kind == "copy":
            shutil.copy2(f, out_dir / out_name)
            stats["copied"] += 1
        else:
            shutil.copy2(f, out_dir / f.name)
            stats["unknown"] += 1

    print(json.dumps({"summary": stats}, ensure_ascii=False, indent=2))


# ============== Build index (text/fact) ==============

def _iter_rows_filtered(chunks_root: Path, include_types: set[str]):
    for jf in chunks_root.rglob("*.jsonl"):
        for row in read_jsonl(jf):
            ft = (row.get("file_type") or "").lower()
            if not ft:
                name = jf.name.lower()
                if "text_chunks" in name:
                    ft = "text"
                elif "fact_chunks" in name:
                    ft = "fact"
                elif "calc_chunks" in name:
                    ft = "cal"
                elif "def_chunks" in name:
                    ft = "def"
            if ft in include_types:
                yield row

def _make_embedder(model_name: str):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device).eval()

    use_amp = (device.type == "cuda")

    def embed(texts: List[str]) -> np.ndarray:
        enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with torch.no_grad():
            if use_amp:
                amp_dtype = torch.bfloat16 if mdl.dtype == torch.bfloat16 else torch.float16
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = mdl(**enc)
            else:
                out = mdl(**enc)

            token_emb = out.last_hidden_state            # [B, T, D]
            attn = enc["attention_mask"].unsqueeze(-1).float()
            summed = (token_emb * attn).sum(dim=1)       # [B, D]
            counts = attn.sum(dim=1).clamp(min=1e-9)
            vec = (summed / counts).detach().cpu().numpy()

        import numpy as np
        vec = np.ascontiguousarray(vec, dtype="float32")
        faiss.normalize_L2(vec)  # cosine via IP
        return vec

    if device.type == "cuda":
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    return embed

def _build_single_index(
    chunks_root: Path,
    model_name: str,
    include_types: set[str],
    index_out: Path,
    meta_out: Path,
    batch_size: int = 64,
):
    import numpy as np

    embed = _make_embedder(model_name)

    meta_out.parent.mkdir(parents=True, exist_ok=True)
    if meta_out.exists():
        meta_out.unlink()

    index = None
    buf_texts: List[str] = []
    buf_metas: List[dict] = []

    def flush():
        nonlocal index, buf_texts, buf_metas
        if not buf_texts:
            return
        embs = embed(buf_texts)
        if index is None:
            d = embs.shape[1]
            index = faiss.IndexFlatIP(d)  # cosine with normalized vectors
        assert embs.dtype == np.float32 and embs.flags["C_CONTIGUOUS"]
        index.add(embs)
        with meta_out.open("a", encoding="utf-8") as mf:
            for m in buf_metas:
                mf.write(json.dumps(m, ensure_ascii=False) + "\n")
        buf_texts, buf_metas = [], []

    n_rows = 0
    for row in _iter_rows_filtered(chunks_root, include_types):
        # choose embedding text
        emb_text = row.get("emb_text") or row.get("text") or row.get("summary_text") or ""
        if not emb_text.strip():
            continue

        # fy/fq extraction with fallbacks
        fy = row.get("fy")
        if fy is None:
            fy = row.get("fy_norm")
        if fy is None:
            ys = row.get("years") or row.get("fys")
            if isinstance(ys, (list, tuple)) and ys:
                fy = ys[0]
        try:
            fy = int(fy) if fy is not None else None
        except Exception:
            pass

        fq = row.get("fq") or row.get("fq_norm")
        if fq is None:
            fqs = row.get("fqs")
            if isinstance(fqs, (list, tuple)) and fqs:
                fq = fqs[0]
        fq = (str(fq).upper() if fq is not None else None)

        # preview text (short)
        text_preview = None
        if (row.get("file_type") == "text"):
            text_preview = (row.get("text_preview") or (row.get("text") or "")[:480])
        else:
            text_preview = (row.get("summary_text") or "")[:480]

        meta = {
            "chunk_id": row.get("chunk_id"),
            "file_type": (row.get("file_type") or "").lower() or None,

            "ticker": row.get("ticker") or (row.get("tickers") or [None])[0],
            "form": row.get("form") or (row.get("forms") or [None])[0],
            "accno": row.get("accno") or (row.get("accnos") or [None])[0],
            "source_path": row.get("source_path") or row.get("source_file"),

            # for filtering
            "year": row.get("year") or (row.get("years") or [None])[0],
            "fy": fy,
            "fq": fq,

            # navigation
            "page_no": row.get("page_no"),
            "section": row.get("section") or row.get("item"),
            "linkrole": row.get("linkrole") or (row.get("linkroles") or [None])[0],
            "filing_date": row.get("filing_date") or row.get("report_date"),
            "period_start": row.get("period_start"),
            "period_end": row.get("period_end"),
            "instant": row.get("instant"),

            # preview
            "text_preview": text_preview,
        }

        # --- aggregate labels from edges if present (for fact groups) ---
        if isinstance(row.get("edges"), list) and row["edges"]:
            es = row["edges"]

            concepts   = [e.get("concept") or e.get("qname") for e in es if (e.get("concept") or e.get("qname"))]
            units      = [e.get("unit") for e in es if e.get("unit")]
            currs      = [e.get("currency") for e in es if e.get("currency")]
            forms_e    = [e.get("form") for e in es if e.get("form")]
            linkroles  = [e.get("linkrole") for e in es if e.get("linkrole")]
            sections   = [e.get("section") or e.get("item") for e in es if (e.get("section") or e.get("item"))]
            pages      = [e.get("page_no") for e in es if e.get("page_no") is not None]

            fys_e, fqs_e, pstarts, pends, instants = [], [], [], [], []
            for e in es:
                fy_e = e.get("fy_norm", e.get("fy"))
                if fy_e is not None:
                    try: fy_e = int(fy_e)
                    except Exception: pass
                    fys_e.append(fy_e)
                fq_e = str(e.get("fq_norm", e.get("fq"))).upper() if (e.get("fq_norm") or e.get("fq")) else None
                if fq_e: fqs_e.append(fq_e)
                if e.get("period_start"): pstarts.append(e["period_start"])
                if e.get("period_end"):   pends.append(e["period_end"])
                if e.get("instant"):      instants.append(e["instant"])

            # collections (capped to avoid huge meta)
            meta.update({
                "edge_concepts":   sorted(list({c for c in concepts}))[:50],
                "edge_units":      sorted(list({u for u in units}))[:20],
                "edge_currencies": sorted(list({c for c in currs}))[:10],
                "edge_fys":        sorted(list({y for y in fys_e if y is not None}))[:20],
                "edge_fqs":        sorted(list({q for q in fqs_e}))[:10],
                "edge_forms":      sorted(list({f for f in forms_e}))[:10],
                "edge_linkroles":  sorted(list({lr for lr in linkroles}))[:30],
                "edge_sections":   sorted(list({s for s in sections}))[:50],
                "edge_pages":      sorted(list({int(p) for p in pages}))[:100],
                "edge_period_starts": sorted(list({p for p in pstarts}))[:50],
                "edge_period_ends":   sorted(list({p for p in pends}))[:50],
                "edge_instants":      sorted(list({p for p in instants}))[:50],
                # representative (fallbacks)
                "concept_primary": meta.get("concept_primary") or _mode(concepts),
                "unit_primary":    meta.get("unit_primary") or _mode(units),
                "currency_primary":meta.get("currency_primary") or _mode(currs),
                "fy":              meta.get("fy") or _mode(fys_e),
                "fq":              meta.get("fq") or _mode(fqs_e),
                "section":         meta.get("section") or _mode(sections),
                "page_no":         meta.get("page_no") or (_mode(pages) if pages else None),
                "form":            meta.get("form") or _mode(forms_e),
                "linkrole":        meta.get("linkrole") or _mode(linkroles),
            })

        buf_texts.append(emb_text)
        buf_metas.append(meta)
        n_rows += 1

        if len(buf_texts) >= batch_size:
            flush()

    flush()

    if index is None:
        raise RuntimeError("No rows to index (check --file-types and chunk files).")

    index_out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_out))

    print(json.dumps({
        "index": str(index_out),
        "meta": str(meta_out),
        "file_types": sorted(include_types),
        "ntotal": index.ntotal,
        "consumed_rows": n_rows
    }, ensure_ascii=False, indent=2))

def build_index_command(args) -> None:
    chunks_root = Path(args.chunks).resolve()
    if not chunks_root.exists():
        raise FileNotFoundError(f"{chunks_root} not found")

    if args.file_types:
        include = {t.strip().lower() for t in args.file_types.split(",")}
    else:
        include = {"text", "fact"}
    include -= {"cal", "def"}

    _build_single_index(
        chunks_root=chunks_root,
        model_name=args.model_name,
        include_types=include,
        index_out=Path(args.index_out),
        meta_out=Path(args.meta_out),
        batch_size=args.batch_size,
    )

def build_two_indices_command(args) -> None:
    root = Path(args.chunks).resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")

    _build_single_index(
        chunks_root=root,
        model_name=args.model_name,
        include_types={"text"},
        index_out=Path(args.index_out_text),
        meta_out=Path(args.meta_out_text),
        batch_size=args.batch_size,
    )
    _build_single_index(
        chunks_root=root,
        model_name=args.model_name,
        include_types={"fact"},
        index_out=Path(args.index_out_fact),
        meta_out=Path(args.meta_out_fact),
        batch_size=args.batch_size,
    )


# ============== CLI ==============

def main():
    ap = argparse.ArgumentParser(description="Chunk folder and build IP Faiss index for BGE.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # chunk
    ap_chunk = sub.add_parser("chunk", help="Traverse input folder and write mirrored chunks")
    ap_chunk.add_argument("--input-root", required=True)
    ap_chunk.add_argument("--output-root", required=True)
    ap_chunk.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    ap_chunk.add_argument("--text-max-tokens", type=int, default=350)
    ap_chunk.add_argument("--text-overlap", type=int, default=48)
    ap_chunk.add_argument("--fact-group-size", type=int, default=20)
    ap_chunk.add_argument("--fact-token-budget", type=int, default=350)
    ap_chunk.set_defaults(func=chunk_command)

    # single index
    ap_index = sub.add_parser("build-index", help="Build a single global IP Faiss index")
    ap_index.add_argument("--chunks", required=True)
    ap_index.add_argument("--index-out", required=True)
    ap_index.add_argument("--meta-out", required=True)
    ap_index.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    ap_index.add_argument("--batch-size", type=int, default=64)
    ap_index.add_argument("--file-types", default="text,fact", help="comma list: text,fact,cal,def")
    ap_index.set_defaults(func=build_index_command)

    # dual indices
    ap_dual = sub.add_parser("build-two-indices", help="Build two indices: text & fact")
    ap_dual.add_argument("--chunks", required=True)
    ap_dual.add_argument("--index-out-text", required=True)
    ap_dual.add_argument("--meta-out-text", required=True)
    ap_dual.add_argument("--index-out-fact", required=True)
    ap_dual.add_argument("--meta-out-fact", required=True)
    ap_dual.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    ap_dual.add_argument("--batch-size", type=int, default=64)
    ap_dual.set_defaults(func=build_two_indices_command)

    args = ap.parse_args()

    # ensure fresh meta for index builds
    if args.cmd == "build-index":
        mp = Path(args.meta_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        if mp.exists():
            mp.unlink()
    elif args.cmd == "build-two-indices":
        for p in [args.meta_out_text, args.meta_out_fact]:
            mp = Path(p)
            mp.parent.mkdir(parents=True, exist_ok=True)
            if mp.exists():
                mp.unlink()

    args.func(args)

if __name__ == "__main__":
    main()




'''
python src/chunking_and_embedding/chunking_and_embedding.py chunk `
  --input-root data/silver `
  --output-root data/chunked `
  --text-max-tokens 350 `
  --text-overlap 48 `
  --fact-group-size 20 `
  --fact-token-budget 350
'''



'''
python src/chunking_and_embedding/chunking_and_embedding.py build-two-indices `
  --chunks data/chunked `
  --index-out-text data/index/ip_bge_text.faiss `
  --meta-out-text  data/index/ip_bge_text.meta.jsonl `
  --index-out-fact data/index/ip_bge_fact.faiss `
  --meta-out-fact  data/index/ip_bge_fact.meta.jsonl

  '''