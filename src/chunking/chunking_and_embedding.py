#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, shutil, sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import faiss
# ============== 基本 IO ==============
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
    # 去除 HTML 标签
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    # 实体
    s = (s.replace("&nbsp;", " ")
           .replace("&amp;", "&")
           .replace("&lt;", "<")
           .replace("&gt;", ">")
           .replace("&quot;", '"')
           .replace("&#39;", "'"))
    # 压空白
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ============== 分词器（AutoTokenizer）===============
# 注意：只在需要时导入 transformers，避免无模型时报错
def load_bge_tokenizer(model_name: str = "BAAI/bge-base-en-v1.5"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok

def count_tokens(tok, text: str) -> int:
    if not text:
        return 0
    return len(tok.encode(text, add_special_tokens=False))

def split_by_paragraphs(text: str) -> List[str]:
    """按自然段切开（空行/双换行/行内硬换行聚合）"""
    if not text:
        return []
    # 统一换行
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # 以双换行分段；如果没有，就按单换行再合并
    paras = re.split(r"\n\s*\n", t)
    # 去空白
    out = []
    for p in paras:
        p = p.strip()
        if p:
            out.append(p)
    return out

def merge_paras_to_windows(paras: List[str], tok, max_tokens: int, overlap_tokens: int) -> List[str]:
    """尽量保持自然段，按 token 预算合并；相邻窗口保留 overlap_tokens 尾部"""
    if not paras:
        return []

    windows = []
    cur_buf, cur_tok = [], 0
    for p in paras:
        ptok = count_tokens(tok, p)
        # 单段超限：硬切
        if ptok > max_tokens:
            # 先把当前窗口吐出
            if cur_buf:
                windows.append("\n\n".join(cur_buf))
                cur_buf, cur_tok = [], 0
            # 按 token 长度slice这一个大段
            words = p.split(" ")
            # 粗略用词近似 token；安全起见逐句累加
            chunk, chunk_tok = [], 0
            for w in words:
                wtok = count_tokens(tok, (w if not chunk else " " + w))
                if chunk_tok + wtok <= max_tokens:
                    chunk.append(w)
                    chunk_tok += wtok
                else:
                    if chunk:
                        windows.append(" ".join(chunk))
                    # overlap
                    if overlap_tokens > 0 and windows:
                        tail = windows[-1]
                        # 用一个简化 overlap：直接把上个窗口末尾文本拼接一小段
                        # 为避免爆表，这里忽略 token 精确对齐，只保证有上下文
                        pass
                    chunk = [w]
                    chunk_tok = count_tokens(tok, w)
            if chunk:
                windows.append(" ".join(chunk))
            continue

        # 正常合并到当前窗口
        add_tok = count_tokens(tok, ("\n\n" if cur_buf else "") + p)
        if cur_tok + add_tok <= max_tokens:
            cur_buf.append(p)
            cur_tok += add_tok
        else:
            # 吐出当前窗口
            if cur_buf:
                windows.append("\n\n".join(cur_buf))
                # overlap：把上一窗口尾部段落尽量带一点进来
                if overlap_tokens > 0 and cur_buf:
                    tail = cur_buf[-1]
                    # 简化：保留尾段最后 overlap_tokens 预算的一截
                    tail_words = tail.split(" ")
                    keep = []
                    ttok = 0
                    for w in reversed(tail_words):
                        wtok = count_tokens(tok, (" " + w if keep else w))
                        if ttok + wtok <= overlap_tokens:
                            keep.append(w)
                            ttok += wtok
                        else:
                            break
                    keep = list(reversed(keep))
                    cur_buf = [" ".join(keep)] if keep else []
                    cur_tok = count_tokens(tok, cur_buf[0]) if keep else 0
                else:
                    cur_buf, cur_tok = [], 0
            # 新窗口从当前段开始
            cur_buf.append(p)
            cur_tok += count_tokens(tok, p)

    if cur_buf:
        windows.append("\n\n".join(cur_buf))
    return windows

# ============== 文本切块 ==============
TEXT_FIELDS = ("text", "content", "text_clean", "text_raw", "paragraph")

def chunk_text_file(in_file: Path, out_file: Path, tok, *, max_tokens: int, overlap_tokens: int) -> int:
    # 聚合整文件的自然段
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

        # 去 HTML（有些 text 抓自 HTML）
        clean = strip_html(txt)
        # 可能出现超长行的 PDF OCR：先按句号/换行软切为段
        for p in split_by_paragraphs(clean):
            paras.append(p)

    if not paras:
        return write_jsonl(out_file, [])

    windows = merge_paras_to_windows(paras, tok, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    base = accno_fallback or in_file.stem
    out_rows = []
    for i, body in enumerate(windows):
        o = dict(first_meta or {})
        o["schema_version"] = o.get("schema_version") or "0.3.0"
        o["language"] = o.get("language") or "en"
        o["chunk_id"] = f"{base}::text::chunk-{i}"
        o["chunk_index"] = i
        o["chunk_count"] = len(windows)
        o["text"] = body
        # embedding 专用字段
        o["emb_text"] = body
        o["file_type"] = "text"
        o["source_file"] = str(in_file)
        out_rows.append(o)

    return write_jsonl(out_file, out_rows)

# ============== fact 组块（统一模板 + 预算控制）=============
def normalize_fact_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """把 fact 统一成稳定模板；TextBlock 去 HTML 保留纯文本"""
    out = dict(r)
    # context.period → period_start / period_end / instant 优先
    per = ((r.get("context") or {}).get("period") or {})
    for src, dst in (("start_date", "period_start"),
                     ("end_date", "period_end"),
                     ("instant", "instant")):
        if per.get(src) is not None:
            out[dst] = per.get(src)

    # 统一期别标签
    fy = out.get("fy_norm") or out.get("fy")
    fq = out.get("fq_norm") or out.get("fq")
    if fy is not None:
        out["fy_norm"] = int(fy)
    if fq is not None:
        fq_str = str(fq).upper()
        m = re.search(r"Q([1-4])", fq_str)
        out["fq_norm"] = (f"Q{m.group(1)}" if m else ("FY" if "FY" in fq_str else None))

    concept = (out.get("concept") or out.get("qname") or "").strip()
    out["concept"] = concept

    # TextBlock 纯文本化
    if concept.lower().endswith("textblock"):
        val = out.get("value") or out.get("value_display") or ""
        out["value_display"] = strip_html(val)[:5000] if isinstance(val, str) else str(val)

    # 生成 rag_text 的兜底（便于 embedding）
    if not out.get("rag_text"):
        parts = [
            out.get("concept") or "",
            str(out.get("value_display") or out.get("value") or out.get("value_raw") or ""),
            f"(FY{out.get('fy_norm') or out.get('fy')}{' ' + str(out.get('fq_norm') or out.get('fq')) if (out.get('fq_norm') or out.get('fq')) else ''}",
            f"instant {out.get('instant')})" if out.get("instant") else f"{out.get('period_start')}→{out.get('period_end')})"
        ]
        out["rag_text"] = " ".join([p for p in parts if p]).strip()

    return out

def estimate_fact_tokens(tok, r: Dict[str, Any]) -> int:
    # 优先 rag_text，否则概念 + 值 + 期间
    s = r.get("rag_text")
    if not s:
        s = "|".join([
            str(r.get("concept") or r.get("qname") or ""),
            str(r.get("value_display") or r.get("value_raw") or r.get("value") or ""),
            str(r.get("period_label") or f"{r.get('period_start')}–{r.get('period_end') or r.get('instant')}")
        ])
    return max(8, count_tokens(tok, s))

def fact_group_chunks(in_file: Path, out_file: Path, tok, *, group_size: int, token_budget: int) -> int:
    rows = [normalize_fact_record(r) for r in read_jsonl(in_file)]
    if not rows:
        return write_jsonl(out_file, [])

    # 先按 linkrole 分桶（同财报片段一起）
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        lr = r.get("linkrole") or "__NA__"
        buckets.setdefault(lr, []).append(r)

    out_rows = []
    groups_all: List[List[Dict[str, Any]]] = []

    # 组内：按 token_budget 打包
    for lr, bucket in buckets.items():
        cur, cur_tok = [], 0
        for r in bucket:
            etok = estimate_fact_tokens(tok, r)
            # 如果当前不是空且加上会超预算：先封组
            if cur and (cur_tok + etok > token_budget or len(cur) >= group_size):
                groups_all.append(cur)
                cur, cur_tok = [], 0
            cur.append(r); cur_tok += etok
        if cur:
            groups_all.append(cur)

    total = len(groups_all)
    base = (rows[0].get("accno") or in_file.stem)

    for gidx, g in enumerate(groups_all):
        # 组块标题头（组级别摘要）：ticker/form/year/linkrole/概念TOP
        tickers = sorted({r.get("ticker") for r in g if r.get("ticker")})
        forms   = sorted({r.get("form") for r in g if r.get("form")})
        years   = sorted({r.get("year") for r in g if r.get("year") is not None})
        lr      = g[0].get("linkrole") or "__NA__"
        top_concepts = []
        for r in g[:10]:
            c = r.get("concept") or r.get("qname")
            if c: top_concepts.append(str(c))
        title = f"[FACT GROUP] {','.join(tickers) or 'NA'} {','.join(forms) or 'NA'} {','.join(map(str,years)) or 'NA'} | {lr} | {', '.join(top_concepts[:6])}"

        # emb_text：标题 + 每条 rag_text 的前若干行
        lines = [title]
        for r in g[:40]:
            t = r.get("rag_text") or ""
            if t: lines.append(t)
        emb_text = "\n".join(lines)

        out = {
            "chunk_id": f"{base}::fact::group::{gidx}",
            "chunk_index": gidx,
            "chunk_count": total,
            "file_type": "fact",
            "source_file": str(in_file),
            "edges": g,                        # 原始归一化记录集合
            "summary_text": title,             # 标题头
            "emb_text": emb_text,              # ✅ 专用于 embedding
            "tickers": sorted({r.get("ticker") for r in g if r.get("ticker")}),
            "forms":   sorted({r.get("form") for r in g if r.get("form")}),
            "years":   sorted({r.get("year") for r in g if r.get("year") is not None}),
            "accnos":  sorted({r.get("accno") for r in g if r.get("accno")}),
            "linkroles": [lr],
            "edge_count": len(g),
            "schema_version": "0.3.0",
            "language": g[0].get("language") or "en",
        }
        out_rows.append(out)

    return write_jsonl(out_file, out_rows)

# ============== 目录遍历 & 镜像 ==============
FILE_POLICIES = {
    "text_corpus.jsonl": ("text", "text_chunks.jsonl"),
    "text.jsonl":        ("text", "text_chunks.jsonl"),
    "fact.jsonl":        ("fact", "fact_chunks.jsonl"),
    "calculation_edges.jsonl": ("copy", "calc_chunks.jsonl"),       # 如需可改为 group
    "definition_arcs.jsonl":   ("copy", "def_chunks.jsonl"),        # 如需可改为 group
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
            # 不切块，直接镜像（或自行替换成 group 逻辑）
            shutil.copy2(f, out_dir / out_name)
            stats["copied"] += 1
        else:
            # 其它未知 *.jsonl：可以先复制，或者走 text/fact 的兜底
            shutil.copy2(f, out_dir / f.name)  # 先复制保留
            stats["unknown"] += 1

    print(json.dumps({"summary": stats}, ensure_ascii=False, indent=2))

# ============== 向量化 & 索引 ==============
# ============== Build index (text/fact) ==============
def _iter_rows_filtered(chunks_root: Path, include_types: set[str]):
    for jf in chunks_root.rglob("*.jsonl"):
        for row in read_jsonl(jf):
            ft = (row.get("file_type") or "").lower()
            # 兜底：某些旧管线没写 file_type，可用文件名猜测
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
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 13400F + NVIDIA，基本都会是 CUDA 可用
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    mdl.to(device)
    mdl.eval()

    # 尝试用更省显存的 dtype
    use_amp = (device.type == "cuda")

    def embed(texts: list[str]) -> np.ndarray:
        # 注意：tokenizer 在 CPU 即可
        enc = tok(
            texts, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        )
        # 把张量搬到 GPU
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with torch.no_grad():
            if use_amp:
                # bfloat16/float16 自动混合精度（根据模型 dtype）
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

        # 保证 C-contiguous + float32，再做 L2 归一化（IP=cos）
        vec = np.ascontiguousarray(vec, dtype="float32")
        faiss.normalize_L2(vec)
        return vec

    # 可选：减少 CPU 抢核
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
    import faiss

    embed = _make_embedder(model_name)

    # 清空旧 meta
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    if meta_out.exists():
        meta_out.unlink()

    index = None
    buf_texts: list[str] = []
    buf_metas: list[dict] = []

    def flush():
        nonlocal index, buf_texts, buf_metas
        if not buf_texts:
            return
        embs = embed(buf_texts)
        if index is None:
            d = embs.shape[1]
            index = faiss.IndexFlatIP(d)  # IP + L2 归一化 = 余弦
        # 强保证
        assert embs.dtype == np.float32 and embs.flags['C_CONTIGUOUS']
        index.add(embs)
        with meta_out.open("a", encoding="utf-8") as mf:
            for m in buf_metas:
                mf.write(json.dumps(m, ensure_ascii=False) + "\n")
        buf_texts, buf_metas = [], []

    n_rows = 0
    for row in _iter_rows_filtered(chunks_root, include_types):
        # 选择 embedding 文本
        emb_text = row.get("emb_text") or row.get("text") or row.get("summary_text") or ""
        if not emb_text.strip():
            continue

        meta = {
            "chunk_id": row.get("chunk_id"),
            "file_type": (row.get("file_type") or "").lower() or None,
            "accno": row.get("accno") or (row.get("accnos") or [None])[0],
            "source_path": row.get("source_path") or row.get("source_file"),
            "ticker": row.get("ticker") or (row.get("tickers") or [None])[0],
            "form": row.get("form") or (row.get("forms") or [None])[0],
            "year": row.get("year") or (row.get("years") or [None])[0],
            "linkrole": row.get("linkrole") or (row.get("linkroles") or [None])[0],
            "page_no": row.get("page_no"),
        }

        buf_texts.append(emb_text)
        buf_metas.append(meta)
        n_rows += 1

        if len(buf_texts) >= batch_size:
            flush()

    flush()  # 收尾

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

    # 解析 file types
    if args.file_types:
        include = {t.strip().lower() for t in args.file_types.split(",")}
    else:
        include = {"text", "fact"}        # 默认 text+fact
    # 明确跳过 calc/def
    include -= {"cal", "def"}

    _build_single_index(
        chunks_root=chunks_root,
        model_name=args.model_name,
        include_types=include,
        index_out=Path(args.index_out),
        meta_out=Path(args.meta_out),
        batch_size=args.batch_size,
    )

# 一次性构建 text + fact 两个索引
def build_two_indices_command(args) -> None:
    root = Path(args.chunks).resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")

    # text
    _build_single_index(
        chunks_root=root,
        model_name=args.model_name,
        include_types={"text"},
        index_out=Path(args.index_out_text),
        meta_out=Path(args.meta_out_text),
        batch_size=args.batch_size,
    )
    # fact
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
    ap = argparse.ArgumentParser(description="Chunk folder (mirror) and build IP Faiss index for BGE.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 你的 chunk 子命令（保留）
    ap_chunk = sub.add_parser("chunk", help="Traverse input folder and write mirrored chunks")
    ap_chunk.add_argument("--input-root", required=True)
    ap_chunk.add_argument("--output-root", required=True)
    ap_chunk.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    ap_chunk.add_argument("--text-max-tokens", type=int, default=350)
    ap_chunk.add_argument("--text-overlap", type=int, default=48)
    ap_chunk.add_argument("--fact-group-size", type=int, default=20)
    ap_chunk.add_argument("--fact-token-budget", type=int, default=350)
    ap_chunk.set_defaults(func=chunk_command)

    # 单索引
    ap_index = sub.add_parser("build-index", help="Build a single global IP Faiss index")
    ap_index.add_argument("--chunks", required=True)
    ap_index.add_argument("--index-out", required=True)
    ap_index.add_argument("--meta-out", required=True)
    ap_index.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    ap_index.add_argument("--batch-size", type=int, default=64)
    ap_index.add_argument("--file-types", default="text,fact", help="comma list: text,fact,cal,def")
    ap_index.set_defaults(func=build_index_command)

    # 双索引
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

    # 清空旧 meta（避免多次追加）
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
python src/chunking/chunking and embedding.py chunk `
  --input-root data/silver `
  --output-root data/chunked `
  --text-max-tokens 350 `
  --text-overlap 48 `
  --fact-group-size 20 `
  --fact-token-budget 350
'''



'''
python src/chunking/chunking_and_embedding.py build-two-indices `
  --chunks data/chunked `
  --index-out-text data/index/ip_bge_text.faiss `
  --meta-out-text  data/index/ip_bge_text.meta.jsonl `
  --index-out-fact data/index/ip_bge_fact.faiss `
  --meta-out-fact  data/index/ip_bge_fact.meta.jsonl

  '''