#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk everything except labels.jsonl:
- Mirror tree from <input_root> (default: data/silver) to <output_root> (default: data/chunked)
- text_corpus.jsonl / text.jsonl     -> token sliding-window chunks  -> *text_chunks.jsonl
- fact.jsonl                          -> group-by-N record chunks    -> *fact_chunks.jsonl
- calculation_edges.jsonl             -> group-by-N record chunks    -> *calc_chunks.jsonl
- definition_arcs.jsonl               -> group-by-N record chunks    -> *def_chunks.jsonl
- labels_best.jsonl / labels_wide.jsonl -> group-by-N record chunks  -> *labels_*_chunks.jsonl
- labels.jsonl                        -> copy as-is (no chunking)
- Unknown *.jsonl (not labels.jsonl)  -> default group-by-N record chunks -> *generic_chunks.jsonl
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# ----------------------------- Utilities -----------------------------
def discover_project_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__).resolve()).parent
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return p.parent.parent  # fallback: typical src/*/chunking.py

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

# ----------------------------- Chunkers -----------------------------
def _as_tokens(text: str) -> List[str]:
    # 简单健壮：按空白切分近似 token
    return text.split()

def chunk_text(text: str, max_tokens: int, overlap: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    返回:
      - chunks: List[str]
      - spans : List[(start, end)]  以 token 为单位的半开区间 [start, end)
    """
    toks = _as_tokens(text)
    if not toks:
        return [], []
    step = max(1, max_tokens - overlap)
    chunks: List[str] = []
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(toks):
        start = i
        end = min(i + max_tokens, len(toks))
        window = toks[start:end]
        if not window:
            break
        chunks.append(" ".join(window))
        spans.append((start, end))
        i += step
    return chunks, spans

TEXT_FIELDS = ("text", "content", "text_clean", "text_raw", "paragraph")

def process_text_jsonl(
    in_file: Path,
    out_file: Path,
    *,
    relpath_under_input: Optional[str],
    max_tokens: int,
    overlap: int,
    default_schema_version: str = "0.3.0",
    default_language: str = "en",
) -> int:
    """
    读取文本 JSONL，滑窗切块，透传/补全元数据，输出增强字段：
      - chunk_id: <file-stem>::row-<row_idx>::chunk-<i>
      - chunk_index, chunk_count
      - tokens_in_chunk, row_tokens_total, chunk_span_tokens: [start, end)
      - source_file (绝对/相对都可), relpath (相对 input_root)
    """
    out_rows: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(read_jsonl(in_file)):
        # 1) 取文本
        txt: Optional[str] = None
        text_key_used: Optional[str] = None
        for k in TEXT_FIELDS:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                txt = v
                text_key_used = k
                break
        if not txt:
            continue

        # 2) 切块 + 统计 token
        chunks, spans = chunk_text(txt, max_tokens=max_tokens, overlap=overlap)
        total_chunks = len(chunks)
        row_tokens_total = len(_as_tokens(txt))

        # 3) 通用元数据补全
        schema_version = row.get("schema_version") or default_schema_version
        language = row.get("language") or default_language

        # 4) 逐块写出（透传原始 metadata）
        for i, ch in enumerate(chunks):
            start, end = spans[i]
            o = dict(row)  # 透传一切已有字段（ticker/fy/fq/accno/section/heading/page_no/…）
            o.pop("tokens", None)
            o["schema_version"] = schema_version
            o["language"] = language

            # 用 chunk 覆盖文本，并写出增强字段
            o["text"] = ch
            o["chunk_index"] = i
            o["chunk_count"] = total_chunks
            o["chunk_id"] = f"{in_file.stem}::row-{row_idx}::chunk-{i}"
            o["source_file"] = str(in_file)
            if relpath_under_input is not None:
                o["relpath"] = relpath_under_input

            o["tokens_in_chunk"] = len(_as_tokens(ch))
            o["row_tokens_total"] = row_tokens_total
            o["chunk_span_tokens"] = [start, end]  # 方便日后重建上下文
            o["text_field"] = text_key_used       # 记录本次使用了哪个文本字段

            out_rows.append(o)

    return write_jsonl(out_file, out_rows)

def process_group_jsonl(
    in_file: Path,
    out_file: Path,
    *,
    group_size: int,
    relpath_under_input: Optional[str] = None,
    add_summary_text: bool = False,
    token_budget: Optional[int] = None,  # 新增：近似 token 上限（例如 350）
) -> int:
    """
    分两步：
      1) 先按 linkrole 分桶（同一报表片段放一起，语义更集中）
      2) 在每个桶中按 group_size 切块；若指定 token_budget，就依据估算的 token 数提前换组
    """
    rows = list(read_jsonl(in_file))
    if not rows:
        return write_jsonl(out_file, [])

    # -------- helpers --------
    def est_edge_tokens(r: Dict[str, Any]) -> int:
        # 非严格估算：parent/child 概念名 + 权重 + 少量上下文
        p = (r.get("parent_concept") or "")
        c = (r.get("child_concept") or "")
        w = str(r.get("weight") if r.get("weight") is not None else "")
        # 近似按空白分词
        return len(p.split()) + len(c.split()) + len(w.split()) + 4

    def pack_with_budget(bucket: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not token_budget:
            # 仅按 group_size
            return [bucket[i:i+group_size] for i in range(0, len(bucket), group_size)]
        groups: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        cur_tok = 0
        for r in bucket:
            e = est_edge_tokens(r)
            # 如果当前空且单条已超预算，也要容忍（避免死循环）
            if cur and (cur_tok + e) > token_budget:
                groups.append(cur)
                cur, cur_tok = [], 0
            cur.append(r)
            cur_tok += e
            if len(cur) >= group_size:
                groups.append(cur)
                cur, cur_tok = [], 0
        if cur:
            groups.append(cur)
        return groups

    # -------- 1) linkrole 分桶 --------
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        lr = r.get("linkrole") or "__NA__"
        buckets.setdefault(lr, []).append(r)

    # -------- 2) 每桶切块 + 汇总输出 --------
    out_rows: List[Dict[str, Any]] = []
    # 全局计数（用于 chunk_count；先累计所有组数）
    all_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    for lr, bucket in buckets.items():
        for grp in pack_with_budget(bucket):
            all_groups.append((lr, grp))
    total = len(all_groups)

    for gidx, (lr, g) in enumerate(all_groups):
        # 统计/聚合
        edge_count = len(g)
        tickers     = sorted({r.get("ticker") for r in g if r.get("ticker")})
        forms       = sorted({r.get("form") for r in g if r.get("form")})
        years       = sorted({r.get("year") for r in g if r.get("year") is not None})
        accnos      = sorted({r.get("accno") for r in g if r.get("accno")})
        file_types  = sorted({r.get("file_type") for r in g if r.get("file_type")})
# 兼容 cal/def 的端点聚合（确保在此片段之前定义）
    def _endpoints(r):
        if "parent_concept" in r or "child_concept" in r:   # calculation_edges
            return r.get("parent_concept"), r.get("child_concept")
        else:                                               # definition_arcs
            return r.get("from_concept"), r.get("to_concept")

    parents  = sorted({p for p, _ in (_endpoints(r) for r in g) if p})
    children = sorted({c for _, c in (_endpoints(r) for r in g) if c})

    arcroles   = sorted({r.get("arcrole") for r in g if r.get("arcrole")})
    linkroles  = sorted({r.get("linkrole") for r in g if r.get("linkrole")})

    # chunk_id 前缀：优先 accno；否则用 relpath（去斜杠）
    acc = next((r.get("accno") for r in g if r.get("accno")), None)
    rp  = (relpath_under_input or str(in_file.name)).replace("\\", "/").replace("/", "|")
    base = acc or rp

    # kind（cal/def/fact/generic）
    if len(file_types) == 1:
        kind = file_types[0] or "generic"
    else:
        stem = in_file.stem.lower()
        if "calculation" in stem:
            kind = "cal"
        elif "definition" in stem or stem.startswith("def"):
            kind = "def"
        elif "fact" == stem:
            kind = "fact"
        else:
            kind = "generic"

    # 将 linkrole 压缩成短标签放进 chunk_id，避免过长
    lr_tag = (lr or "__NA__").split("/")[-1][:48]  # 取末段并截断
    chunk_id = f"{base}::{kind}::{lr_tag}::group::{gidx}"

    # 清理冗余字段
    for r in g:
        if r.get("preferred_label") is None:
            r.pop("preferred_label", None)

    # 更安全的 summary_text：仅拼有端点的前若干条
    lines = []
    for r in g:
        a, b = _endpoints(r)
        if not (a and b):
            continue
        mid = r.get("arcrole") or r.get("weight")
        mid = (mid.split("/")[-1] if isinstance(mid, str) else str(mid)) if mid is not None else ""
        lines.append(f"{a} --{mid}--> {b}")
        if len(lines) >= 10:   # 避免过长
            break
    summary_text = "\n".join(lines)

    out = {
        "parent_concepts": parents,
        "arcroles":        arcroles,
        "child_concepts":  children,
        "chunk_id":        chunk_id,
        "chunk_index":     gidx,
        "chunk_count":     total,
        "source_file":     str(in_file),
        "edge_count":      edge_count,
        "file_type":       (file_types[0] if len(file_types) == 1 else file_types or kind),
        "tickers":         tickers,
        "forms":           forms,
        "years":           years,
        "accnos":          accnos,
        "linkroles":       [lr],        # 当前组对应的 linkrole（保持与 chunk_id 一致）
        "edges":           g,           # 如需沿用 items，这里改回 "items": g
        "summary_text":    summary_text,
    }
    if relpath_under_input is not None:
        out["relpath"] = relpath_under_input


        if add_summary_text:
            lines = []
            for r in g:
                p, c = _endpoints(r)
                ar = r.get("arcrole")
                if p and c:
                    if ar:
                        lines.append(f"{p} --{ar.split('/')[-1]}--> {c}")
                    else:
                        lines.append(f"{p} --> {c}")
            out["summary_text"] = "\n".join(lines)

        out_rows.append(out)

    return write_jsonl(out_file, out_rows)



# ----------------------------- Orchestrator -----------------------------
FILE_POLICIES = {
    "text_corpus.jsonl": ("text", "text_chunks.jsonl"),
    "text.jsonl": ("text", "text_chunks.jsonl"),
    "fact.jsonl": ("group", "fact_chunks.jsonl"),
    "calculation_edges.jsonl": ("group", "calc_chunks.jsonl"),
    "definition_arcs.jsonl": ("group", "def_chunks.jsonl"),
    "labels_best.jsonl": ("group", "labels_best_chunks.jsonl"),
    "labels_wide.jsonl": ("group", "labels_wide_chunks.jsonl"),
    "labels.jsonl": ("copy", "labels.jsonl"),
}

def run(
    input_root: Path,
    output_root: Path,
    *,
    max_tokens: int = 300,
    overlap: int = 60,
    group_size: int = 50,
    copy_labels: bool = True,
    chunk_unknown_jsonl: bool = True,
) -> Dict[str, int]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    stats = {
        "text_files": 0,
        "group_files": 0,
        "labels_copied": 0,
        "unknown_chunked": 0,
        "skipped": 0,
        "chunks_written": 0,
    }

    for f in input_root.rglob("*.jsonl"):
        if not f.is_file():
            continue
        rel_dir = f.relative_to(input_root).parent
        relpath = str(f.relative_to(input_root)).replace("\\", "/")
        out_dir = (output_root / rel_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        name = f.name
        policy, out_name = FILE_POLICIES.get(name, (None, None))

        # labels.jsonl → copy
        if policy == "copy":
            if copy_labels:
                shutil.copy2(f, out_dir / out_name)
                stats["labels_copied"] += 1
            else:
                stats["skipped"] += 1
            continue

        # 文本滑窗
        if policy == "text":
            n = process_text_jsonl(
                f, out_dir / out_name,
                relpath_under_input=relpath,
                max_tokens=max_tokens, overlap=overlap,
            )
            stats["text_files"] += 1
            stats["chunks_written"] += n
            continue

        # 结构分组
        if policy == "group":
            n = process_group_jsonl(
                f, out_dir / out_name, group_size=group_size,
                relpath_under_input=relpath,
            )
            stats["group_files"] += 1
            stats["chunks_written"] += n
            continue

        # 未知的 *.jsonl：按 group 分块（可关闭）
        if policy is None and name != "labels.jsonl":
            if chunk_unknown_jsonl:
                n = process_group_jsonl(
                    f, out_dir / "generic_chunks.jsonl", group_size=group_size,
                    relpath_under_input=relpath,
                )
                stats["unknown_chunked"] += 1
                stats["chunks_written"] += n
            else:
                stats["skipped"] += 1
            continue

        stats["skipped"] += 1

    return stats

# ----------------------------- CLI -----------------------------
def parse_args() -> argparse.Namespace:
    prj = discover_project_root()
    ap = argparse.ArgumentParser(description="Chunk all JSONL except labels.jsonl; mirror silver -> chunked.")
    ap.add_argument("--input-root", type=Path, default=prj / "data" / "silver",
                    help="Input root (default: <PROJECT_ROOT>/data/silver)")
    ap.add_argument("--output-root", type=Path, default=prj / "data" / "chunked",
                    help="Output root (default: <PROJECT_ROOT>/data/chunked)")
    ap.add_argument("--max-tokens", type=int, default=300, help="Max tokens per text chunk")
    ap.add_argument("--overlap", type=int, default=60, help="Token overlap between text chunks")
    ap.add_argument("--group-size", type=int, default=20, help="Records per group chunk")
    ap.add_argument("--no-copy-labels", action="store_true", help="Do not copy labels.jsonl")
    ap.add_argument("--no-unknown", action="store_true", help="Skip unknown *.jsonl instead of chunking generically")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    stats = run(
        input_root=args.input_root,
        output_root=args.output_root,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        group_size=args.group_size,
        copy_labels=not args.no_copy_labels,
        chunk_unknown_jsonl=not args.no_unknown,
    )
    print(json.dumps({"summary": stats}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
