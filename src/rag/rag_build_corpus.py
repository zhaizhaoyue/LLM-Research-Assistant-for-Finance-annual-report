# rag/rag_build_corpus.py
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from collections import Counter

# ----------------------- IO & utils -----------------------

def to_posix_path(s):
    return None if not s else str(s).replace("\\", "/")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                print(f"[warn] bad json in {path}: {e}", file=sys.stderr)

def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)

def sanitize(x: Any) -> Any:
    # NaN -> None；递归处理
    if is_nan(x):
        return None
    if isinstance(x, dict):
        return {k: sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [sanitize(v) for v in x]
    return x

def norm_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def norm_int(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None

def first_scalar(x: Any) -> Any:
    if x is None or x == "":
        return None
    if isinstance(x, (list, tuple)):
        return x[0] if x else None
    return x

def to_posix_path(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return str(s).replace("\\", "/")

def safe_id(x: Optional[str]) -> str:
    import re
    if not x: return "NA"
    s = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(x))
    return s[:180]

def period_key(instant: Any, p_start: Any, p_end: Any) -> str:
    if instant:
        return f"instant={instant}"
    if p_start or p_end:
        return f"{p_start or 'NA'}~{p_end or 'NA'}"
    return "NA"

def take_first(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, "", []):
            return row[k]
    return None

# ----------------------- kind 推断 -----------------------

KIND_SET = {"auto","text","fact","labels","labels_best","labels_wide","cal","def"}

def infer_kind_from_path(p: Path) -> str:
    name = p.name.lower()
    stem = p.stem.lower()
    # 文件名优先
    if any(k in name for k in ["text_corpus","text_chunks","text.jsonl","text_chunks.jsonl","text"]):
        return "text"
    if "fact" in name:
        return "fact"
    if "calculation" in name or "cal_edges" in name or "cal" in name:
        return "cal"
    if "definition" in name or "def_arcs" in name or name.endswith("def.jsonl"):
        return "def"
    if "labels_best" in name:
        return "labels_best"
    if "labels_wide" in name:
        return "labels_wide"
    if "labels" in name:
        return "labels"
    # stem 兜底
    for k in ["text","fact","cal","def","labels_best","labels_wide","labels"]:
        if k in stem:
            return k
    return "text"  # 最安全的默认

# ----------------------- 统一元字段抽取 -----------------------

def base_meta_from_any(row: Dict[str, Any], default_file_type: str, source_path_hint: Optional[str]) -> Dict[str, Any]:
    return {
        "ticker": norm_str(first_scalar(row.get("ticker") or first_scalar(row.get("tickers")))),
        "form":   norm_str(first_scalar(row.get("form")   or first_scalar(row.get("forms")))),
        "year":   norm_int(first_scalar(row.get("year")   or first_scalar(row.get("years")))),
        "accno":  norm_str(first_scalar(row.get("accno")  or first_scalar(row.get("accnos")))),
        "file_type": default_file_type,
        "source_path": to_posix_path(row.get("source_path") or source_path_hint),
        "item": None,
        "section": norm_str(row.get("section")),
        "page_no": row.get("page_no") if isinstance(row.get("page_no"), (int, type(None))) else None,
    }

# ----------------------- 各类 standardizer -----------------------

def std_text(row: Dict[str, Any], src_file: Path, i: int, min_text_len: int) -> Optional[Dict[str, Any]]:
    row = sanitize(row)
    txt = take_first(row, [
        "text","content","body","paragraph","rag_text","summary_text","raw_text",
        "clean_text","text_block","text_chunk","text_clean"
    ])
    if not txt:
        lines = row.get("lines") or row.get("sentences") or row.get("paragraphs")
        if isinstance(lines, list):
            txt = " ".join([str(x) for x in lines if x not in (None, "")])
    txt = norm_str(txt) or ""
    if len(txt) < min_text_len:
        return None

    meta = base_meta_from_any(row, "text", source_path_hint=row.get("source_file") or row.get("source_path"))
    chunk_id = norm_str(row.get("chunk_id")) or f"{safe_id(meta['accno'])}::text::chunk-{i}"
    extra = {
        "relpath": row.get("relpath"),
        "heading": row.get("heading"),
        "statement_hint": row.get("statement_hint"),
        "text_field": row.get("text_field"),
        "tokens_in_chunk": row.get("tokens_in_chunk"),
        "chunk_span_tokens": row.get("chunk_span_tokens"),
    }
    return {"chunk_id": chunk_id, "text": txt, **meta, "extra": extra}

def std_labels_single(row: Dict[str, Any], src_file: Path, i: int, flavor: str, min_text_len: int) -> Optional[Dict[str, Any]]:
    # 单条 label（如你贴的 labels 行）
    row = sanitize(row)
    concept = norm_str(take_first(row, ["concept","qname","name"]))
    label_best = norm_str(take_first(row, [
        "label_best","label_text","label","label_best_text","label_standard_en-US","label_standard_en_us","label_standard_en-US".replace("-","_")
    ]))
    label_doc = norm_str(take_first(row, ["label_doc","label_documentation_en-US","label_documentation_en_us"]))
    role = norm_str(take_first(row, ["label_best_role","role","linkrole","arcrole"]))
    # 拼文本（尽量不空）
    parts = [x for x in [
        f"{concept}" if concept else None,
        f"{label_best}" if label_best else None,
        f"[{role}]" if role else None,
        f"({label_doc})" if label_doc else None,
    ] if x]
    txt = " ".join(parts) or (concept or label_best or label_doc or "")
    txt = norm_str(txt) or ""
    if len(txt) < min_text_len:
        return None

    meta = base_meta_from_any(row, "labels", source_path_hint=row.get("source_file"))
    chunk_id = norm_str(row.get("chunk_id")) or f"{safe_id(meta['accno'])}::labels::{flavor}::{i}"
    extra = {k: v for k, v in row.items() if k not in {"ticker","fy","fq","form","accno","doc_date","concept","label_best","label_doc"}}
    return {"chunk_id": chunk_id, "text": txt, **meta, "extra": extra}

def std_labels_edges(row: Dict[str, Any], src_file: Path, i: int, flavor: str, min_text_len: int) -> List[Dict[str, Any]]:
    # labels_best.jsonl / labels_wide.jsonl：行内携带 edges 列表
    out: List[Dict[str, Any]] = []
    row = sanitize(row)
    edges = row.get("edges") or []
    base_meta = base_meta_from_any(row, "labels", source_path_hint=row.get("source_file"))
    for j, e in enumerate(edges):
        e = sanitize(e)
        concept = norm_str(take_first(e, ["concept","qname","name"]))
        label_best = norm_str(take_first(e, [
            "label_best","label_standard_en-US","label_standard_en_us"
        ]))
        label_doc = norm_str(take_first(e, ["label_doc","label_documentation_en-US","label_documentation_en_us"]))
        role = norm_str(take_first(e, ["label_best_role","role","linkrole","arcrole"]))
        parts = [x for x in [
            f"{concept}" if concept else None,
            f"{label_best}" if label_best else None,
            f"[{role}]" if role else None,
            f"({label_doc})" if label_doc else None,
        ] if x]
        txt = norm_str(" ".join(parts)) or ""
        if len(txt) < min_text_len:
            continue

        meta = dict(base_meta)
        meta["ticker"] = norm_str(e.get("ticker")) or meta["ticker"]
        meta["form"]   = norm_str(e.get("form"))   or meta["form"]
        meta["year"]   = norm_int(e.get("year"))   or meta["year"]
        meta["accno"]  = norm_str(e.get("accno"))  or meta["accno"]
        meta["source_path"] = to_posix_path(e.get("source_path")) or meta["source_path"]
        chunk_id = f"{safe_id(meta['accno'])}::labels::{flavor}::{safe_id(concept)}::e{j}"
        extra = {k: v for k, v in e.items() if k not in {"concept","label_best","label_doc","role","label_best_role"}}
        out.append({"chunk_id": chunk_id, "text": txt, **meta, "extra": extra})
    # 追加 group（可选：如果行上有 summary_text）
    summ = norm_str(row.get("summary_text"))
    if summ:
        out.append({
            "chunk_id": f"{safe_id(base_meta['accno'])}::labels::{flavor}::group::{i}",
            "text": summ, **base_meta, "extra": {"relpath": row.get("relpath")}
        })
    return out

def std_cal_row(row: Dict[str, Any], src_file: Path, i: int, mode: str, min_text_len: int) -> List[Dict[str, Any]]:
    # calculation_edges.jsonl（你贴的 calc_chunks）
    out: List[Dict[str, Any]] = []
    row = sanitize(row)
    base_meta = base_meta_from_any(row, "cal", source_path_hint=row.get("source_file"))

    if mode in ("explode","both"):
        for j, e in enumerate(row.get("edges") or []):
            e = sanitize(e)
            parent = norm_str(e.get("parent_concept") or first_scalar(row.get("parent_concepts")))
            child  = norm_str(e.get("child_concept")  or first_scalar(row.get("child_concepts")))
            weight = e.get("weight")
            role   = norm_str(e.get("linkrole") or first_scalar(row.get("linkroles")))
            txt = " --".join([t for t in [parent or "", norm_str(weight) or ""] if t]) + f"--> {child or ''}"
            if role: txt = f"{txt} [{role}]"
            txt = norm_str(txt) or ""
            if len(txt) < min_text_len:
                continue

            meta = dict(base_meta)
            meta["ticker"] = norm_str(e.get("ticker")) or meta["ticker"]
            meta["form"]   = norm_str(e.get("form"))   or meta["form"]
            meta["year"]   = norm_int(e.get("year"))   or meta["year"]
            meta["accno"]  = norm_str(e.get("accno"))  or meta["accno"]
            meta["source_path"] = to_posix_path(e.get("source_path")) or meta["source_path"]

            per = period_key(e.get("instant"), e.get("period_start"), e.get("period_end"))
            chunk_id = f"{safe_id(meta['accno'])}::cal::{safe_id(parent)}->{safe_id(child)}::{safe_id(per)}::e{j}"
            extra = {k: v for k, v in e.items() if k not in {"parent_concept","child_concept","weight","linkrole"}}
            out.append({"chunk_id": chunk_id, "text": txt, **meta, "extra": extra})

    if mode in ("summary_only","both"):
        summ = norm_str(row.get("summary_text"))
        if summ and len(summ) >= min_text_len:
            gid = norm_str(row.get("chunk_id")) or f"{safe_id(base_meta['accno'])}::cal::group::{i}"
            out.append({"chunk_id": gid, "text": summ, **base_meta, "extra": {"relpath": row.get("relpath")}})
    return out

def std_def_row(row: Dict[str, Any], src_file: Path, i: int, mode: str, min_text_len: int) -> List[Dict[str, Any]]:
    # definition_arcs.jsonl（你贴的 def_chunks）
    out: List[Dict[str, Any]] = []
    row = sanitize(row)
    base_meta = base_meta_from_any(row, "def", source_path_hint=row.get("source_file"))

    if mode in ("explode","both"):
        for j, e in enumerate(row.get("edges") or []):
            e = sanitize(e)
            frm = norm_str(e.get("from_concept") or first_scalar(row.get("parent_concepts")))
            to  = norm_str(e.get("to_concept")   or first_scalar(row.get("child_concepts")))
            arc = norm_str(e.get("arcrole") or first_scalar(row.get("arcroles")))
            role = norm_str(e.get("linkrole") or first_scalar(row.get("linkroles")))
            txt = f"{frm or ''} --{arc or 'domain-member'}--> {to or ''}"
            if role: txt = f"{txt} [{role}]"
            txt = norm_str(txt) or ""
            if len(txt) < min_text_len:
                continue

            meta = dict(base_meta)
            meta["ticker"] = norm_str(e.get("ticker")) or meta["ticker"]
            meta["form"]   = norm_str(e.get("form"))   or meta["form"]
            meta["year"]   = norm_int(e.get("year"))   or meta["year"]
            meta["accno"]  = norm_str(e.get("accno"))  or meta["accno"]
            meta["source_path"] = to_posix_path(e.get("source_path")) or meta["source_path"]

            chunk_id = f"{safe_id(meta['accno'])}::def::{safe_id(frm)}->{safe_id(to)}::{safe_id(arc)}::e{j}"
            extra = {k: v for k, v in e.items() if k not in {"from_concept","to_concept","arcrole","linkrole"}}
            out.append({"chunk_id": chunk_id, "text": txt, **meta, "extra": extra})

    if mode in ("summary_only","both"):
        summ = norm_str(row.get("summary_text"))
        if summ and len(summ) >= min_text_len:
            gid = norm_str(row.get("chunk_id")) or f"{safe_id(base_meta['accno'])}::def::group::{i}"
            out.append({"chunk_id": gid, "text": summ, **base_meta, "extra": {"relpath": row.get("relpath")}})
    return out

def std_fact_row(row: Dict[str, Any], src_file: Path, i: int, mode: str, min_text_len: int, dedup_edge: bool) -> List[Dict[str, Any]]:
    # 兼容你之前的 fact（含 edges & summary_text）
    out: List[Dict[str, Any]] = []
    row = sanitize(row)
    base_meta = base_meta_from_any(row, "fact", source_path_hint=row.get("source_file"))
    seen = set()

    if mode in ("explode","both"):
        for j, e in enumerate(row.get("edges") or []):
            e = sanitize(e)
            txt = norm_str(e.get("rag_text"))
            if not txt:
                qname = norm_str(e.get("qname"))
                vdisp = e.get("value_display");  vdisp = str(vdisp) if vdisp not in (None, "") else None
                pl    = norm_str(e.get("period_label"))
                txt = " : ".join([t for t in (qname, vdisp) if t]) or ""
                if pl: txt += f" ({pl})"
            txt = norm_str(txt) or ""
            if len(txt) < min_text_len:
                continue

            qn  = norm_str(e.get("qname")) or "NA"
            per = period_key(e.get("instant"), e.get("period_start"), e.get("period_end"))
            vkey = e.get("value_display") if e.get("value_display") not in (None, "") else e.get("value")
            sig = (qn, per, str(vkey))
            if dedup_edge and sig in seen:
                continue
            seen.add(sig)

            meta = dict(base_meta)
            meta["ticker"] = norm_str(e.get("ticker")) or meta["ticker"]
            meta["form"]   = norm_str(e.get("form"))   or meta["form"]
            meta["year"]   = norm_int(e.get("year"))   or meta["year"]
            meta["accno"]  = norm_str(e.get("accno"))  or meta["accno"]
            meta["source_path"] = to_posix_path(e.get("source_path")) or meta["source_path"]
            chunk_id = (
                f"{safe_id(meta['accno'])}::fact::{safe_id(qn)}::{safe_id(per)}::e{j}"
            )
            extra = {k: v for k, v in e.items() if k not in {
                "qname","rag_text","value_display","period_label","source_path","page_no",
                "ticker","form","year","accno","instant","period_start","period_end"
            }}
            out.append({"chunk_id": chunk_id, "text": txt, **meta, "extra": extra})

    if mode in ("summary_only","both"):
        summ = norm_str(row.get("summary_text"))
        if summ and len(summ) >= min_text_len:
            gid = norm_str(row.get("chunk_id")) or f"{safe_id(base_meta['accno'])}::fact::group::{i}"
            out.append({"chunk_id": gid, "text": summ, **base_meta, "extra": {"relpath": row.get("relpath")}})
    return out

# ----------------------- finalize -----------------------
def finalize_chunk(chunk: dict) -> dict:
    # 1) 如果 source_path 为空，就用 extra.relpath 补
    if not chunk.get("source_path"):
        rel = chunk.get("extra", {}).get("relpath")
        if rel:
            chunk["source_path"] = rel.replace("\\", "/")

    # 2) 把 source_path 统一为 POSIX
    if chunk.get("source_path"):
        chunk["source_path"] = chunk["source_path"].replace("\\", "/")

    # 3) 去掉 extra 里重复的元字段
    META_KEYS = {"ticker","form","year","accno","file_type","source_path","page_no","item","section"}
    extra = chunk.get("extra") or {}
    chunk["extra"] = {k: v for k, v in extra.items() if k not in META_KEYS}

    return chunk


# ----------------------- 主流程 -----------------------

def process_file(src_root: Path, src_file: Path, dst_root: Path, args) -> int:
    rel = src_file.relative_to(src_root)
    dst_file = dst_root / rel
    ensure_parent(dst_file)

    kind = args.kind
    if kind == "auto":
        kind = infer_kind_from_path(src_file)

    written = 0
    stats = Counter()

    with dst_file.open("w", encoding="utf-8") as out:
        for i, row in enumerate(iter_jsonl(src_file)):
            try:
                if kind == "text":
                    ck = std_text(row, src_file, i, args.min_text_len)
                    if ck: out.write(dumps(ck) + "\n"); written += 1
                    else: stats["drop_text"] += 1

                elif kind == "labels":
                    # 既支持单条对象，也支持携带 labels/edges 数组
                    if isinstance(row.get("edges"), list):
                        cks = std_labels_edges(row, src_file, i, "labels", args.min_text_len)
                    if cks:
                        for ck in cks:
                            ck = finalize_chunk(ck)
                            out.write(json.dumps(ck, ensure_ascii=False) + "\n")
                    elif isinstance(row.get("labels"), list):
                        base = {k: v for k, v in row.items() if k != "labels"}
                        cnt = 0
                        for j, lab in enumerate(row["labels"]):
                            merged = {**base, **lab}
                            ck = std_labels_single(merged, src_file, i*100000+j, "labels", args.min_text_len)
                            if ck: out.write(dumps(ck) + "\n"); cnt += 1
                        written += cnt
                    else:
                        ck = std_labels_single(row, src_file, i, "labels", args.min_text_len)
                        if ck: out.write(dumps(ck) + "\n"); written += 1
                        else: stats["drop_labels"] += 1

                elif kind == "labels_best":
                    cks = std_labels_edges(row, src_file, i, "labels_best", args.min_text_len)
                    if cks:
                        for ck in cks:
                            ck = finalize_chunk(ck)
                            out.write(json.dumps(ck, ensure_ascii=False) + "\n")
                        written += len(cks)
                    else:
                        # 某些仓库 labels_best 是单条对象
                        ck = std_labels_single(row, src_file, i, "labels_best", args.min_text_len)
                        if ck: out.write(dumps(ck) + "\n"); written += 1
                        else: stats["drop_labels_best"] += 1

                elif kind == "labels_wide":
                    cks = std_labels_edges(row, src_file, i, "labels_wide", args.min_text_len)
                    if cks:
                        for ck in cks:
                            ck = finalize_chunk(ck)
                            out.write(json.dumps(ck, ensure_ascii=False) + "\n")
                        written += len(cks)
                    else:
                        ck = std_labels_single(row, src_file, i, "labels_wide", args.min_text_len)
                        if ck: out.write(dumps(ck) + "\n"); written += 1
                        else: stats["drop_labels_wide"] += 1

                elif kind == "cal":
                    cks = std_cal_row(row, src_file, i, args.mode, args.min_text_len)
                    for ck in cks:
                        ck = finalize_chunk(ck)
                        out.write(json.dumps(ck, ensure_ascii=False) + "\n")
                    written += len(cks)

                elif kind == "def":
                    cks = std_def_row(row, src_file, i, args.mode, args.min_text_len)
                    for ck in cks:
                        ck = finalize_chunk(ck)
                        out.write(json.dumps(ck, ensure_ascii=False) + "\n")
                    written += len(cks)

                elif kind == "fact":
                    cks = std_fact_row(row, src_file, i, args.mode, args.min_text_len, args.dedup_edge)
                    for ck in cks:
                        ck = finalize_chunk(ck)
                        out.write(json.dumps(ck, ensure_ascii=False) + "\n")
                    written += len(cks)

                else:
                    # 不认识的类型，按 text 兜底
                    ck = std_text(row, src_file, i, args.min_text_len)
                    if ck:
                        ck = finalize_chunk(ck)   # ✅ 在这里调用
                        out.write(json.dumps(ck, ensure_ascii=False) + "\n")
                    else: stats["drop_unknown"] += 1

            except Exception as e:
                stats["error"] += 1
                if args.debug:
                    print(f"[debug] error {rel}#{i}: {e}", file=sys.stderr)

    print(f"[ok] {rel} -> {written} chunks ; {dict(stats)}")
    return written

def scan_files(src: Path, src_glob: Optional[str]) -> List[Path]:
    if src_glob:
        return sorted(src.glob(src_glob))
    # 默认：所有 jsonl
    return sorted(src.glob("**/*.jsonl"))

# ----------------------- CLI -----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True, help="源根目录（含各类 *.jsonl）")
    p.add_argument("--dst", type=str, required=True, help="输出根目录（标准化后，镜像路径）")
    p.add_argument("--src-glob", type=str, default=None, help='限定匹配的文件（例如 "**/AAPL/**/text_*.jsonl"）')
    p.add_argument("--kind", type=str, default="auto", choices=list(KIND_SET),
                   help="处理的文件类型：auto/text/fact/labels/labels_best/labels_wide/cal/def")
    p.add_argument("--mode", type=str, default="both", choices=["explode","summary_only","both"],
                   help="对 cal/def/fact 的输出模式（explode=只 edges，summary_only=只 group，both=两者）")
    p.add_argument("--min-text-len", type=int, default=3, help="最短文本长度（小于即丢弃）")
    p.add_argument("--dedup-edge", action="store_true", default=True, help="fact 的行内轻量去重")
    p.add_argument("--debug", type=int, default=0, help=">0 时输出异常调试信息")
    return p.parse_args()

def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    files = scan_files(src_root, args.src_glob)

    total_files, total_chunks = 0, 0
    for f in files:
        # 当 kind!=auto 时，只处理“推断出的类型 == 指定 kind”的文件；避免跨类混写
        if args.kind != "auto":
            inferred = infer_kind_from_path(f)
            if inferred != args.kind:
                continue
        total_files += 1
        total_chunks += process_file(src_root, f, dst_root, args)

    print(f"[DONE] files={total_files} chunks={total_chunks}\n[OUT]  {dst_root.as_posix()}")

if __name__ == "__main__":
    main()


#python -m src.rag.rag_build_corpus --src data/chunked --dst data/normalized --kind auto