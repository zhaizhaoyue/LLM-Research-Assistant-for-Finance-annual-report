from __future__ import annotations
import argparse, json, re, gzip, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# -------- Optional dependency: rapidfuzz (fallback to difflib) --------
try:
    from rapidfuzz import fuzz, process  # type: ignore
    HAVE_RAPIDFUZZ = True
except Exception:
    import difflib
    HAVE_RAPIDFUZZ = False

# -------- Regex helpers --------
YEAR_RE = re.compile(r"(20\d{2}|19\d{2})")
QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.I)
ACCNO_IN_PATH_RE = re.compile(r"\d{10}-\d{2}-\d{6}")
# -------- Normalizers --------
def extract_accno_from_path(path: Path) -> Optional[str]:
    m = ACCNO_IN_PATH_RE.search(str(path))
    return m.group(0) if m else None

def concept_from_record(rec: dict) -> Optional[str]:
    # 1) 从 chunk_id 解析: accno::fact::<CONCEPT>::...
    cid = rec.get("chunk_id") or ""
    if "::fact::" in cid:
        try:
            after = cid.split("::fact::", 1)[1]
            return after.split("::", 1)[0]
        except Exception:
            pass
    # 2) 从 text 解析: "<CONCEPT>:  (FY... ...)"
    txt = rec.get("text") or ""
    if ":" in txt:
        head = txt.split(":", 1)[0].strip()
        # 粗过滤：us-gaap:XXX 或 dei:XXX
        if re.match(r"^[a-z0-9\-]+:[A-Za-z0-9]", head):
            return head
    return None

def norm_accno(s: Optional[str]) -> str:
    """Uppercase, strip, keep digits and hyphens only."""
    s = (s or "").strip().upper()
    return re.sub(r"[^0-9\-]", "", s)

def norm_currency(s: Optional[str]) -> str:
    """Uppercase letters only (USD, EUR, ...)."""
    return re.sub(r"[^A-Z]", "", (s or "").upper())

def norm_concept(s: Optional[str]) -> str:
    return (s or "").strip()

# -------- IO utils --------
def iter_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def discover_fact_files(path: Path) -> List[Path]:
    """If 'path' is a file, return [path]. If directory, rglob likely names."""
    if path.is_file():
        return [path]
    files: List[Path] = []
    # Common patterns
    for pat in ["fact.jsonl", "fact.jsonl.gz", "fact_chunks.jsonl", "fact_chunks.jsonl.gz"]:
        files.extend(path.rglob(pat))
    # De-duplicate while preserving order
    seen = set()
    unique_files = []
    for fp in files:
        key = fp.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(fp)
    return unique_files

# -------- Fuzzy helpers --------
def norm_text(s: Optional[str]) -> str:
    if s is None: return ""
    s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = s.replace("non current", "noncurrent")
    s = s.replace("long term", "long-term")
    return s

def best_match_concept(account_text: str,
                       concept_list: List[str],
                       min_score: int,
                       topk: int = 5) -> Tuple[Optional[str], int, List[Tuple[str, int]]]:
    """
    返回：(best_concept, best_score, topk_list)
    - best_concept/best_score：若 best_score < min_score，则 best_concept=None, best_score=0
    - topk_list：[(concept, score), ...]，按分数降序（仅在 debug 时写入）
    """
    if not account_text or not concept_list:
        return None, 0, []

    q = norm_text(account_text)

    # 优先 rapidfuzz
    if HAVE_RAPIDFUZZ:
        # 全量 Top-K
        all_matches = process.extract(q, concept_list, scorer=fuzz.WRatio, limit=max(1, topk))
        # 归一到 [(str, int)]
        topk_list = [(c, int(s)) for c, s, _ in all_matches]
        if topk_list:
            best_c, best_s = topk_list[0]
            if best_s >= min_score:
                return best_c, best_s, topk_list
        return None, 0, topk_list

    # difflib 退化路径
    # 先简单把分数计算一遍
    scores = []
    for c in concept_list:
        # difflib.SequenceMatcher(None, q, norm_text(c)).ratio() ∈ [0,1]
        import difflib
        s = difflib.SequenceMatcher(None, q, norm_text(c)).ratio()
        scores.append((c, int(round(s * 100))))
    scores.sort(key=lambda x: x[1], reverse=True)
    topk_list = scores[:max(1, topk)]
    if topk_list and topk_list[0][1] >= min_score:
        return topk_list[0][0], topk_list[0][1], topk_list
    return None, 0, topk_list


# -------- Period helpers --------
def extract_period(header: str) -> Dict[str, Any]:
    """Parse column header to period info dict: {'year': 2024, 'quarter': 2} possibly empty."""
    h = str(header)
    m_y = YEAR_RE.search(h)
    m_q = QUARTER_RE.search(h)
    out: Dict[str, Any] = {}
    if m_y: out["year"] = int(m_y.group(0))
    if m_q: out["quarter"] = int(m_q.group(1))
    hl = h.lower()
    if "current" in hl and "year" not in out:
        out["tag"] = "current"
    if "prior" in hl or "previous" in hl:
        out["tag"] = "prior"
    return out

def fact_period_key(f: Dict[str, Any]) -> Tuple[int, Optional[int]]:
    """Create sortable period key from common fields: prefer fy/fp; fallback to period/instant dates."""
    fy = f.get("fy")
    fp = f.get("fp")
    if isinstance(fy, int):
        q = None
        if isinstance(fp, str) and fp.upper().startswith("Q"):
            try:
                q = int(fp[1])
            except Exception:
                q = None
        return (fy, q)
    per = str(f.get("period") or "")
    m_y = YEAR_RE.search(per)
    m_q = QUARTER_RE.search(per)
    y = int(m_y.group(0)) if m_y else 0
    q = int(m_q.group(1)) if m_q else None
    return (y, q)

# -------- Facts loading --------
# === 用这版替换你脚本中的 load_facts_from_root ===
def load_facts_from_root(root_or_file: Path, progress_every: int = 50) -> Dict[str, List[Dict[str, Any]]]:
    by_accno: Dict[str, List[Dict[str, Any]]] = {}
    fact_files = discover_fact_files(root_or_file)
    print(f"[info] scanning {len(fact_files)} facts file(s) under {root_or_file}", flush=True)

    for i, fp in enumerate(fact_files, 1):
        if i % max(1, progress_every) == 0:
            print(f"[progress] {i}/{len(fact_files)} files loaded", flush=True)

        # 支持 JSONL（逐行），gz 仍用原 iter_jsonl；普通文件优先逐行解析
        try:
            it = iter_jsonl(fp) if str(fp).endswith(".gz") else (json.loads(line) for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        except Exception:
            print(f"[warn] failed to open {fp}", flush=True)
            continue

        for rec in it:
            if not isinstance(rec, dict):
                continue

            # --- accno ---
            acc_raw = rec.get("accno") or rec.get("accessionNumber") or rec.get("accession") or rec.get("filedAs")
            if not acc_raw:
                acc_raw = extract_accno_from_path(fp)  # 从路径兜底
            acc = norm_accno(acc_raw)
            if not acc:
                continue

            # --- concept ---
            concept = concept_from_record(rec)
            if not concept:
                continue  # 概念缺失就跳过（没有可对齐的 key）

            # --- value / currency / period from extra ---
            extra = rec.get("extra") or {}
            val = (extra.get("value_num") if extra.get("value_num") is not None else
                   extra.get("value_raw") if extra.get("value_raw") is not None else
                   extra.get("value") if extra.get("value") is not None else
                   rec.get("value") or rec.get("val") or rec.get("amount"))

            # 币种：unit / unit_normalized；你的示例里是 null，先别严格过滤
            cur = norm_currency(extra.get("unit") or extra.get("unit_normalized") or rec.get("currency") or rec.get("unit") or rec.get("unitRef"))

            # 年度/季度与期间
            fy = extra.get("fy_norm") or extra.get("fy")
            fq = extra.get("fq_norm") or extra.get("fq") or extra.get("fq_norm")
            fp = f"Q{int(fq)}" if fq and str(fq).isdigit() else extra.get("fq") or None
            period = extra.get("doc_date") or (f"{fy}Q{fq}" if fy and fq else "")
            ctx = extra.get("context") or {}
            per = ctx.get("period") or {}
            start = per.get("start_date") or extra.get("start")
            end = per.get("end_date") or extra.get("end")
            instant = per.get("instant") or extra.get("instant")

            fact = {
                "accno": acc,
                "concept": norm_concept(concept),
                "value": val,
                "currency": cur,  # 可能为空；上游用 --no_currency_filter 时会忽略
                "fy": fy,
                "fp": fp,
                "period": period,
                "start": start,
                "end": end,
                "instant": instant,
            }
            by_accno.setdefault(acc, []).append(fact)

    return by_accno


# -------- Alignment core --------
def is_nan(x: Any) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None

def align_table(csv_path: Path,
                meta_path: Path,
                facts_by_accno: Dict[str, List[Dict[str, Any]]],
                min_score: int,
                limit_concepts: int,
                max_rows: int,
                out_suffix: str = ".aligned",
                no_currency_filter: bool = False,
                show_nearest: bool = False,
                debug_match: bool = False,
                topk: int = 5):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    accno_raw = meta.get("accno") or meta.get("accession") or meta.get("accessionNumber")
    if not accno_raw:
        print(f"[warn] {meta_path} missing accno; skip", flush=True)
        return
    accno = norm_accno(accno_raw)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[error] cannot read CSV {csv_path}: {e}", flush=True)
        return

    # identify columns
    account_col = meta.get("account_col") or df.columns[0]
    if account_col not in df.columns:
        account_col = df.columns[0]
    period_cols = [c for c in df.columns if c != account_col]
    period_info = {c: extract_period(c) for c in period_cols}

    facts = facts_by_accno.get(accno, [])
    if not facts:
        if show_nearest and facts_by_accno:
            # nearest accno hints
            try:
                if HAVE_RAPIDFUZZ:
                    from rapidfuzz import process, fuzz  # type: ignore
                    near = process.extract(accno, list(facts_by_accno.keys()), scorer=fuzz.WRatio, limit=3)
                    print(f"[info] no facts for accno={accno}; nearest accno keys: {near}", flush=True)
                else:
                    print(f"[info] no facts for accno={accno} (rapidfuzz not installed, no nearest hints)", flush=True)
            except Exception:
                pass
        print(f"[info] no facts for accno={accno} -> {csv_path.name}", flush=True)
        facts_concepts = []
    else:
        m_currency = meta.get("currency")
        if m_currency and not no_currency_filter:
            cur = norm_currency(m_currency)
            before = len(facts)
            facts = [f for f in facts if norm_currency(f.get("currency")) == cur]
            after = len(facts)
            if before and after == 0:
                print(f"[info] currency filter removed all facts for accno={accno} "
                    f"(meta currency={cur}, before={before}, after={after})", flush=True)

        # collect unique concepts (limit for speed)
        unique_concepts = list({str(f["concept"]) for f in facts if f.get("concept")})
        facts_concepts = unique_concepts[:max(1, limit_concepts)]

    align_rows = []
    for ridx, row in df.head(max_rows).iterrows():
        account_text = str(row[account_col])
        concept, score, topk_list = best_match_concept(
            account_text, facts_concepts, min_score=min_score, topk=topk
        )
        concept_facts = [f for f in facts if f.get("concept") == concept] if concept else []
        col_values: Dict[str, Any] = {}
        for c in period_cols:
            col_text = row[c]
            p = period_info[c]
            val = None
            chosen = None
            if concept_facts:
                def dist(f):
                    fy, fq = fact_period_key(f)
                    py = p.get("year", 0)
                    pq = p.get("quarter")
                    dy = abs(fy - py) if py else 5
                    dq = 0 if (pq is None or fq is None) else abs((fq or 0) - (pq or 0))
                    return dy*10 + dq
                concept_facts_sorted = sorted(concept_facts, key=dist)
                chosen = concept_facts_sorted[0] if concept_facts_sorted else None
                if chosen is not None:
                    val = chosen.get("value")
            col_values[str(c)] = {
                "table_value": None if is_nan(col_text) else col_text,
                "fact_value": val,
                "fact_period": {"fy": chosen.get("fy") if chosen else None,
                                "fp": chosen.get("fp") if chosen else None,
                                "period": chosen.get("period") if chosen else None}
            }

        row_info = {
            "row_index": int(ridx),
            "account_text": account_text,
            "matched_concept": concept,
            "match_score": int(score),
            "columns": col_values
        }

        if debug_match:
            row_info["debug_topk"] = [{"concept": c, "score": s} for c, s in topk_list]

        align_rows.append(row_info)

    out_meta = dict(meta)
    out_meta["alignment"] = {
        "account_column": str(account_col),
        "period_columns": [str(c) for c in period_cols],
        "rows": align_rows,
        "facts_used": len(facts),
        "concept_candidates": len(facts_concepts),
        "min_score": min_score,
        "currency_filter_applied": bool(meta.get("currency")) and not no_currency_filter
    }

    out_path = meta_path.with_suffix(meta_path.suffix.replace(".json", "") + f"{out_suffix}.json")
    Path(out_path).write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[aligned] {out_path}", flush=True)

# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact_root", required=True, help="Root folder of compact tables")
    ap.add_argument("--facts", required=True, help="Path to facts ROOT directory or a single facts file")
    ap.add_argument("--out_suffix", default=".aligned", help="Suffix for new meta file (default: .aligned)")
    ap.add_argument("--min_score", type=int, default=70, help="Fuzzy match threshold 0..100 (default 70)")
    ap.add_argument("--limit_concepts", type=int, default=100, help="Max candidate concepts per accno (default 100)")
    ap.add_argument("--max_rows", type=int, default=99999, help="Max rows per table to align")
    ap.add_argument("--no_currency_filter", action="store_true", help="Do not filter facts by currency")
    ap.add_argument("--accno_filter", type=str, default="", help="Only align this accno (normalized)")
    ap.add_argument("--show_nearest", action="store_true", help="When no facts found, show nearest accno keys")
    # 调试开关：只在这里定义一次
    ap.add_argument("--debug_match", action="store_true",
                    help="Write top-K candidate concepts and scores into alignment rows (debug_topk)")
    ap.add_argument("--topk", type=int, default=5,
                    help="How many top candidate concepts to show when --debug_match is on (default 5)")
    return ap.parse_args()


def main():
    args = parse_args()
    compact_root = Path(args.compact_root)
    facts_path = Path(args.facts)

    print(f"[info] loading facts from {facts_path} ...", flush=True)
    facts_by_accno = load_facts_from_root(facts_path)
    total_facts = sum(len(v) for v in facts_by_accno.values())
    print(f"[info] facts loaded: {total_facts} items across {len(facts_by_accno)} accno", flush=True)

    metas = list(compact_root.rglob("table_*.meta.json"))
    print(f"[info] found {len(metas)} meta files under {compact_root}", flush=True)

    accno_filter_norm = norm_accno(args.accno_filter) if args.accno_filter else ""

    for meta_path in metas:
        csv_path = meta_path.with_suffix("").with_suffix(".csv")
        if not csv_path.exists():
            print(f"[skip] csv not found for {meta_path}", flush=True)
            continue
        # if filter is set, only run matching accno
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[error] cannot read meta {meta_path}: {e}", flush=True)
            continue
        meta_accno = norm_accno(meta.get("accno") or meta.get("accession") or meta.get("accessionNumber"))
        if accno_filter_norm and meta_accno != accno_filter_norm:
            continue

        try:
            align_table(
                csv_path=csv_path,
                meta_path=meta_path,
                facts_by_accno=facts_by_accno,
                min_score=args.min_score,
                limit_concepts=args.limit_concepts,
                max_rows=args.max_rows,
                out_suffix=args.out_suffix,
                no_currency_filter=args.no_currency_filter,
                show_nearest=args.show_nearest,
                debug_match=args.debug_match,   # ← 传入
                topk=args.topk                  # ← 传入
            )
        except Exception as e:
            print(f"[error] {meta_path}: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()


# python -m src.parse.fact_align `
#   --compact_root data/compact_tables `
#   --facts data/normalized `
#   --min_score 65 `
#   --limit_concepts 500 `
#   --debug_match `
#   --topk 5
