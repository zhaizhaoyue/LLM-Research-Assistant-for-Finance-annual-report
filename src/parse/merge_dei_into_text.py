#666
# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from collections import Counter

ROOT = Path("data/processed")   # 处理过的申报根目录

DEI_FQ  = "dei:DocumentFiscalPeriodFocus"
DEI_FY  = "dei:DocumentFiscalYearFocus"
DEI_DPD = "dei:DocumentPeriodEndDate"  # YYYY-MM-DD

def _yyyymmdd(s: str | None) -> str | None:
    if not s: return None
    s = s.strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m: return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m2 = re.fullmatch(r"(\d{8})", s)
    return m2.group(1) if m2 else None

def load_dei_from_facts(facts_path: Path) -> dict:
    """
    从 facts.jsonl 汇总权威 DEI -> {'fy': int | None, 'fq': 'Q1|Q2|Q3|Q4|FY' | None, 'doc_date': 'YYYYMMDD' | None}
    """
    fq_counter = Counter()
    fy_counter = Counter()
    doc_dates  = []

    if not facts_path.exists():
        return {}

    with facts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            qn = (r.get("qname") or "").strip()
            # 兼容不同字段承载数值
            val = r.get("value_raw")
            if val is None:
                val = r.get("value_display")
            if val is None and r.get("value") is not None:
                val = str(r["value"])

            if not qn or val is None:
                continue

            if qn == DEI_FQ:
                v = str(val).upper().strip()
                if v in {"Q1","Q2","Q3","Q4","FY"}:
                    fq_counter[v] += 1
            elif qn == DEI_FY:
                vs = str(val).strip()
                if vs.isdigit():
                    fy_counter[int(vs)] += 1
            elif qn == DEI_DPD:
                ymd = _yyyymmdd(str(val))
                if ymd:
                    doc_dates.append(ymd)

    out = {}
    if fq_counter:
        out["fq"] = fq_counter.most_common(1)[0][0]
    if fy_counter:
        out["fy"] = fy_counter.most_common(1)[0][0]
    if doc_dates:
        out["doc_date"] = max(doc_dates)  # 多个的话取最新
    return out

def reorder_fields(r: dict) -> dict:
    """
    调整字段顺序，把 fq 放在 doc_date 前面；保持其它字段的常见顺序。
    未在列表中的键会按原相对顺序追加在末尾。
    """
    preferred_order = [
        "schema_version", "source_path", "ticker", "form", "year",
        "accno", "fy", "fq", "doc_date",   # ✅ fq 在 doc_date 之前
        "css_path", "language", "tokens", "created_at", "id",
        "section", "heading", "statement_hint", "text", "tag",
        "idx_source", "chunk_tok_start", "chunk_tok_end",
        "part", "item", "page_no", "page_anchor", "xpath"
    ]
    out = {}
    # 先放优先顺序
    for k in preferred_order:
        if k in r:
            out[k] = r[k]
    # 再放剩余字段（保持插入顺序）
    for k, v in r.items():
        if k not in out:
            out[k] = v
    return out

def merge_into_jsonl(jsonl_path: Path, dei: dict, *, overwrite=True) -> tuple[int,int]:
    """
    把 DEI 合并到 jsonl_path。
    overwrite=True 时：以 facts 为准覆盖原值；否则仅在为空/缺失时填充。
    返回 (读入行数, 写出行数)。
    """
    if not jsonl_path.exists():
        return 0, 0

    bak = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
    if bak.exists():
        bak.unlink()
    jsonl_path.replace(bak)

    n_in, n_out = 0, 0
    with bak.open("r", encoding="utf-8") as fin, jsonl_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            try:
                r = json.loads(line)
            except Exception:
                # 保底原样写回
                fout.write(line); n_out += 1; continue

            # 合并 fy/fq/doc_date
            for k in ("fy", "fq", "doc_date"):
                if k in dei and dei[k] is not None:
                    if overwrite or r.get(k) in (None, "", 0):
                        r[k] = dei[k]

            # 调整输出顺序（确保 fq 在 doc_date 之前）
            r = reorder_fields(r)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1

    # 需要可以删除备份：bak.unlink(missing_ok=True)
    return n_in, n_out

def main():
    updated = 0
    for text_path in ROOT.rglob("text.jsonl"):
        filing_dir = text_path.parent
        facts_path = filing_dir / "facts.jsonl"  # 你也可以换成 facts.parquet 的读取逻辑
        dei = load_dei_from_facts(facts_path)
        if not dei:
            continue

        n_text = merge_into_jsonl(text_path, dei, overwrite=True)
        facts_like_path = filing_dir / "facts_like.jsonl"
        n_like = (0, 0)
        if facts_like_path.exists():
            n_like = merge_into_jsonl(facts_like_path, dei, overwrite=True)

        updated += 1
        print(f"[ok] {filing_dir} -> text {n_text[0]}→{n_text[1]} lines; "
              f"facts_like {n_like[0]}→{n_like[1]} lines; DEI={dei}")

    print(f"✅ done. updated {updated} filings.")

if __name__ == "__main__":
    main()
