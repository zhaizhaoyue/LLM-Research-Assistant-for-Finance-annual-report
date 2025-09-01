#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import pandas as pd

CLEAN_ROOT_DEFAULT = Path("data/clean").resolve()
FACT_CANDIDATES = ("fact.jsonl", "fact.parquet")
SKIP_PREFIXES = ("fact", "text")     # 不处理以这些前缀命名的文件
TARGET_SUFFIX = ".jsonl"             # 只处理 jsonl（如需兼容 parquet，可再扩展）

def read_fq_from_fact(dirp: Path):
    """从 fact.jsonl 或 fact.parquet 读取 fq（容错 period_fq）。返回 (fq 或 None)。"""
    for name in FACT_CANDIDATES:
        fp = dirp / name
        if not fp.exists():
            continue
        if fp.suffix == ".jsonl":
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    fq = obj.get("fq") or obj.get("period_fq")
                    if isinstance(fq, str) and fq.strip():
                        return fq.strip()
        elif fp.suffix == ".parquet":
            try:
                df = pd.read_parquet(fp)
                for col in ("fq", "period_fq"):
                    if col in df.columns:
                        s = df[col].dropna().astype(str)
                        if not s.empty and s.iloc[0].strip():
                            return s.iloc[0].strip()
            except Exception:
                pass
    return None

def patch_one_jsonl(path: Path, fq_value: str, force: bool=False) -> int:
    """把 fq_value 写入单个 JSONL 文件。返回更新的行数。"""
    updated = 0
    out_lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                out_lines.append(line); continue
            try:
                obj = json.loads(line)
            except Exception:
                out_lines.append(line); continue

            cur = obj.get("fq") or obj.get("period_fq")
            need = force or (cur is None or (isinstance(cur, str) and not cur.strip()))
            if need:
                obj["fq"] = fq_value
                updated += 1
            out_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")

    if updated > 0:
        with path.open("w", encoding="utf-8") as f:
            f.writelines(out_lines)
    return updated

def main():
    ap = argparse.ArgumentParser(description="Propagate fq from clean/fact.* to other JSONL files in the same clean directory.")
    ap.add_argument("--root", default=str(CLEAN_ROOT_DEFAULT), help="clean 根目录（默认 data/clean）")
    ap.add_argument("--force", action="store_true", help="强制覆盖已有 fq")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    total_dirs = 0
    total_files = 0
    total_rows = 0

    # 遍历每个 filing 目录（包含 fact.jsonl / fact.parquet 的目录）
    for fact_path in list(root.rglob("fact.jsonl")) + list(root.rglob("fact.parquet")):
        dirp = fact_path.parent
        total_dirs += 1
        fq = read_fq_from_fact(dirp)
        if not fq:
            continue

        for target in dirp.glob("*" + TARGET_SUFFIX):
            name = target.name
            if any(name.startswith(pref) for pref in SKIP_PREFIXES):
                continue  # 跳过 fact*.jsonl / text*.jsonl
            rows = patch_one_jsonl(target, fq, force=args.force)
            total_rows += rows
            total_files += 1

    print(f"Done. Dirs scanned: {total_dirs}, Files touched: {total_files}, Rows updated: {total_rows}")

if __name__ == "__main__":
    main()
