from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

# -----------------------------
# Project paths & config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean_raw"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


CONFIG = load_config()

# -----------------------------
# Heuristics / Regex
# -----------------------------
SCALE_PATTERNS: Dict[str, float] = {
    r"\bin\s+millions\b": 1e6,
    r"\bin\s+thousands\b": 1e3,
    r"\bin\s+billions\b": 1e9,
}
CURRENCY_TOKENS = {
    "USD": {"usd", "us$", "$", "u.s. dollars", "us dollars", "dollars"},
    "CNY": {"cny", "rmb", "人民币", "元", "￥"},
    "EUR": {"eur", "€", "euro", "euros"},
}

# 列头/期间抽取（FY/Q1..Q4/年份）
PERIOD_HEADER_RE = re.compile(
    r"^(?:(?:fy|fye)\s*)?(?P<year>20\d{2})(?:\s*\(?audited\)?)?$|^(?P<q>q[1-4])(?:\s*(?P<year2>20\d{2}))?$",
    re.I,
)
PERCENT_RE = re.compile(r"^\s*([+-]?[0-9][0-9,]*?(?:\.[0-9]+)?)\s*%\s*$")
NUM_RE = re.compile(r"^\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*$")
PAREN_NEG_RE = re.compile(r"^\s*\((.+)\)\s*$")
ACC_RE = re.compile(r"(?P<acc>\b\d{10}-\d{2}-\d{6}\b)")  # 0000320193-24-000006
TYPE_RE = re.compile(r"\b(10-K|10-Q)\b", re.I)
Q_RE = re.compile(r"\bQ([1-4])\b", re.I)
FY_RE = re.compile(r"\bFY(?:\s*|_)?(20\d{2})\b", re.I)
YEAR_RE = re.compile(r"\b(20\d{2})\b")

# 新增：识别一行里“第一个非空文本”
def _first_nonempty(*cells: str) -> str:
    for c in cells:
        s = str(c).strip()
        if s:
            return s
    return ""

# 新增：按“行式期间”分段
ROW_PERIOD_RE = re.compile(
    r"^(?:FY\s*)?(?P<year>20\d{2})$|^(?P<q>Q[1-4])\s*(?P<year2>20\d{2})?$",
    re.I,
)

def _expand_period_by_row_sections(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, Dict[str, Optional[str]]]]]:
    """
    适配“行式期间”版式（如：一行'2024'，下面多行数据；再一行'2023'…）
    返回 (value_df, period_meta)，其中 value_df 至少含 ['row_label_raw','period_col','raw_value']。
    """
    if df.empty:
        return pd.DataFrame(), []

    rows = []
    period_meta = []          # 这里我们用“虚拟列名”做 period 键
    current_key = None
    current_meta = None
    counter = 0

    for _, row in df.iterrows():
        # 把整行拉平成字符串列表
        vals = [str(v).strip() for v in row.tolist()]
        # 查首个非空文本
        head = _first_nonempty(*vals[:2])  # 通常前两列之一是标签/期间
        # 是不是期间行？
        m = ROW_PERIOD_RE.match(head)
        if m and all(v == "" for v in vals[2:]):  # 期间行一般其余列为空或很少有数
            year = m.group("year") or m.group("year2")
            q = m.group("q")
            fiscal_period = (q.upper() if q else ("FY" if year else None))
            label = head
            # 新开一个“虚拟列”作为该期间的 period_col
            counter += 1
            current_key = f"__row_period__{counter}"
            current_meta = {"fiscal_year": year, "fiscal_period": fiscal_period, "period_label": label}
            period_meta.append((current_key, current_meta))
            continue

        # 数据行：需要有一个当前期间
        if current_key:
            # 标签：前两列里第一个非空
            row_label = head
            if not row_label:
                continue

            # 数值：从右往左找第一个能被 _parse_number 解析的单元
            raw_val = None
            for v in reversed(vals[1:]):    # 跳过第一列标签
                if _parse_number(v) is not None:
                    raw_val = v
                    break
            if raw_val is None:
                continue

            rows.append({
                "row_label_raw": row_label,
                "period_col": current_key,
                "raw_value": raw_val,
            })

    if not rows:
        return pd.DataFrame(), []

    val_df = pd.DataFrame(rows)[["row_label_raw", "period_col", "raw_value"]]
    return val_df, period_meta


def _extract_accession_and_type(text: str) -> Tuple[Optional[str], Optional[str]]:
    """从文本里抽取 accession number 和报表类型 10-K/10-Q"""
    acc = None
    m = ACC_RE.search(text)
    if m:
        acc = m.group("acc")
    typ = None
    m2 = TYPE_RE.search(text)
    if m2:
        typ = m2.group(1).upper()
    return acc, typ

def _guess_year_quarter(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    尽力从文本里猜年份和季度:
    - 年份: 20xx
    - 季度: Q1..Q4（没有就 None）
    """
    year = None
    q = None
    mfy = FY_RE.search(text)
    if mfy:
        year = mfy.group(1)
    else:
        my = YEAR_RE.search(text)
        if my:
            year = my.group(1)

    mq = Q_RE.search(text)
    if mq:
        q = f"Q{mq.group(1)}"
    return year, q

def _build_report_id_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    依据路径/文件名生成 (report_type, report_id)
    优先使用: <TYPE>_<ACCESSION>
    回退: <TYPE>_<YEAR>[_Qn]
    再回退: REPORT_<stem>
    """
    blob = " ".join([path.name, path.stem, str(path.parent), str(path.parts)])
    acc, rtype = _extract_accession_and_type(blob)

    if acc and rtype:
        return rtype, f"{rtype}_{acc}"

    # 没有 accession，用 年份/季度 组装
    year, q = _guess_year_quarter(blob)
    if rtype and year:
        if q:
            return rtype, f"{rtype}_{year}_{q}"
        return rtype, f"{rtype}_{year}"

    # 最后兜底
    return None, f"REPORT_{_safe_stem(path)}"


@dataclass
class TableContext:
    source_path: Path
    table_idx: int
    statement_type: Optional[str] = None
    scale: Optional[float] = None
    unit: Optional[str] = None
    currency: Optional[str] = None
    company_id: Optional[str] = None
    report_id: Optional[str] = None


# -----------------------------
# IO utils
# -----------------------------
def _iter_csv_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix.lower() == ".csv" or p.name.lower().endswith(".csv.gz")):
            yield p


def _safe_stem(p: Path) -> str:
    return p.stem.replace(" ", "_").replace("/", "_")


def _read_csv(path: Path, *, sep: str, encoding: str, header: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        header=header,
        engine="python",
    ).fillna("")
    # 清理不间断空格与首尾空白
    cleaner = lambda x: str(x).replace("\u00a0", " ").strip()
    # pandas>=2.2 推荐 DataFrame.map
    if hasattr(df, "map"):
        df = df.map(cleaner)
    else:  # 兼容低版本
        df = df.applymap(cleaner)
    return df


def _split_on_blank_rows(df: pd.DataFrame) -> List[pd.DataFrame]:
    is_blank = df.apply(lambda r: all((str(v).strip() == "" for v in r)), axis=1)
    if not is_blank.any():
        return [df]
    parts: List[pd.DataFrame] = []
    start = 0
    for i, blank in enumerate([*is_blank.tolist(), True]):  # sentinel
        if blank:
            if i > start:
                parts.append(df.iloc[start:i].copy())
            start = i + 1
    return [p for p in parts if not p.empty]


def _drop_empty(df: pd.DataFrame, *, drop_rows: bool, drop_cols: bool) -> pd.DataFrame:
    out = df
    if drop_rows:
        out = out.loc[~out.apply(lambda r: all((str(v).strip() == "" for v in r)), axis=1)]
    if drop_cols:
        out = out.loc[:, ~out.apply(lambda c: all((str(v).strip() == "" for v in c)), axis=0)]
    return out


# -----------------------------
# Header / metadata detection
# -----------------------------
def _maybe_promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """当列名大多是 0/1/2/3 时，自动把首行提升为表头。"""
    cols = [str(c).strip() for c in df.columns]
    ratio_num = sum(c.isdigit() for c in cols) / max(1, len(cols))
    if ratio_num < 0.6 or df.empty:
        return df
    first = df.iloc[0].astype(str).str.strip()
    if (first != "").mean() < 0.5:
        return df
    df2 = df.iloc[1:].copy()
    df2.columns = first.tolist()
    return df2


def _detect_scale(text_blob: str) -> Optional[float]:
    t = text_blob.lower()
    for pat, val in SCALE_PATTERNS.items():
        if re.search(pat, t):
            return val
    return None


def _detect_currency(text_blob: str) -> Optional[str]:
    t = text_blob.lower()
    for code, toks in CURRENCY_TOKENS.items():
        for tok in toks:
            if tok in t:
                return code
    return None


def _infer_statement_type_from_path(path: Path) -> Optional[str]:
    """非常基础的报表类型推断：文件名/父目录名里搜关键词。"""
    name = " ".join([path.stem, path.name, str(path.parent.name)]).lower()
    if any(k in name for k in ["cash flow", "cashflow"]):
        return "CF"
    if any(k in name for k in ["balance sheet", "financial position", "assets", "liabilities", "equity"]):
        return "BS"
    if any(k in name for k in ["income statement", "p&l", "profit", "operations", "net sales", "gross margin"]):
        return "IS"
    return None
IS_HINT = {"total net sales","revenue","cost of sales","operating income"}
BS_HINT = {"total assets","total liabilities","shareholders’ equity"}
CF_HINT = {"net cash provided by","cash generated by operating activities","purchases of marketable securities"}

def _infer_statement_type_from_rows(sample_rows: list[str]) -> Optional[str]:
    t = " ".join(sample_rows).lower()
    score = {"IS": sum(k in t for k in IS_HINT),
             "BS": sum(k in t for k in BS_HINT),
             "CF": sum(k in t for k in CF_HINT)}
    top = max(score, key=score.get)
    return top if score[top] >= 2 else None


# -----------------------------
# Number normalization
# -----------------------------
def _parse_number(cell: str) -> Optional[float]:
    if cell == "":
        return None
    m = PERCENT_RE.match(cell)  # "12.3%"
    if m:
        try:
            return float(m.group(1).replace(",", "")) / 100.0
        except Exception:
            return None
    m = PAREN_NEG_RE.match(cell)  # "(1,234)"
    s = m.group(1) if m else cell
    if NUM_RE.match(s):
        try:
            return float(s.replace(",", "")) * (-1.0 if m else 1.0)
        except Exception:
            return None
    return None


# -----------------------------
# Normalize: columns → long
# -----------------------------
def _expand_period_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, Dict[str, Optional[str]]]]]:
    """
    返回 (value_df, period_meta)
    - value_df: [row_label_raw, <period columns>]
    - period_meta: [(colname, {fiscal_year, fiscal_period, period_label}), ...]
    """
    if df.empty:
        return df, []
    if df.shape[1] < 2:  # 只有 1 列，大概率不是可用数据表
        return pd.DataFrame(), []

    first_col = df.columns[0]
    value_cols: List[str] = []
    meta: List[Tuple[str, Dict[str, Optional[str]]]] = []

    # 忽略的辅助列（全部小写对齐）
    IGNORE_COLS = {"__table_id__", "__page__", "__source__", "__file__"}

    for c in df.columns[1:]:
        col_lc = str(c).strip().lower()
        if col_lc in IGNORE_COLS:
           continue

        label = str(c).strip()
        m = PERIOD_HEADER_RE.match(label)
        if m:
            year = m.group("year") or m.group("year2")
            q = m.group("q")
            fiscal_period = (q.upper() if q else ("FY" if year else None))
            period_label = label
            meta.append((c, {"fiscal_year": year, "fiscal_period": fiscal_period, "period_label": period_label}))
            value_cols.append(c)
        else:
            meta.append((c, {"fiscal_year": None, "fiscal_period": None, "period_label": label}))
            value_cols.append(c)

    keep = [first_col] + value_cols
    return df[keep].rename(columns={first_col: "row_label_raw"}), meta


def normalize_table(df: pd.DataFrame, ctx: TableContext) -> pd.DataFrame:
    # 1) 规模/币种探测
    probe_text = " ".join([" ".join(map(str, df.columns))] + [" ".join(map(str, df.iloc[i].tolist())) for i in range(min(len(df), 3))])
    scale = ctx.scale or _detect_scale(probe_text)
    currency = ctx.currency or _detect_currency(probe_text)

    # 2) 优先：列头期间
    val_df, period_meta = _expand_period_columns(df)

    # 3) 兜底：行式期间
    if val_df.empty or not period_meta:
        val_df, period_meta = _expand_period_by_row_sections(df)

    if val_df.empty:
        return pd.DataFrame()

    melted = val_df.copy()
    if "period_col" not in melted.columns:  # 列头路径返回的是 row_label_raw + 各期间列，需要 melt
        melted = val_df.melt(id_vars=["row_label_raw"], var_name="period_col", value_name="raw_value")

    period_map = {c: m for c, m in period_meta}
    melted["value"] = melted["raw_value"].apply(_parse_number)
    melted = melted.loc[melted["value"].notna()]  # 只留数值

    melted["unit"] = ctx.unit
    melted["currency"] = currency
    melted["scale"] = scale
    melted["statement_type"] = ctx.statement_type

    melted["period_label"] = melted["period_col"].map(lambda c: (period_map.get(c) or {}).get("period_label"))
    melted["fiscal_year"]   = melted["period_col"].map(lambda c: (period_map.get(c) or {}).get("fiscal_year"))
    melted["fiscal_period"] = melted["period_col"].map(lambda c: (period_map.get(c) or {}).get("fiscal_period"))

    cid = ctx.company_id or ""
    rid = ctx.report_id or ""
    melted["source_path"] = f"{cid}/{rid}".strip("/")
    melted["table_id"] = f"{_safe_stem(ctx.source_path)}__t{ctx.table_idx}"
    melted["company_id"] = ctx.company_id
    melted["report_id"] = ctx.report_id
    v = melted["value"]
    year_like = v.notna() & (v.fillna(0) % 1 == 0) & v.between(1900, 2100)
    melted = melted.loc[ melted["value"].notna() & ~((melted["row_label_raw"].str.strip()=="") & year_like) ]


    cols = ["company_id","report_id","source_path","table_id","row_label_raw","period_label",
            "fiscal_year","fiscal_period","value","unit","currency","scale","statement_type","raw_value"]
    return melted[cols]


# -----------------------------
# Small helpers
# -----------------------------
def _derive_ids_from_relpath(rel_parts: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """从相对路径的前两段推导 company_id / report_id。"""
    if len(rel_parts) >= 2:
        return rel_parts[0], rel_parts[1]
    if len(rel_parts) == 1:
        return None, rel_parts[0]
    return None, None

def _derive_company_and_report(path: Path, input_root: Optional[Path]) -> tuple[str, str]:
    """
    规则：
    1) 优先用 input_root 之后的相对路径：
         <company>/<year>/<report>/file.csv → (company, report)
         <company>/<report>/file.csv        → (company, report)
    2) 若只有 <company>/file.csv → (company, REPORT_<stem>)
    3) 若无法相对化或路径太短，则：
         - company 用上上级或上级目录名
         - report 用 _build_report_id_from_path(path) 的结果或文件名
    4) 最后兜底：company 从文件名里猜（AAPL、MSFT、CIK 等），都没有则 "UNKNOWN"
    """
    # 允许 path 不在 input_root 之下
    rel_parts: tuple[str, ...] = ()
    if input_root is not None:
        try:
            rel_parts = path.relative_to(input_root).parts
        except Exception:
            rel_parts = ()

    company_id: Optional[str] = None
    report_id: Optional[str] = None

    # --- 主路径推断 ---
    if len(rel_parts) >= 3:
        # 形如 company/year/report/file.csv
        company_id = rel_parts[0]
        # 第二层经常是年份，第三层才是 report 目录
        report_id = rel_parts[2]
    elif len(rel_parts) == 2:
        # 形如 company/report/file.csv  或  company/year/file.csv
        company_id = rel_parts[0]
        # 尝试从完整 path 中精准识别 report（10-Q_.../10-K_...）
        rtype, rid = _build_report_id_from_path(path)
        report_id = rid or rel_parts[1]
    elif len(rel_parts) == 1:
        # 形如 company/file.csv
        company_id = rel_parts[0]

    # --- 备份：不在 input_root 下或层级不足时，从父目录兜底 ---
    if company_id is None or report_id is None:
        parent_parts = path.parent.parts
        if company_id is None:
            if len(parent_parts) >= 2:
                company_id = parent_parts[-2]
            elif len(parent_parts) >= 1:
                company_id = parent_parts[-1]

        if report_id is None:
            rtype, rid = _build_report_id_from_path(path)
            if rid:
                report_id = rid
            else:
                # 如果父目录存在，优先用父目录；否则用文件名
                report_id = parent_parts[-1] if parent_parts else path.stem

    # --- 最终兜底：从文件名里猜测 company（如 AAPL_2024_xxx.csv）---
    if not company_id:
        m = re.match(r"([A-Za-z0-9._-]+)[-_].*", path.stem)
        if m:
            company_id = m.group(1)
    if not company_id:
        company_id = "UNKNOWN"

    # 规范化：去两端空白，常见股票代码转大写
    company_id = str(company_id).strip()
    if re.fullmatch(r"[A-Za-z]{1,10}", company_id):
        company_id = company_id.upper()

    report_id = str(report_id).strip()
    if report_id == "":
        report_id = f"REPORT_{path.stem}"

    return company_id, report_id

# -----------------------------
# Pipeline
# -----------------------------
def process_file(
    path: Path,
    *,
    sep: str,
    encoding: str,
    header: Optional[int],
    split_on_blank_rows: bool,
    drop_empty_rows: bool,
    drop_empty_cols: bool,
    statement_hint: Optional[str],
    input_root: Optional[Path] = None,
) -> List[pd.DataFrame]:
    # 1) 读取并（必要时）提升首行为表头
    df = _read_csv(path, sep=sep, encoding=encoding, header=header)
    df = _maybe_promote_header(df)

    # 2) 切分成子表（避免变量名与路径 parts 冲突）
    table_parts = _split_on_blank_rows(df) if split_on_blank_rows else [df]
    indexed_parts: List[Tuple[int, pd.DataFrame]] = list(enumerate(table_parts))

    # 3) company_id / report_id 推导（稳健、不会覆盖 table_parts）
    try:
        company_id, report_id = _derive_company_and_report(path, input_root)
    except NameError:
        # 如果你还没定义 _derive_company_and_report，可先用简化回退：
        # ——优先相对 input_root，否则用父目录/文件名兜底
        try:
            rel_parts = path.relative_to(input_root).parts if input_root else ()
        except Exception:
            rel_parts = ()
        if len(rel_parts) >= 2:
            company_id = rel_parts[0]
        else:
            parent_parts = path.parent.parts
            company_id = parent_parts[-2] if len(parent_parts) >= 2 else (parent_parts[-1] if parent_parts else "UNKNOWN")
        _rt, rid = _build_report_id_from_path(path)
        report_id = rid or (rel_parts[1] if len(rel_parts) >= 2 else path.stem)

    # 4) 语句类型：显式参数优先，其次基于路径的推断
    stype_hint = statement_hint or _infer_statement_type_from_path(path)
    
    # 5) 逐子表归一化
    out: List[pd.DataFrame] = []
    for idx, t in indexed_parts:
        t2 = _drop_empty(t, drop_rows=drop_empty_rows, drop_cols=drop_empty_cols)
        if t2.empty:
            continue
        ctx = TableContext(
            source_path=path,
            table_idx=idx,
            statement_type=stype_hint,
            company_id=company_id,
            report_id=report_id,
        )
        norm = normalize_table(t2, ctx)
        if not norm.empty:
            out.append(norm)

    return out


def process_dir(dir_path: Path, **kwargs) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for f in _iter_csv_files(dir_path):
        try:
            frames.extend(process_file(f, **kwargs))
        except Exception as e:
            print(f"[WARN] Failed {f}: {e}")
    return [f for f in frames if f is not None and not f.empty]


def _mirror_write_for_file(
    src_path: Path,
    frames: List[pd.DataFrame],
    *,
    input_root: Path,
    out_root: Path,
    write_jsonl: bool = True,
) -> dict:
    """
    根据 input_root 计算相对路径，并将输出写到 out_root/相对路径.parquet（可选 .jsonl）
    """
    rel = src_path.relative_to(input_root)
    rel_posix = PurePosixPath(*rel.parts)
    dest = out_root / rel_posix.with_suffix(".parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        out_df = pd.DataFrame(
            columns=[
                "company_id",
                "report_id",
                "source_path",
                "table_id",
                "row_label_raw",
                "period_label",
                "fiscal_year",
                "fiscal_period",
                "value",
                "unit",
                "currency",
                "scale",
                "statement_type",
                "raw_value",
            ]
        )
    else:
        out_df = pd.concat(frames, ignore_index=True)

    # 主存 Parquet
    out_df.to_parquet(dest, index=False)

    # 可选 JSONL（便于 VSCode / RAG 采样）
    if write_jsonl:
        out_df.to_json(dest.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)

    return {"file": str(src_path), "rows": int(len(out_df)), "out": str(dest)}


# -----------------------------
# Programmatic API
# -----------------------------
def run(
    input_path: str | Path = DEFAULT_INPUT_DIR,
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    no_header: bool = False,
    split_on_blank_rows: bool = False,
    keep_empty_rows: bool = False,
    keep_empty_cols: bool = False,
    statement_hint: str | None = None,
    write_jsonl: bool = True,
) -> None:
    """
    供 Python 内直接调用：
        from cleaning.table_clean import run
        run("data/processed", "data/clean_raw", write_jsonl=True)
    """
    in_path = Path(input_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    header = None if no_header else 0
    drop_rows = not keep_empty_rows
    drop_cols = not keep_empty_cols

    if in_path.is_dir():
        stats = []
        for f in _iter_csv_files(in_path):
            frames = process_file(
                f,
                sep=sep,
                encoding=encoding,
                header=header,
                split_on_blank_rows=split_on_blank_rows,
                drop_empty_rows=drop_rows,
                drop_empty_cols=drop_cols,
                statement_hint=statement_hint,
                input_root=in_path,  # 用目录作为 root 来推导 company/report
            )
            stat = _mirror_write_for_file(
                f,
                frames,
                input_root=in_path,  # 与上面一致
                out_root=out_root,
                write_jsonl=write_jsonl,
            )
            print(stat)
            stats.append(stat)
        total_rows = sum(s["rows"] for s in stats)
        print({"files": len(stats), "total_rows": int(total_rows), "out_dir": str(out_root.resolve())})
        return

    # 单文件：input_root 用其父目录，镜像写出到 out_root
    frames = process_file(
        in_path,
        sep=sep,
        encoding=encoding,
        header=header,
        split_on_blank_rows=split_on_blank_rows,
        drop_empty_rows=drop_rows,
        drop_empty_cols=drop_cols,
        statement_hint=statement_hint,
        input_root=in_path.parent,
    )
    stat = _mirror_write_for_file(
        in_path,
        frames,
        input_root=in_path.parent,
        out_root=out_root,
        write_jsonl=write_jsonl,
    )
    print(stat)


# -----------------------------
# CLI
# -----------------------------
def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="Parse+Normalize CSV → clean_raw (mirror output per file).")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT_DIR), help="CSV file or directory")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Mirror output root directory")
    ap.add_argument("--sep", type=str, default=",")
    ap.add_argument("--encoding", type=str, default="utf-8")
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--split-on-blank-rows", action="store_true")
    ap.add_argument("--keep-empty-rows", action="store_true")
    ap.add_argument("--keep-empty-cols", action="store_true")
    ap.add_argument("--write-jsonl", action="store_true", help="Also write JSONL alongside parquet")
    ap.add_argument("--statement-hint", type=str, default=None)
    args = ap.parse_args()

    write_jsonl_flag = args.write_jsonl or _env_flag("WRITE_JSONL", True)

    run(
        input_path=args.input,
        out_dir=args.out_dir,
        sep=args.sep,
        encoding=args.encoding,
        no_header=args.no_header,
        split_on_blank_rows=args.split_on_blank_rows,
        keep_empty_rows=args.keep_empty_rows,
        keep_empty_cols=args.keep_empty_cols,
        statement_hint=args.statement_hint,
        write_jsonl=write_jsonl_flag,
    )


if __name__ == "__main__":
    main_cli()
