# src/utils/chunking.py
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import tiktoken  # 精确 token 计数（可选）
except Exception:  # pragma: no cover
    tiktoken = None

# ---- 你的 schema 类型 ----
from src.indexer.schema import (
    WithMeta,
    TextChunk,
    CleanedTable,
    FilingForm,
)

# =========================
# 简易 tokenizer（tiktoken 不在时的回退）
# =========================
_WORD_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[%$€¥]|[\u4e00-\u9fff]",
    re.UNICODE,
)


def _simple_tokenize(s: str) -> List[str]:
    return _WORD_RE.findall(s)


def get_tokenizer(model_name: str = "cl100k_base") -> Callable[[str], int]:
    """返回一个函数：给定文本→返回 token 数。
    优先用 tiktoken，否则用粗略分词长度。
    """
    if tiktoken is not None:
        enc = tiktoken.get_encoding(model_name)

        def _count(s: str) -> int:
            return len(enc.encode(s))

        return _count

    def _fallback(s: str) -> int:
        return len(_simple_tokenize(s))

    return _fallback


# =========================
# 噪声判定 & 段落切分
# =========================
_NOISE_PATTERNS = [
    r"table of contents",
    r"index of exhibits",
    r"forward-looking statements",
    r"safe harbor",
    r"signature[s]?\s*[-–—]*$",
]


def is_noise(paragraph: str) -> bool:
    p = paragraph.strip().lower()
    if not p or len(p) < 3:
        return True
    for pat in _NOISE_PATTERNS:
        if re.search(pat, p):
            return True
    return False


def _split_by_seps(s: str, seps: Sequence[str]) -> List[str]:
    out = [s]
    for sep in seps:
        nxt: List[str] = []
        for x in out:
            nxt.extend(x.split(sep))
        out = nxt
    return out


# =========================
# 文本 → TextChunk 切块
# =========================
def chunk_text_to_textchunks(
    *,
    text: str,
    source_path: Optional[str],
    base_meta: Dict[str, Any] | None = None,
    section: Optional[str] = None,
    heading: Optional[str] = None,
    language: Optional[str] = "en",
    max_tokens: int = 350,
    overlap: int = 80,
    paragraph_seps: Sequence[str] = ("\n\n", "\n"),
    tokenizer_name: str = "cl100k_base",
) -> List[TextChunk]:
    """将一段长文本分块为若干 TextChunk（与你的 schema 完全兼容）。

    - 以段落为单位尽量合并，超过阈值再切；
    - 对极长段落做滑窗切分，保留 overlap；
    - 会过滤常见噪声段（目录、签名页等）。
    """
    count_tokens = get_tokenizer(tokenizer_name)
    meta = dict(base_meta or {})
    chunks: List[TextChunk] = []

    # 1) 拆成段落并过滤噪声
    parts = _split_by_seps(text, paragraph_seps)
    parts = [p.strip() for p in parts if p.strip()]
    parts = [p for p in parts if not is_noise(p)]

    # 2) 按 token 上限拼段
    buf: List[str] = []
    buf_tokens = 0

    def flush_buffer():
        nonlocal buf, buf_tokens
        if not buf:
            return
        chunk_text = "\n\n".join(buf).strip()
        chunks.append(
            TextChunk(
                text=chunk_text,
                source_path=source_path,
                section=section,
                heading=heading,
                language=language,
                extra_meta=meta,
                # 其他 WithMeta 字段从 base_meta 里来（如 ticker/form/year/accno 等）
                ticker=meta.get("ticker"),
                form=meta.get("form"),
                year=meta.get("year"),
                accno=meta.get("accno"),
                company=meta.get("company"),
                cik=meta.get("cik"),
                doc_date=meta.get("doc_date"),
                tokens=count_tokens(chunk_text),
            )
        )
        buf, buf_tokens = [], 0

    for para in parts:
        t = count_tokens(para)
        # 正常累积
        if buf_tokens + t <= max_tokens:
            buf.append(para)
            buf_tokens += t
            continue

        # 输出已有缓存
        flush_buffer()

        # 如果单段就超长 → 滑窗切
        if t > max_tokens:
            windows = _sliding_windows_by_tokens(
                para, max_tokens=max_tokens, overlap=overlap, count_tokens=count_tokens
            )
            for w in windows:
                chunks.append(
                    TextChunk(
                        text=w,
                        source_path=source_path,
                        section=section,
                        heading=heading,
                        language=language,
                        extra_meta=meta,
                        ticker=meta.get("ticker"),
                        form=meta.get("form"),
                        year=meta.get("year"),
                        accno=meta.get("accno"),
                        company=meta.get("company"),
                        cik=meta.get("cik"),
                        doc_date=meta.get("doc_date"),
                        tokens=count_tokens(w),
                    )
                )
        else:
            # 否则把该段放入新的 buf
            buf = [para]
            buf_tokens = t

    flush_buffer()
    return chunks


def _sliding_windows_by_tokens(
    s: str,
    *,
    max_tokens: int,
    overlap: int,
    count_tokens: Callable[[str], int],
) -> List[str]:
    """将超长段落按 token 数做滑窗切分。"""
    if max_tokens <= 0:
        return [s]

    # 近似做法：对字符级滑窗进行二分调参使 token 数接近 max_tokens
    # 简化实现：先粗分，再微调边界
    windows: List[str] = []
    start = 0
    step_guess = max(1, len(s) // (max(1, len(s) // (max_tokens * 3))))
    while start < len(s):
        end = min(len(s), start + step_guess)
        # 扩到不超过 max_tokens
        seg = s[start:end]
        while end < len(s) and count_tokens(seg) < max_tokens:
            end += 1
            seg = s[start:end]
        # 回退到不超过上限
        while end > start and count_tokens(seg) > max_tokens:
            end -= 1
            seg = s[start:end]
        if not seg:
            break
        windows.append(seg)
        # 计算 overlap 的“字符级”近似
        # 这里用 token 比例估计字符 overlap
        tok = count_tokens(seg)
        char_overlap = int(len(seg) * (overlap / max(1, tok)))
        start = end - max(0, char_overlap)
        if start <= len(s) and start == end:
            start += 1
    return windows


# =========================
# CleanedTable → TextChunk（序列化表格）
# =========================
def cleaned_table_to_textchunk(
    tbl: CleanedTable,
    *,
    source_path: Optional[str],
    fmt: str = "csv",  # 'csv' | 'md'
    keep_rows: int = 40,
    keep_cols: int = 8,
) -> TextChunk:
    """将规范化表格（CleanedTable）压缩为一个 TextChunk。
    - 目的：让 LLM 可以把表格作为“可读文本”使用；
    - 会把单位/币种/期间等通过 extra_meta 携带；
    - 标记 extra_meta['kind'] = 'table'，后续检索/路由更方便。
    """
    df = _cleaned_table_to_df(tbl)

    # 限列
    if keep_cols and df.shape[1] > keep_cols:
        df = df.iloc[:, :keep_cols]

    # 限行：优先保留数值密集的行
    if keep_rows and df.shape[0] > keep_rows:
        df = _top_value_rows(df, keep_rows)

    payload = _serialize_df(df, fmt=fmt)

    header_parts = []
    if tbl.title:
        header_parts.append(f"Table: {tbl.title}")
    if tbl.year:
        header_parts.append(f"Year: {tbl.year}")
    if tbl.ticker:
        header_parts.append(f"Ticker: {tbl.ticker}")
    if tbl.form:
        header_parts.append(f"Form: {tbl.form}")
    if header_parts:
        payload = "# " + " | ".join(header_parts) + "\n" + payload

    extra = dict(tbl.extra_meta or {})
    extra.update(
        {
            "kind": "table",
            "statement_hint": tbl.statement_hint.value if tbl.statement_hint else None,
            "units": tbl.units,
            "scales": tbl.scales,
        }
    )

    return TextChunk(
        text=payload,
        source_path=source_path,
        section=extra.get("section"),
        heading=tbl.title,
        language="en",
        extra_meta=extra,
        ticker=tbl.ticker,
        form=tbl.form,
        year=tbl.year,
        accno=tbl.accno,
        company=tbl.company,
        cik=tbl.cik,
        doc_date=tbl.doc_date,
        tokens=get_tokenizer()(payload),
    )


def _cleaned_table_to_df(tbl: CleanedTable) -> pd.DataFrame:
    headers = list(tbl.headers or [])
    rows = list(tbl.rows or [])
    df = pd.DataFrame(rows, columns=headers if headers else None)
    # 尽量把数字串转成数字（不破坏空值）
    for c in df.columns:
        df[c] = _maybe_to_number(df[c])
    return df


def _maybe_to_number(s: pd.Series) -> pd.Series:
    def parse(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return x
        xs = str(x).strip().replace(",", "")
        # 允许百分号等留给 LLM 识别，不强转
        try:
            return float(xs)
        except Exception:
            return x

    return s.map(parse)


def _serialize_df(df: pd.DataFrame, *, fmt: str = "csv") -> str:
    df2 = df.fillna("")
    if fmt == "md":
        # 简洁 Markdown
        header = "| " + " | ".join(map(str, df2.columns)) + " |"
        sep = "| " + " | ".join(["---"] * df2.shape[1]) + " |"
        rows = [
            "| " + " | ".join(map(lambda x: str(x).replace("\n", " "), map(str, r))) + " |"
            for _, r in df2.iterrows()
        ]
        return "\n".join([header, sep, *rows])
    # 默认 CSV：更利于按列名索引
    return df2.to_csv(index=False)


def _top_value_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    scores = []
    for i, row in df.iterrows():
        non_empty = (row.astype(str) != "").sum()
        numeric_like = 0
        for v in row.values:
            try:
                float(str(v).replace(",", ""))
                numeric_like += 1
            except Exception:
                pass
        score = numeric_like * 2 + non_empty
        scores.append((i, score))
    idx = [i for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:n]]
    return df.loc[idx]


# =========================
# 批量入口（供 indexer 使用）
# =========================
def make_text_chunks_from_records(
    records: Iterable[Dict[str, Any]],
    *,
    source_field: str = "source_path",
    text_field: str = "text",
    meta_fields: Sequence[str] = ("ticker", "form", "year", "accno", "company", "cik", "doc_date", "section", "heading"),
    max_tokens: int = 350,
    overlap: int = 80,
    tokenizer_name: str = "cl100k_base",
) -> List[TextChunk]:
    """将清洗后的文本记录（如 text.jsonl 的行）批量切成 TextChunk 列表。"""
    out: List[TextChunk] = []
    for r in records:
        src = r.get(source_field)
        txt = str(r.get(text_field, "") or "")
        base_meta = {k: r.get(k) for k in meta_fields if k in r}
        out.extend(
            chunk_text_to_textchunks(
                text=txt,
                source_path=src,
                base_meta=base_meta,
                section=r.get("section"),
                heading=r.get("heading"),
                language=r.get("language", "en"),
                max_tokens=max_tokens,
                overlap=overlap,
                tokenizer_name=tokenizer_name,
            )
        )
    return out


def make_table_chunks_from_cleaned(
    tables: Iterable[CleanedTable],
    *,
    source_path_resolver: Callable[[CleanedTable], Optional[str]] | None = None,
    fmt: str = "csv",
    keep_rows: int = 40,
    keep_cols: int = 8,
) -> List[TextChunk]:
    """将一批 CleanedTable 转换为 TextChunk（1:1，每张表一个片段）。"""
    out: List[TextChunk] = []
    for tbl in tables:
        src = source_path_resolver(tbl) if source_path_resolver else tbl.source_path
        out.append(
            cleaned_table_to_textchunk(
                tbl,
                source_path=src,
                fmt=fmt,
                keep_rows=keep_rows,
                keep_cols=keep_cols,
            )
        )
    return out


# =========================
# 自检
# =========================
if __name__ == "__main__":  # 手动快速测试
    sample_text = (
        "Management Discussion and Analysis\n\n"
        "Revenue increased by 12% year-over-year mainly due to iPhone sales.\n\n"
        "Forward-looking statements ...\n\n"
        "Operating margin improved."
    )
    chunks = chunk_text_to_textchunks(
        text=sample_text,
        source_path="US_AAPL_2024_10-K_md&a.txt",
        base_meta={"ticker": "AAPL", "form": FilingForm.TEN_K, "year": 2024},
        section="MD&A",
        max_tokens=40,
        overlap=10,
    )
    print(f"[demo] text chunks: {len(chunks)}")

    tbl = CleanedTable(
        ticker="AAPL",
        form=FilingForm.TEN_K,
        year=2024,
        title="Income Statement",
        headers=["Item", "FY2023", "FY2024", "Unit"],
        rows=[
            ["Revenue", 383_285, 390_000, "USD mn"],
            ["COGS", 213_000, 220_500, "USD mn"],
            ["Gross Profit", 170_285, 169_500, "USD mn"],
            ["SG&A", 25_000, 25_500, "USD mn"],
            ["R&D", 26_251, 27_200, "USD mn"],
        ],
        units={"FY2023": "USD", "FY2024": "USD"},
        scales={"FY2023": 1e6, "FY2024": 1e6},
    )
    tchunk = cleaned_table_to_textchunk(tbl, source_path="US_AAPL_2024_10-K_is.csv", fmt="csv")
    print(f"[demo] table chunk tokens: {tchunk.tokens}")
