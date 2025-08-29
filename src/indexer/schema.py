from __future__ import annotations
from math import isnan
from datetime import datetime, UTC, date
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Dict, Iterable, Type, TypeVar, TypeAlias, Union
from pathlib import Path
from uuid import UUID, uuid4
from enum import Enum
import json
from pydantic import BaseModel, Field, ConfigDict, AliasChoices, field_validator, model_validator
import sys
import argparse
import pandas as pd



def dump_parquet(path: Path, records: Iterable):
    if pd is None:
        raise RuntimeError("pandas 未安装，无法导出 parquet。请 `pip install pandas pyarrow`")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in records])
    df.to_parquet(path, index=False)

def dump_csv(path: Path, records: Iterable):
    if pd is None:
        raise RuntimeError("pandas 未安装，无法导出 CSV。请 `pip install pandas`")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in records])
    df.to_csv(path, index=False, encoding="utf-8")

def process_tree(
    in_root: Path,
    out_root: Path,
    *,
    export_format: str = "jsonl",   # jsonl | parquet | csv
    overwrite: bool = False,
    verbose: bool = True,
) -> dict:
    """
    从 in_root 递归读取 *.jsonl -> 解析/清洗 -> 输出到 out_root 的镜像路径。
    仅处理 pick_model_for_file 能识别的文件，其它跳过。
    """
    assert export_format in {"jsonl", "parquet", "csv"}
    stats = {"ok": 0, "skip_no_model": 0, "skip_exists": 0, "fail": 0}

    for p in sorted(in_root.rglob("*.jsonl")):
        Model = pick_model_for_file(p)
        if Model is None:
            stats["skip_no_model"] += 1
            if verbose:
                print(f"[skip] {p} (no model router)")
            continue

        rel = p.relative_to(in_root)               # 镜像路径
        out_path = (out_root / rel)

        if export_format == "parquet":
            out_path = out_path.with_suffix(".parquet")
        elif export_format == "csv":
            out_path = out_path.with_suffix(".csv")
        else:
            # jsonl：保持 .jsonl
            pass

        if out_path.exists() and not overwrite:
            stats["skip_exists"] += 1
            if verbose:
                print(f"[skip] {out_path} (exists, use --overwrite to force)")
            continue

        try:
            recs = load_jsonl(p, Model)
            if export_format == "jsonl":
                dump_jsonl(out_path, recs)
            elif export_format == "parquet":
                dump_parquet(out_path, recs)
            else:
                dump_csv(out_path, recs)

            stats["ok"] += 1
            if verbose:
                print(f"[ok] {p} -> {out_path} ({Model.__name__}, {len(recs)} rows)")
        except Exception as e:
            stats["fail"] += 1
            print(f"[fail] {p}: {e}", file=sys.stderr)

    if verbose:
        print("[summary]", stats)
    return stats

SchemaVersion = str
# -----------------------------
# 基础枚举 / 辅助
FactScalar = Union[Decimal, int, float, str, bool, None]
# -----------------------------
class CalcEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")   # 先宽松
    parent_concept: str
    child_concept: str
    weight: Optional[float] = 1.0
    source_path: Optional[str] = None

# 定义图（来自 *_def.xml）
class DefArc(BaseModel):
    model_config = ConfigDict(extra="ignore")
    from_concept: str
    to_concept: str
    arcrole: Optional[str] = None
    preferred_label: Optional[str] = None
    source_path: Optional[str] = None

# 标签映射（来自 *_lab.xml）
class LabelItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    concept: str
    label: Optional[str] = None
    role: Optional[str] = None
    lang: Optional[str] = None
    source_path: Optional[str] = None
# --------------
class FilingForm(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    OTHER = "OTHER"

class StatementHint(str, Enum):
    INCOME = "income"
    BALANCE = "balance"
    CASHFLOW = "cashflow"
    NOTES = "notes"
    OTHER = "other"
# 通用基类（元信息）
# -----------------------------
class RecordBase(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 继续严格
    schema_version: SchemaVersion = "0.3.0"

    # 源头追溯
    source_path: str

    # —— 增强：别名与规范化 —— 
    ticker: Optional[str] = Field(default=None, validation_alias=AliasChoices('ticker', 'symbol'))
    form: Optional[FilingForm] = Field(default=None, validation_alias=AliasChoices('form', 'filing_form'))
    year: Optional[int] = Field(default=None, validation_alias=AliasChoices('year', 'fiscal_year', 'report_year'))
    accno: Optional[str] = Field(default=None, validation_alias=AliasChoices('accno', 'accession', 'acc_no'))

    # 文档定位
    page_no: Optional[int] = None
    page_anchor: Optional[str] = None
    xpath: Optional[str] = None

    # 统计 / 语言
    language: str = Field(default="en") 
    tokens: Optional[int] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("language")
    @classmethod
    def _norm_lang(cls, v: Optional[str]) -> Optional[str]:
        return v.lower() if isinstance(v, str) else v

    @field_validator("ticker")
    @classmethod
    def _norm_ticker(cls, v: Optional[str]) -> Optional[str]:
        return v.upper().strip() if isinstance(v, str) else v

    @field_validator("year")
    @classmethod
    def _coerce_year(cls, v):
        # 允许 "2023" → 2023
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v

    @field_validator("form")
    @classmethod
    def _coerce_form(cls, v):
        # 允许 "10-k"/"10-K" 等 → FilingForm
        if isinstance(v, str):
            s = v.strip().upper()
            try:
                return FilingForm(s)
            except Exception:
                return FilingForm.OTHER
        return v

# -----------------------------
# 文本块（text.jsonl）
# -----------------------------
class TextChunk(RecordBase):
    id: UUID = Field(default_factory=uuid4)
    section: Optional[str] = None
    heading: Optional[str] = None
    statement_hint: Optional[StatementHint] = None
    text: str

# 宽松入口（忽略多余字段）
class TextChunkInput(TextChunk):
    model_config = ConfigDict(extra="ignore")  # 入口宽松，忽略多余键

    @model_validator(mode="before")
    def _coalesce_and_alias(cls, data: dict) -> dict:
        # 1) NaN → None；空字符串/纯空白 → None
        for k, v in list(data.items()):
            if isinstance(v, float) and isnan(v):
                data[k] = None
            elif isinstance(v, str) and (v.strip() == ""):
                data[k] = None

        # 2) 常见别名补位（若用户给了 accession，则映射到 accno）
        if data.get("accno") is None and data.get("accession"):
            data["accno"] = data.get("accession")

        # 3) 语义小修：支持 statement_hint 字符串自动归类
        sh = data.get("statement_hint")
        if isinstance(sh, str):
            s = sh.strip().lower()
            mapping = {
                "income": StatementHint.INCOME,
                "balance": StatementHint.BALANCE,
                "cashflow": StatementHint.CASHFLOW,
                "cash flow": StatementHint.CASHFLOW,
                "notes": StatementHint.NOTES,
                "other": StatementHint.OTHER,
            }
            data["statement_hint"] = mapping.get(s, data.get("statement_hint"))

        return data

# -----------------------------
# XBRL 上下文（Fact 的期间/实体/维度）
# -----------------------------
FactScalar = Union[Decimal, int, float, str, bool, None]

class XbrlPeriod(BaseModel):
    model_config = ConfigDict(extra="ignore")
    start_date: Optional[str] = Field(default=None, validation_alias=AliasChoices('start_date','startDate','period_start','start'))
    end_date:   Optional[str] = Field(default=None, validation_alias=AliasChoices('end_date','endDate','period_end','end'))
    instant:    Optional[str] = Field(default=None, validation_alias=AliasChoices('instant','period_instant'))

class XbrlContext(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entity: Optional[str] = None
    period: Optional[XbrlPeriod] = None
    dimensions: Dict[str, str] = {}

class FactInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_path: Optional[str] = None
    ticker: Optional[str] = None
    form:   Optional[str] = None
    year:   Optional[int]  = None
    accno:  Optional[str]  = None
    doc_date: Optional[str] = None
    fy: Optional[int] = None
    fq: Optional[str] = None

    qname: str = Field(validation_alias=AliasChoices('qname','concept','name'))
    value: FactScalar = None

    value_num_clean: Optional[float] = None
    value_raw_clean: Optional[str] = None
    value_display:   Optional[str] = None

    unit: Optional[str] = Field(default=None, validation_alias=AliasChoices('unit','uom'))
    decimals: Optional[int] = None
    context_id: Optional[str] = Field(default=None, validation_alias=AliasChoices('context_id','contextRef'))
    context: Optional[XbrlContext] = None

    label_text: Optional[str] = None
    period_label: Optional[str] = None
    rag_text: Optional[str] = None

    @model_validator(mode="before")
    def coalesce_fields(cls, data: dict) -> dict:
        # —— 新增：把所有 NaN 统一成 None（避免落到 str/int 等字段上报错）——
        for k, v in list(data.items()):
            if isinstance(v, float) and isnan(v):
                data[k] = None

        # year -> int（若是 "2023"）
        y = data.get("year")
        if isinstance(y, str) and y.isdigit():
            data["year"] = int(y)

        # 组装 context.period（由 period_start/period_end/instant 来）
        if data.get("context") is None and any(k in data for k in ("period_start","period_end","instant")):
            data["context"] = {
                "period": {
                    "period_start": data.get("period_start"),
                    "period_end":   data.get("period_end"),
                    "instant":      data.get("instant"),
                }
            }

        # 统一 value（你原来已有的逻辑保留）
        v_num = data.get("value_num_clean")
        if isinstance(v_num, float) and isnan(v_num):
            v_num = None
        if v_num is not None:
            data["value"] = v_num
        else:
            v_raw = data.get("value_raw_clean")
            if isinstance(v_raw, str):
                s = v_raw.strip().lower()
                if s in ("true","false"):
                    data["value"] = (s == "true")
                else:
                    from decimal import Decimal, InvalidOperation
                    try:
                        data["value"] = Decimal(v_raw.replace(",", ""))
                    except (InvalidOperation, AttributeError):
                        data["value"] = data.get("value_display")
        return data
# -----------------------------
# 数值事实（fact.jsonl）
# -----------------------------
class Fact(RecordBase):
    id: UUID = Field(default_factory=uuid4)
    qname: str                                   # us-gaap:Revenues
    value: Decimal                               # 用 Decimal 避免精度损失
    unit: Optional[str] = None                   # USD, shares, pure ...
    decimals: Optional[int] = None               # XBRL decimals
    context_id: Optional[str] = None
    context: Optional[XbrlContext] = None

    # 可选：把事实映射到财报表
    statement_hint: Optional[StatementHint] = None

    @field_validator("qname")
    @classmethod
    def _must_have_colon(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError("qname must include namespace, e.g., 'us-gaap:Revenues'")
        return v


# -----------------------------
# 向量索引条目（用于 RAG）
# -----------------------------
class IndexItem(RecordBase):
    id: UUID = Field(default_factory=uuid4)
    text: str
    embedding: Optional[List[float]] = None      # 也可另存 .npy
    meta: Dict[str, str] = Field(default_factory=dict)

class IndexItemInput(IndexItem):
    model_config = ConfigDict(extra="ignore")

def pick_model_for_file(p: Path) -> Type[BaseModel] | None:
    name = p.name.lower()
    if name.endswith("fact.jsonl"):
        return FactInput
    if name.endswith("text.jsonl") or "text_corpus" in name:
        return TextChunkInput
    if "calculation_edges" in name:
        return CalcEdge
    if "definition_arcs" in name:
        return DefArc
    if "labels" in name:
        return LabelItem
    return None

# -----------------------------
# 通用 IO
# -----------------------------
T = TypeVar("T", bound=BaseModel)

def dump_jsonl(path: Path | str, records: Iterable[T]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def load_jsonl_recursive_auto(root: Path | str) -> dict[str, int]:
    root = Path(root)
    stats: dict[str, int] = {}
    for p in root.rglob("*.jsonl"):
        Model = pick_model_for_file(p)
        if Model is None:
            print(f"[skip] {p} (no model router)")
            continue
        try:
            recs = load_jsonl(p, Model)  # 复用你的单文件加载
            stats.setdefault(Model.__name__, 0)
            stats[Model.__name__] += len(recs)
            print(f"[ok] {p} -> {len(recs)} ({Model.__name__})")
        except Exception as e:
            print(f"[fail] {p}: {e}")
    return stats

def load_jsonl(path: Path | str, model: type[T]) -> List[T]:
    path = Path(path)
    out: List[T] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out.append(model.model_validate(obj))
            except Exception as e:
                raise ValueError(f"{path}:{ln}: {e}") from e
    return out

def load_jsonl_recursive(root: Path | str, model: type[T]) -> List[T]:
    root = Path(root)
    acc: List[T] = []
    for p in root.rglob("*.jsonl"):
        acc.extend(load_jsonl(p, model))
    return acc

# -----------------------------
# 简单 smoke 测试（可删）
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch parse & export clean → silver (mirror paths)")
    parser.add_argument("--in-root",  type=str, default="data/clean",  help="Input root directory (clean)")
    parser.add_argument("--out-root", type=str, default="data/silver", help="Output root directory (silver)")
    parser.add_argument("--format",   type=str, default="jsonl", choices=["jsonl", "parquet", "csv"],
                        help="Export format")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--quiet", action="store_true", help="Less logs")
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    verbose = not args.quiet

    if not in_root.exists():
        print(f"[error] input root not found: {in_root}", file=sys.stderr)
        sys.exit(1)

    process_tree(
        in_root,
        out_root,
        export_format=args.format,
        overwrite=args.overwrite,
        verbose=verbose,
    )