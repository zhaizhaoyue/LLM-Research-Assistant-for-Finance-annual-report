# numeric_cache 是 数值事实查询引擎，提供干净、快速、结构化的财报数值访问接口，避免 retriever/LLM 去“猜”数字。
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict, Any

FQ_MAP = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}

class FactsNumericCache:
    def __init__(self, path: str | Path = "data/index/facts_numeric.parquet"):
        self.path = Path(path)
        if not self.path.exists():
            self.df = None
            return
        df = pd.read_parquet(self.path)
        # 规范列名 → numeric 层习惯
        rename = {
            "fy_norm": "fy",
            "fq_norm": "fq_num",
            "value_std": "value",
            "unit_std": "unit",
        }
        for k, v in rename.items():
            if k in df.columns:
                df = df.rename(columns={k: v})
        # 映射 fq
        if "fq" not in df.columns and "fq_num" in df.columns:
            df["fq"] = df["fq_num"].map(FQ_MAP)
        # 统一 form 大写
        if "form" in df.columns:
            df["form"] = df["form"].astype(str).str.upper()
        self.df = df

    def ok(self) -> bool:
        return self.df is not None and len(self.df) > 0

    def _annual_like_mask(self, df: pd.DataFrame, require_annual_like: bool) -> pd.Series:
        if not require_annual_like:
            return pd.Series([True] * len(df), index=df.index)
        # “年度口径”规则：FY 或 Q4（满足你要的 FY+4+Q4）
        has_fq = "fq" in df.columns
        if has_fq:
            return df["fq"].astype(str).str.upper().isin(["FY", "Q4", "FQ4", "4", "Y"])
        return pd.Series([True] * len(df), index=df.index)  # 没有 fq 时放宽

    def query_two_periods(
        self,
        ticker: str,
        fy: int,
        form: str,
        concepts: Iterable[str] | None = None,
        concepts_regex: Optional[str] = None,
        require_annual_like: bool = True,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        返回两个 dict：cur(FY), prev(FY-1)，各含 value/unit/currency/fy/fq/form/accno/source_path/period_*…
        """
        if not self.ok():
            return None
        df = self.df
        q = (df["ticker"] == ticker) & (df["form"] == form.upper()) & (df["fy"].isin([fy, fy - 1]))
        df2 = df[q].copy()
        if df2.empty:
            return None

        # 概念白名单 / 正则
        if concepts:
            df2 = df2[df2["concept"].isin(concepts)]
        if concepts_regex:
            df2 = df2[df2["concept"].str.contains(concepts_regex, case=False, regex=True, na=False)]

        # 年度口径
        mask_ann = self._annual_like_mask(df2, require_annual_like)
        df2 = df2[mask_ann]
        if df2.empty:
            return None

        cur = df2[df2["fy"] == fy]
        prev = df2[df2["fy"] == fy - 1]
        if cur.empty or prev.empty:
            return None

        # 选择策略：按 value 最大的一条（避免维度行/子构成）
        c = cur.sort_values("value", ascending=False).iloc[0].to_dict()
        p = prev.sort_values("value", ascending=False).iloc[0].to_dict()
        # 填充缺失键
        for k in ("currency",):
            c.setdefault(k, None); p.setdefault(k, None)
        # 归一 keys
        for r in (c, p):
            r["fq"] = str(r.get("fq") or "")
        return c, p
