# src/qa/testing/run_answer_api_numeric_demo.py
import os, sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pandas as pd
from src.qa import answer_api

def inject_fake_facts():
    df = pd.DataFrame([
        {
            "ticker": "AAPL", "form": "10-K", "accno": "0000320193-23-000106",
            "concept": "us-gaap:SalesRevenueNet",
            "fy_norm": 2022, "fq_norm": None, "period_type": "duration",
            "period_start": "2021-09-26", "period_end": "2022-09-24", "instant": None,
            "value_std": 394_328_000_000.0, "unit_std": "money",
            "source_path": r"data\raw_reports\standard\US_AAPL_2022_10-K_0000320193-22-000108.html",
            "page_no": 26,
        },
        {
            "ticker": "AAPL", "form": "10-K", "accno": "0000320193-23-000106",
            "concept": "us-gaap:SalesRevenueNet",
            "fy_norm": 2023, "fq_norm": None, "period_type": "duration",
            "period_start": "2022-09-25", "period_end": "2023-09-30", "instant": None,
            "value_std": 383_285_000_000.0, "unit_std": "money",
            "source_path": r"data\raw_reports\standard\US_AAPL_2023_10-K_0000320193-23-000106.html",
            "page_no": 26,
        },
    ])
    answer_api._FACTS_NUMERIC_DF = df  # 直接注入，绕过读 parquet

def fake_hits():
    return [
        {
            "snippet": "Total net sales decreased 3% or $11.0 billion during 2023 compared to 2022.",
            "chunk_id": "cid-1",
            "meta": {
                "source_path": r"data\raw_reports\standard\US_AAPL_2023_10-K_0000320193-23-000106.html",
                "accno": "0000320193-23-000106",
                "ticker": "AAPL",
                "form": "10-K",
                "fy": 2023,
                "page_no": 26,
                "concept": "us-gaap:SalesRevenueNet",
                "file_type": "text_chunk",
            }
        }
    ]

# monkeypatch retriever.hybrid_search -> 返回我们手造的 hits
class _FakeRetriever:
    @staticmethod
    def hybrid_search(query: str, filters: dict, topk: int = 8):
        return fake_hits()

# 覆盖真正的 retriever
import types
answer_api.hybrid_search = types.FunctionType(_FakeRetriever.hybrid_search.__code__, globals(), "hybrid_search")

if __name__ == "__main__":
    inject_fake_facts()

    q = "What is the year-over-year revenue in 2023?"
    filters = {"ticker": "AAPL", "form": "10-K", "year": 2023}
    res = answer_api.answer_question(q, filters)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    assert "YoY" in res["final_answer"]
    assert res["citations"], "should have citations"
    assert res["citations_display"], "should have display strings"
    print("\n[OK] answer_api numeric demo finished.")
