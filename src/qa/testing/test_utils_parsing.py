# src/qa/testing/run_utils_parsing_demo.py
from __future__ import annotations
import os, sys

# 让脚本可直接运行（无需设置 PYTHONPATH）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.qa.utils_parsing import (
    parse_numeric_targets,
    pick_two_period_values,
    compute_change,
    format_number_with_unit,
)

def fake_hit(
    snippet: str,
    *,
    concept: str = "us-gaap:SalesRevenueNet",
    label_search_tokens: str = "revenue net sales",
    value=None,
    fy=None,
    page_no=12,
    source_path="data/raw_reports/standard/US_AAPL_2023_10-K_0000320193-23-000106.html",
    unit="USD",
    currency="USD",
    file_type="fact",
    chunk_id="cidX",
    ticker="AAPL",
    form="10-K",
):
    """构造一条模拟的检索命中 hit。"""
    return {
        "snippet": snippet,
        "chunk_id": chunk_id,
        "meta": {
            "source_path": source_path,
            "page_no": page_no,
            "file_type": file_type,
            "ticker": ticker,
            "form": form,
            "fy": fy,  # 用于当前期/上期选择
            "concept": concept,
            "label_search_tokens": label_search_tokens,
            "value": value,        # 若为 None，会回退从 snippet 里抓一个数字
            "unit": unit,
            "currency": currency,
            "accno": "0000320193-23-000106",
        },
    }

def demo_yoy_revenue_2023():
    print("\n=== DEMO 1: YoY revenue 2023 ===")
    query = "What is the year-over-year revenue in 2023?"
    target, change_type = parse_numeric_targets(query)
    print("parse_numeric_targets ->", target, change_type)

    # 构造两期：FY2022 / FY2023 的“净销售额”
    hits = [
        fake_hit(
            "Total net sales: 2022 = 394,328 million USD.",
            value=394_328_000_000.0,  # 标准美元（假设 scale 已归一）
            fy=2022,
            page_no=26,
            chunk_id="cid_2022",
            file_type="table",
        ),
        fake_hit(
            "Total net sales: 2023 = 383,285 million USD.",
            value=383_285_000_000.0,
            fy=2023,
            page_no=26,
            chunk_id="cid_2023",
            file_type="table",
        ),
        # 噪音命中，不应影响结果
        fake_hit("R&D expense increased 14% 13% 12%",
         concept="us-gaap:ResearchAndDevelopmentExpense",
         label_search_tokens="research development expense r&d",   # 不要 'revenue'
         file_type="text",                                         # 让它排在后面
         fy=2023, page_no=45, chunk_id="cid_noise")
    ]

    filters = {"ticker": "AAPL", "form": "10-K", "year": 2023}
    v_cur, v_prev, unit, currency, cits = pick_two_period_values(hits, target, filters)
    print("picked values ->", v_cur, v_prev, unit, currency)
    print("citations ->", cits)

    yoy = compute_change(v_cur, v_prev, change_type)
    print("compute_change (YoY) ->", yoy)

    ans = format_number_with_unit(yoy, unit=unit, currency=currency, change_type=change_type)
    print("final formatted ->", ans)  # 期望 -2.80% 左右（这里只是示例）

def demo_raw_cash_latest():
    print("\n=== DEMO 2: latest cash (raw) ===")
    query = "现金余额是多少？"
    target, change_type = parse_numeric_targets(query)
    print("parse_numeric_targets ->", target, change_type)

    hits = [
        fake_hit("Cash and cash equivalents were $29,965 million as of September 30, 2023.",
                 concept="us-gaap:CashAndCashEquivalentsAtCarryingValue",
                 label_search_tokens="cash cash equivalents", value=29_965_000_000.0, fy=2023, page_no=34),
        fake_hit("As of 2022, cash was $23,646 million.",
                 concept="us-gaap:CashAndCashEquivalentsAtCarryingValue",
                 label_search_tokens="cash cash equivalents", value=23_646_000_000.0, fy=2022, page_no=34),
    ]
    filters = {"ticker": "AAPL"}  # 没给 year，则 pick_two_period_values 会把遇到的第一条当“当前期”
    v_cur, v_prev, unit, currency, cits = pick_two_period_values(hits, target, filters)
    print("picked values ->", v_cur, unit, currency)
    raw = compute_change(v_cur, v_prev, change_type)  # raw -> 直接返回 v_cur
    print("compute_change (raw) ->", raw)
    ans = format_number_with_unit(raw, unit=unit, currency=currency, change_type=change_type)
    print("final formatted ->", ans)  # 期望 $29.97B

def demo_diff_profit_2023():
    print("\n=== DEMO 3: diff net income 2023 vs 2022 ===")
    query = "苹果公司 2023 净利润较 2022 变化多少（差额）？"
    target, change_type = parse_numeric_targets(query + " 差额")  # 带上“差额”提示
    print("parse_numeric_targets ->", target, change_type)

    hits = [
        fake_hit("Net income for 2022 was $99,803 million.",
                 concept="us-gaap:NetIncomeLoss", label_search_tokens="net income profit", value=99_803_000_000.0, fy=2022, page_no=27),
        fake_hit("Net income for 2023 was $97,000 million.",
                 concept="us-gaap:NetIncomeLoss", label_search_tokens="net income profit", value=97_000_000_000.0, fy=2023, page_no=27),
    ]
    filters = {"ticker": "AAPL", "year": 2023}
    v_cur, v_prev, unit, currency, cits = pick_two_period_values(hits, target, filters)
    print("picked values ->", v_cur, v_prev, unit, currency)
    diff = compute_change(v_cur, v_prev, change_type)
    print("compute_change (diff) ->", diff)
    ans = format_number_with_unit(diff, unit=unit, currency=currency, change_type=change_type)
    print("final formatted ->", ans)  # 期望 -$2.80B 左右（示例）

if __name__ == "__main__":
    demo_yoy_revenue_2023()
    demo_raw_cash_latest()
    demo_diff_profit_2023()
    print("\n[OK] utils_parsing demos finished.")
