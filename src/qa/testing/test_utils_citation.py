import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.qa.hybrid_qa import answer_textual_or_mixed
from src.qa.utils_citation import citation_to_display

def _hit(snippet, page_no, chunk_id):
    return {
        "snippet": snippet,
        "chunk_id": chunk_id,
        "meta": {
            "source_path": r"data\raw_reports\standard\US_AAPL_2023_10-K_0000320193-23-000106.html",
            "accno": "0000320193-23-000106",
            "ticker": "AAPL",
            "form": "10-K",
            "fy": 2023,
            "fq": None,
            "section": "Item 7",
            "page_no": page_no,
            "concept": "us-gaap:SalesRevenueNet",
            "file_type": "text_chunk",
        }
    }

def test_textual_citations_and_display():
    hits = [
        _hit("Total net sales decreased 3% or $11.0 billion during 2023 compared to 2022.", 26, "cid-1"),
        _hit("The weakness in foreign currencies accounted for more than the entire decrease.", 27, "cid-2"),
    ]
    final, reason, cits = answer_textual_or_mixed(
        query="Why did revenue decrease in 2023?",
        hits=hits,
        filters={},
        use_llm=False,
        k_ctx=4,
        max_chars=300
    )

    # 基本断言：答案是规则拼接；引用两条
    assert "decreased 3%" in final
    assert reason.startswith("LLM") or "规则拼接" in reason
    assert len(cits) == 2

    # 字段完整性
    for c in cits:
        assert c["ticker"] == "AAPL"
        assert c["form"] == "10-K"
        assert c["fy"] == 2023
        assert c["page"] in (26, 27)
        assert c["concept"] == "us-gaap:SalesRevenueNet"

    # 展示短串
    disp = [citation_to_display(c) for c in cits]
    assert disp == ["AAPL 10-K FY2023 p.26 [Item 7]", "AAPL 10-K FY2023 p.27 [Item 7]"]
