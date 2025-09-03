# tests/test_hybrid_qa.py
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.qa.hybrid_qa import (
    answer_textual_or_mixed,
    _clean_snippet,
    _rule_based_answer,
    _build_llm_prompt,
)


@pytest.fixture
def sample_hits():
    return [
        {
            "chunk_id": "cid_1",
            "snippet": "The company's revenue increased by 10% year over year in 2023.",
            "meta": {
                "source_path": "data/raw_reports/standard/US_AAPL_2023_10-K.html",
                "accno": "0000320193-23-000106",
                "ticker": "AAPL",
                "form": "10-K",
                "fy": 2023,
                "fq": "FY",
                "page_no": 26,
                "section": "Item 7",
            },
        },
        {
            "chunk_id": "cid_2",
            "snippet": "In 2022, the revenue was slightly higher compared to 2023.",
            "meta": {
                "source_path": "data/raw_reports/standard/US_AAPL_2022_10-K.html",
                "accno": "0000320193-22-000108",
                "ticker": "AAPL",
                "form": "10-K",
                "fy": 2022,
                "fq": "FY",
                "page_no": 24,
                "section": "Item 7",
            },
        },
    ]


def test_rule_based_answer(sample_hits):
    blocks = [sample_hits[0]["snippet"], sample_hits[1]["snippet"]]
    ans = _rule_based_answer("What is the revenue trend?", blocks)
    assert "10%" in ans or "revenue" in ans


def test_clean_snippet_removes_noise():
    dirty = "  Revenue   \t increased ││ by 5% ●● due to growth \n\n in demand "
    clean = _clean_snippet(dirty)
    assert "│" not in clean and "●" not in clean
    assert "Revenue" in clean and "5%" in clean


def test_llm_prompt_contains_context(sample_hits):
    blocks = [h["snippet"] for h in sample_hits]
    prompt = _build_llm_prompt("What happened?", blocks)
    assert "Context:" in prompt
    assert "What happened?" in prompt
    assert "SEC filings" in prompt


def test_answer_rule_based_only(sample_hits):
    # 不调用 LLM
    ans, reasoning, cites = answer_textual_or_mixed(
        "Explain revenue change", sample_hits, filters={}, use_llm=False
    )
    assert isinstance(ans, str)
    assert "规则拼接" in reasoning
    assert cites and isinstance(cites, list)


def test_answer_with_llm(monkeypatch, sample_hits):
    # 模拟 llm_summarize 返回值
    monkeypatch.setattr(
        "src.qa.hybrid_qa.llm_summarize", lambda prompt, **kw: "This is a mock summary."
    )
    ans, reasoning, cites = answer_textual_or_mixed(
        "Explain revenue change", sample_hits, filters={}, use_llm=True
    )
    assert "mock summary" in ans.lower()
    assert "LLM" in reasoning
    assert cites and all("ticker" in c for c in cites)


def test_answer_with_empty_hits():
    ans, reasoning, cites = answer_textual_or_mixed(
        "What is revenue?", [], filters={}, use_llm=False
    )
    assert "无法给出充分的信息" in ans
    assert cites == []
