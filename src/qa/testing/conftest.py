# tests/conftest.py
import pytest

@pytest.fixture
def fake_hits_revenue_yoy():
    # 命中列表按 score 排序（降序），meta 字段尽量齐全
    return [
        {
            "chunk_id": "AAPL::fact::rev::2023",
            "score": 0.92,
            "snippet": "Net sales for fiscal 2023 were $383,285 million.",
            "meta": {
                "source_path": "data/raw_reports/standard/US_AAPL_2023_10-K.html",
                "accno": "0000320193-23-000106",
                "ticker": "AAPL",
                "form": "10-K",
                "fy": 2023,
                "fq": "Q4",
                "page_no": 123,
                "section": "Item 7",
                "file_type": "fact",
                "concept": "us-gaap:Revenues",
                "label_search_tokens": "net sales total net sales revenue",
                "value": 383285000000.0,
                "unit": "USD",
                "currency": "USD",
                "period_end": "2023-09-30",
            },
        },
        {
            "chunk_id": "AAPL::fact::rev::2022",
            "score": 0.88,
            "snippet": "Net sales for fiscal 2022 were $394,328 million.",
            "meta": {
                "source_path": "data/raw_reports/standard/US_AAPL_2022_10-K.html",
                "accno": "0000320193-22-000010",
                "ticker": "AAPL",
                "form": "10-K",
                "fy": 2022,
                "fq": "Q4",
                "page_no": 121,
                "section": "Item 7",
                "file_type": "fact",
                "concept": "us-gaap:Revenues",
                "label_search_tokens": "net sales total net sales revenue",
                "value": 394328000000.0,
                "unit": "USD",
                "currency": "USD",
                "period_end": "2022-09-24",
            },
        },
    ]

@pytest.fixture
def fake_hits_textual():
    return [
        {
            "chunk_id": "AAPL::text::drivers",
            "score": 0.90,
            "snippet": "The increase in Services revenue was driven by growth in cloud and payments.",
            "meta": {
                "source_path": "data/raw_reports/standard/US_AAPL_2023_10-K.html",
                "accno": "0000320193-23-000106",
                "ticker": "AAPL",
                "form": "10-K",
                "fy": 2023,
                "fq": "Q4",
                "page_no": 150,
                "section": "Item 7",
                "file_type": "text",
            },
        },
        {
            "chunk_id": "AAPL::text::drivers2",
            "score": 0.79,
            "snippet": "Wearables also contributed to revenue growth.",
            "meta": {
                "source_path": "data/raw_reports/standard/US_AAPL_2023_10-K.html",
                "accno": "0000320193-23-000106",
                "ticker": "AAPL",
                "form": "10-K",
                "fy": 2023,
                "fq": "Q4",
                "page_no": 151,
                "section": "Item 7",
                "file_type": "text",
            },
        },
    ]

@pytest.fixture
def filters_2023():
    return {"ticker": "AAPL", "form": "10-K", "year": 2023}

# 伪造 llm_summarize（避免真实 API）
@pytest.fixture(autouse=True)
def patch_llm(monkeypatch):
    def fake_llm(prompt: str, **kwargs):
        # 简短、稳定的可测试输出
        return "（LLM总结）根据提供的上下文，关键点已在引用段落中说明。要点 / Key Points:\n- services增长\n- wearables贡献"
    try:
        import src.qa.llm as llm
        monkeypatch.setattr(llm, "llm_summarize", fake_llm, raising=False)
    except Exception:
        pass
    yield

# 伪造 retrieval.hybrid_search（仅用于 answer_api 集成测）
@pytest.fixture
def patch_retrieval(monkeypatch, fake_hits_revenue_yoy):
    def fake_search(query: str, filters: dict, topk: int = 8):
        return fake_hits_revenue_yoy
    import src.retrieval as retrieval
    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    return True
