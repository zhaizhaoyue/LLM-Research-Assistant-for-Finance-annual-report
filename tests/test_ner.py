"""Tests for the NER module."""
import pytest
from src.ner.fin_entities import extract_tickers, extract_fin_metrics, extract_custom_entities
from src.ner.normalizer import normalize_entity, deduplicate_entities, normalize_org


class TestFinEntities:
    """Test custom finance entity extractors."""

    def test_extract_ticker_in_parens(self):
        text = "Apple Inc. (AAPL) reported strong earnings."
        results = extract_tickers(text)
        tickers = [r["text"] for r in results]
        assert "AAPL" in tickers

    def test_extract_fin_metric(self):
        text = "The company's EBITDA margin was 25% and EPS grew by 10%."
        results = extract_fin_metrics(text)
        labels = [r["text"] for r in results]
        assert "EBITDA" in labels
        assert "EPS" in labels

    def test_ticker_blocklist(self):
        text = "THE company AND its subsidiaries"
        results = extract_tickers(text)
        tickers = [r["text"] for r in results]
        assert "THE" not in tickers
        assert "AND" not in tickers

    def test_custom_entities_combined(self):
        text = "Apple (AAPL) reported EBITDA of $5B."
        results = extract_custom_entities(text)
        labels = {r["label"] for r in results}
        assert "TICKER" in labels
        assert "FIN_METRIC" in labels

    def test_extract_ticker_with_keyword(self):
        text = "Ticker: MSFT is trading at $400."
        results = extract_tickers(text)
        tickers = [r["text"] for r in results]
        assert "MSFT" in tickers

    def test_no_false_positive_tickers(self):
        text = "The CEO of the LLC reported to SEC."
        results = extract_tickers(text)
        tickers = [r["text"] for r in results]
        assert "CEO" not in tickers
        assert "LLC" not in tickers
        assert "SEC" not in tickers

    def test_multiple_fin_metrics(self):
        text = "ROE improved while CAGR of revenue reached 15%. P/E ratio is 25."
        results = extract_fin_metrics(text)
        terms = [r["text"] for r in results]
        assert "ROE" in terms
        assert "CAGR" in terms
        assert "P/E" in terms


class TestNormalizer:
    """Test entity normalization."""

    def test_normalize_org_suffix_removal(self):
        assert normalize_org("Apple Inc.") == "APPLE"
        assert normalize_org("Microsoft Corporation") == "MICROSOFT"

    def test_normalize_org_alias(self):
        assert normalize_org("Google") == "ALPHABET INC"
        assert normalize_org("Facebook") == "META PLATFORMS INC"

    def test_normalize_entity_org(self):
        ent = {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10, "confidence": 0.99}
        result = normalize_entity(ent)
        assert result["normalized"] == "APPLE"

    def test_normalize_entity_person(self):
        ent = {"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8, "confidence": 0.95}
        result = normalize_entity(ent)
        assert result["normalized"] == "TIM COOK"

    def test_normalize_entity_ticker(self):
        ent = {"text": "AAPL", "label": "TICKER", "start": 0, "end": 4, "confidence": 0.80}
        result = normalize_entity(ent)
        assert result["normalized"] == "AAPL"

    def test_deduplicate(self):
        entities = [
            {"text": "Apple", "label": "ORG", "normalized": "APPLE",
             "confidence": 0.9, "start": 0, "end": 5},
            {"text": "Apple Inc.", "label": "ORG", "normalized": "APPLE",
             "confidence": 0.95, "start": 10, "end": 20},
        ]
        deduped = deduplicate_entities(entities)
        assert len(deduped) == 1
        assert deduped[0]["confidence"] == 0.95

    def test_deduplicate_different_labels(self):
        entities = [
            {"text": "Apple", "label": "ORG", "normalized": "APPLE",
             "confidence": 0.9, "start": 0, "end": 5},
            {"text": "Apple", "label": "GPE", "normalized": "APPLE",
             "confidence": 0.7, "start": 20, "end": 25},
        ]
        deduped = deduplicate_entities(entities)
        assert len(deduped) == 2  # Different labels => no dedup


class TestNERRecognizer:
    """Integration tests for the NER model (requires model download)."""

    @pytest.fixture(scope="class")
    def recognizer(self):
        from src.ner.recognizer import NERRecognizer
        return NERRecognizer(
            model_name="dslim/bert-base-NER",
            device=-1,
            batch_size=8,
        )

    def test_recognize_org(self, recognizer):
        text = "Apple Inc. reported revenue of $394 billion."
        entities = recognizer.recognize(text)
        org_entities = [e for e in entities if e["label"] == "ORG"]
        assert len(org_entities) >= 1
        assert any("Apple" in e["text"] for e in org_entities)

    def test_recognize_empty(self, recognizer):
        assert recognizer.recognize("") == []
        assert recognizer.recognize("   ") == []

    def test_recognize_batch(self, recognizer):
        texts = [
            "Goldman Sachs upgraded Apple to Buy.",
            "Warren Buffett sold his position in Tesla.",
        ]
        results = recognizer.recognize_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
