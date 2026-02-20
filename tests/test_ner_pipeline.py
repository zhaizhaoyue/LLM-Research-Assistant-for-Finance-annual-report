"""Integration test: NER stage end-to-end."""
import json
import tempfile
from pathlib import Path
from src.ner.process import process_file


def test_process_file_e2e():
    """Test NER processing of a single JSONL file."""
    rows = [
        {"idx_source": 0, "text": "Apple Inc. (AAPL) reported EBITDA of $5B.",
         "ticker": "AAPL", "year": 2024, "form": "10-K"},
        {"idx_source": 1, "text": "Tim Cook stated that revenue grew 8% YoY.",
         "ticker": "AAPL", "year": 2024, "form": "10-K"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input" / "text_corpus.jsonl"
        out_path = Path(tmpdir) / "output" / "text_corpus.jsonl"
        in_path.parent.mkdir(parents=True)

        with open(in_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        result = process_file(in_path, out_path)
        assert result["total"] == 2
        assert out_path.exists()

        with open(out_path) as f:
            output_rows = [json.loads(line) for line in f]

        assert len(output_rows) == 2
        # Every row should have entities and entity_labels fields
        for row in output_rows:
            assert "entities" in row
            assert "entity_labels" in row
            assert isinstance(row["entities"], list)

        # First row should have TICKER and FIN_METRIC from custom extractors
        first_labels = {e["label"] for e in output_rows[0]["entities"]}
        assert "TICKER" in first_labels
        assert "FIN_METRIC" in first_labels
