"""Entity timeline: track entity mentions across years and filings."""
from __future__ import annotations
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MAX_SAMPLE_CONTEXTS = 3  # Max context samples per year per entity


def build_timelines(ner_dir: str | Path) -> dict[str, dict]:
    """Build entity timelines from NER output.

    Returns dict keyed by "LABEL::NORMALIZED" with structure:
        {
            "normalized": "APPLE",
            "label": "ORG",
            "total_mentions": 42,
            "entries": [
                {"year": 2022, "form": "10-K", "ticker": "AAPL",
                 "mention_count": 15, "sample_contexts": ["...", ...]},
                ...
            ]
        }
    """
    ner_dir = Path(ner_dir)
    files = sorted(ner_dir.rglob("text_corpus.jsonl"))

    if not files:
        log.warning("No text_corpus.jsonl files found under %s", ner_dir)
        return {}

    # Accumulate: (label, normalized) -> year -> {count, ticker, form, contexts}
    acc: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {
            "count": 0, "tickers": set(), "forms": set(), "contexts": []
        })
    )

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entities = row.get("entities", [])
                year = row.get("year")
                ticker = row.get("ticker", "")
                form = row.get("form", "")
                text = row.get("text", "")

                if not year:
                    continue

                year = int(year)

                for ent in entities:
                    normalized = ent.get("normalized", ent.get("text", ""))
                    label = ent.get("label", "MISC")
                    key = (label, normalized)
                    entry = acc[key][year]
                    entry["count"] += 1
                    if ticker:
                        entry["tickers"].add(ticker)
                    if form:
                        entry["forms"].add(form)
                    if text and len(entry["contexts"]) < MAX_SAMPLE_CONTEXTS:
                        # Store a short snippet around the entity
                        snippet = text[:200].strip()
                        if snippet not in entry["contexts"]:
                            entry["contexts"].append(snippet)

    # Build output
    timelines: dict[str, dict] = {}
    for (label, normalized), year_data in acc.items():
        timeline_id = f"{label}::{normalized}"
        entries = []
        total = 0
        for year in sorted(year_data.keys()):
            data = year_data[year]
            total += data["count"]
            entries.append({
                "year": year,
                "tickers": sorted(data["tickers"]),
                "forms": sorted(data["forms"]),
                "mention_count": data["count"],
                "sample_contexts": data["contexts"],
            })
        timelines[timeline_id] = {
            "normalized": normalized,
            "label": label,
            "total_mentions": total,
            "entries": entries,
        }

    log.info("Built timelines for %d entities from %d files",
             len(timelines), len(files))
    return timelines


def save_timelines(timelines: dict[str, dict], output_path: str | Path) -> None:
    """Save timelines to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(timelines, f, ensure_ascii=False, indent=2)
    log.info("Saved %d timelines to %s", len(timelines), output_path)


def load_timelines(path: str | Path) -> dict[str, dict]:
    """Load timelines from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def query_timeline(
    timelines: dict[str, dict],
    entity_name: str,
    label: str = "ORG",
) -> dict | None:
    """Query timeline for a specific entity."""
    key = f"{label}::{entity_name.strip().upper()}"
    return timelines.get(key)
