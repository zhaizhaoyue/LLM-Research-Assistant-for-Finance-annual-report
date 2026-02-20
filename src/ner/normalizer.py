"""Entity normalization: canonical forms, deduplication, alias resolution."""
from __future__ import annotations
import re

# ---- ORG normalization rules ----

ORG_SUFFIXES = re.compile(
    r"\s*,?\s*\b(Corporation|Inc\.?|Corp\.?|Limited|Ltd\.?|Company|Holdings?|LLC|LLP|PLC|Co\.?|Group)\s*$",
    re.IGNORECASE,
)

# Common aliases -> canonical name
ORG_ALIASES: dict[str, str] = {
    "ALPHABET": "ALPHABET INC",
    "GOOGLE": "ALPHABET INC",
    "META": "META PLATFORMS INC",
    "FACEBOOK": "META PLATFORMS INC",
    "AMAZON": "AMAZON.COM INC",
    "MICROSOFT": "MICROSOFT",
}


def normalize_org(text: str) -> str:
    """Normalize an ORG entity to its canonical form."""
    normed = text.strip().upper()
    normed = ORG_SUFFIXES.sub("", normed).strip()
    return ORG_ALIASES.get(normed, normed)


def normalize_entity(entity: dict) -> dict:
    """Add a 'normalized' key based on entity label."""
    label = entity["label"]
    text = entity["text"]

    if label == "ORG":
        entity["normalized"] = normalize_org(text)
    elif label == "PERSON":
        entity["normalized"] = text.strip().upper()
    elif label == "GPE":
        entity["normalized"] = text.strip().upper()
    elif label == "TICKER":
        entity["normalized"] = text.strip().upper()
    elif label == "FIN_METRIC":
        entity["normalized"] = text.strip().upper()
    elif label in ("MONEY", "PERCENT", "DATE"):
        entity["normalized"] = text.strip()
    else:
        entity["normalized"] = text.strip()

    return entity


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Remove duplicate entities (same normalized text + same label).

    When duplicates exist, keep the one with highest confidence.
    """
    best: dict[tuple[str, str], dict] = {}
    for ent in entities:
        key = (ent.get("normalized", ent["text"]), ent["label"])
        if key not in best or ent["confidence"] > best[key]["confidence"]:
            best[key] = ent
    return list(best.values())
