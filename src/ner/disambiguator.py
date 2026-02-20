"""Entity disambiguation: resolve ambiguous entities using context and known company data."""
from __future__ import annotations
import csv
import re
from pathlib import Path
from typing import Optional

# Financial context keywords that suggest an entity is a company, not a common noun
FINANCIAL_CONTEXT_RE = re.compile(
    r"\b(revenue|earnings|profit|loss|sales|income|dividend|stock|share|"
    r"fiscal|quarter|annual|report|filing|SEC|10-K|10-Q|"
    r"market\s*cap|valuation|IPO|CEO|CFO|CTO|COO|"
    r"subsidiary|acquisition|merger|acquired|"
    r"EBITDA|EPS|ROE|ROA|CAGR|P/E)\b",
    re.IGNORECASE,
)

DEFAULT_COMPANIES_CSV = Path("data/companies.csv")


class Disambiguator:
    """Context-aware entity disambiguation using known company data."""

    def __init__(self, companies_csv: str | Path = DEFAULT_COMPANIES_CSV):
        # ticker -> company name
        self.ticker_to_name: dict[str, str] = {}
        # normalized name variants -> ticker
        self.name_to_ticker: dict[str, str] = {}

        self._load_companies(Path(companies_csv))

    def _load_companies(self, csv_path: Path) -> None:
        if not csv_path.exists():
            return
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = (row.get("ticker") or "").strip().upper()
                notes = (row.get("notes") or "").strip()
                aliases = self._parse_aliases(row.get("aliases"))
                if not ticker:
                    continue
                self.ticker_to_name[ticker] = notes

                # Build reverse index: multiple name forms -> ticker
                name_seeds = []
                if notes:
                    name_seeds.append(notes)
                name_seeds.extend(aliases)
                for name_seed in name_seeds:
                    for name_form in self._name_variants(ticker, name_seed):
                        self.name_to_ticker[name_form] = ticker

    @staticmethod
    def _parse_aliases(raw_aliases: str | None) -> list[str]:
        """Parse aliases column value into a list.

        Supported separators: "|" and ";"
        """
        if not raw_aliases:
            return []
        aliases = []
        for part in re.split(r"[|;]", raw_aliases):
            alias = part.strip()
            if alias:
                aliases.append(alias)
        return aliases

    @staticmethod
    def _name_variants(ticker: str, full_name: str) -> list[str]:
        """Generate normalized name variants for reverse lookup."""
        variants = []
        upper = full_name.strip().upper()
        if upper:
            variants.append(upper)
        # First word (e.g., "Apple" from "Apple Inc.")
        first_word = upper.split()[0] if upper else ""
        if first_word and len(first_word) >= 3:
            variants.append(first_word)
        # Ticker itself
        variants.append(ticker.upper())
        # Strip common suffixes
        for suffix in [" INC", " INC.", " CORP", " CORP.", " LTD", " LTD.",
                       " LLC", " PLC", " CO", " CO.", " GROUP", " PLATFORMS"]:
            if upper.endswith(suffix):
                stripped = upper[: -len(suffix)].strip()
                if stripped and len(stripped) >= 2:
                    variants.append(stripped)
        return list(set(variants))

    def resolve_ticker(self, entity_text: str) -> Optional[str]:
        """Try to resolve entity text to a known ticker symbol."""
        normed = entity_text.strip().upper()
        # Direct ticker match
        if normed in self.ticker_to_name:
            return normed
        # Name lookup
        return self.name_to_ticker.get(normed)

    def disambiguate(
        self,
        entity: dict,
        context_text: str = "",
        doc_ticker: str = "",
    ) -> dict:
        """Disambiguate an entity using context and known company data.

        Modifies the entity dict in-place and returns it:
        - Adds 'resolved_ticker' if the entity matches a known company
        - Boosts confidence if financial context is present
        - Adds 'is_company' flag for ORG entities
        """
        label = entity.get("label", "")
        text = entity.get("text", "")
        confidence = entity.get("confidence", 0.0)

        if label != "ORG":
            return entity

        # Try to resolve to a known company
        resolved = self.resolve_ticker(text)

        # If doc_ticker is known, the filing's own company gets priority
        if not resolved and doc_ticker:
            doc_name = self.ticker_to_name.get(doc_ticker.upper(), "")
            if doc_name and text.strip().upper() in self._name_variants(doc_ticker.upper(), doc_name):
                resolved = doc_ticker.upper()

        if resolved:
            entity["resolved_ticker"] = resolved
            entity["is_company"] = True
            # Boost confidence for known companies
            entity["confidence"] = min(1.0, confidence + 0.05)
        else:
            # Check if financial context suggests this is a company
            has_financial_context = bool(FINANCIAL_CONTEXT_RE.search(context_text))
            entity["is_company"] = has_financial_context
            if has_financial_context:
                entity["confidence"] = min(1.0, confidence + 0.02)

        return entity
