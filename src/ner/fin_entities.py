"""Rule-based extractors for finance-specific entity types
that general NER models do not cover: TICKER and FIN_METRIC.
"""
from __future__ import annotations
import re

# ---- TICKER recognition ----

# Common US stock ticker format: 1-5 uppercase letters with optional keyword prefix
RE_TICKER = re.compile(
    r"(?<![A-Za-z])"
    r"(?:ticker|symbol|stock)[\s:]+?"
    r"([A-Z]{1,5})"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)

# Ticker in parentheses: "Apple Inc. (AAPL)"
RE_TICKER_PAREN = re.compile(
    r"\(([A-Z]{1,5})\)"
)

# Known non-ticker abbreviations to avoid false positives
TICKER_BLOCKLIST = {
    "THE", "AND", "FOR", "NOT", "ALL", "BUT", "ARE", "WAS", "HAS",
    "HAD", "HIS", "HER", "ITS", "OUR", "WHO", "HOW", "MAY", "CAN",
    "DID", "GET", "LET", "SAY", "SHE", "TOO", "USE", "HIM", "OLD",
    "SEE", "NOW", "WAY", "ANY", "NEW", "SEC", "IRS", "CEO", "CFO",
    "COO", "CTO", "USA", "USD", "EUR", "JPY", "GBP", "CNY", "FYI",
    "IPO", "LLC", "LLP", "INC", "LTD", "PLC", "ETF", "FAQ", "PDF",
    "XML", "HTML", "API", "SQL", "NER", "AI", "ML",
}

# ---- FIN_METRIC recognition ----

FIN_METRIC_TERMS = {
    "EBITDA", "EBIT", "EPS", "P/E", "ROE", "ROA", "ROI", "ROIC",
    "CAGR", "WACC", "DCF", "NAV", "NTA", "FCF", "CAPEX", "OPEX",
    "SG&A", "COGS", "R&D", "D&A", "YoY", "QoQ", "MoM",
    "TTM", "LTM", "NTM", "FWD",
    "EV/EBITDA", "EV/EBIT", "EV/Revenue", "P/E", "P/B", "P/S",
    "P/FCF", "PEG",
}

RE_FIN_METRIC = re.compile(
    r"(?<![A-Za-z/])"
    r"(" + "|".join(re.escape(t) for t in sorted(FIN_METRIC_TERMS, key=len, reverse=True)) + ")"
    r"(?![A-Za-z])",
)


def extract_tickers(text: str) -> list[dict]:
    """Extract stock ticker symbols from text."""
    results = []
    seen: set[str] = set()
    for pattern in [RE_TICKER, RE_TICKER_PAREN]:
        for m in pattern.finditer(text):
            ticker = m.group(1) if m.lastindex else m.group(0)
            ticker = ticker.upper()
            if ticker in TICKER_BLOCKLIST or ticker in seen:
                continue
            if len(ticker) < 1:
                continue
            seen.add(ticker)
            results.append({
                "text": ticker,
                "label": "TICKER",
                "start": m.start(1) if m.lastindex else m.start(),
                "end": m.end(1) if m.lastindex else m.end(),
                "confidence": 0.80,
            })
    return results


def extract_fin_metrics(text: str) -> list[dict]:
    """Extract financial metric terms from text."""
    results = []
    seen: set[str] = set()
    for m in RE_FIN_METRIC.finditer(text):
        term = m.group(1)
        if term in seen:
            continue
        seen.add(term)
        results.append({
            "text": term,
            "label": "FIN_METRIC",
            "start": m.start(1),
            "end": m.end(1),
            "confidence": 1.0,
        })
    return results


def extract_custom_entities(text: str) -> list[dict]:
    """Run all custom finance entity extractors."""
    entities = []
    entities.extend(extract_tickers(text))
    entities.extend(extract_fin_metrics(text))
    return entities
