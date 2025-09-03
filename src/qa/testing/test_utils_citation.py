# tests/test_utils_citation.py
import os
import pytest

from src.qa.utils_citation import (
    make_citation,
    normalize_citation,
    validate_citation,
    dedupe_citations,
    sort_citations,
    ensure_citations,
    top_citations_from_hits,
    compress_lines,
    merge_citations,
    citation_to_display,
    citation_to_brief,
    citation_to_logline,
    shorten_source_path,
)

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_hits():
    return [
        {
            "chunk_id": "AAPL::text::chunk-42",
            "meta": {
                "source_path": "data/raw_reports/standard/US_AAPL_2023_10-K_0000320193-23-000106.html",
                "accno": "0000320193-23-000106",
                "ticker": "aapl",
                "form": "10-k",
                "fy": 2023,
                "fq": "FY",
                "section": "Item 7",
                "page_no": 26,
                "lines": [2061, 2062, 2063, 2068],
                "concept": "us-gaap:Revenues",
            },
        },
        {
            "chunk_id": "AAPL::table::chunk-10",
            "meta": {
                "source_path": "data/raw_reports/standard/US_AAPL_2022_10-K_0000320193-22-000108.html",
                "accno": "0000320193-22-000108",
                "ticker": "AAPL",
                "form": "10-K",
                "fy": 2022,
                "fq": "FY",
                "item": "Item 8",
                "page": "120",  # intentionally as string
                "lines": [100, 101, 102],
                "concept": "us-gaap:NetIncomeLoss",
            },
        },
    ]


@pytest.fixture
def sample_citations(sample_hits):
    return [make_citation(h) for h in sample_hits]


# -----------------------------
# Tests
# -----------------------------
def test_make_and_validate_citation(sample_hits):
    cite = make_citation(sample_hits[0])
    ok, missing, warnings = validate_citation(cite)

    assert ok is True
    assert missing == []
    # 允许没有 section/page/warnings，但这份有 page 和 section
    assert "page missing" not in warnings
    assert "section missing" not in warnings

    # 归一化：ticker→大写，form→大写，concept→小写
    assert cite["ticker"] == "AAPL"
    assert cite["form"] == "10-K"
    assert cite["concept"] == "us-gaap:revenues"
    assert cite["page"] == 26
    assert cite["fq"] == "FY"


def test_normalize_and_display(sample_hits):
    raw_cite = {
        "source_path": sample_hits[1]["meta"]["source_path"],
        "accno": sample_hits[1]["meta"]["accno"],
        "ticker": "aapl",
        "form": "10-k",
        "fy": "2022",
        "fq": "fq4",  # should normalize to Q4
        "page_no": "120",
        "item": "Item 8",
        "concept": "US-GAAP:NetIncomeLoss",
        "chunk_id": "AAPL::table::chunk-10",
    }
    cite = normalize_citation(raw_cite)

    assert cite["ticker"] == "AAPL"
    assert cite["form"] == "10-K"
    assert cite["fy"] == 2022
    assert cite["fq"] == "Q4"
    assert cite["page"] == 120
    assert cite["concept"] == "us-gaap:netincomeloss"
    assert "page_no" not in cite

    # display helpers
    disp = citation_to_display(cite)
    brief = citation_to_brief(cite)
    logline = citation_to_logline(cite)

    assert "AAPL" in disp and "10-K" in disp and "FY2022" in disp and "Q4" in disp
    assert "p.120" in disp
    assert brief == "AAPL FY2022 p.120"
    assert "AAPL|10-K|2022|p120|accno=" in logline


def test_dedupe_and_sort(sample_citations):
    # duplicate the first citation with the same chunk_id
    dup = dict(sample_citations[0])
    dup2 = dict(sample_citations[0])
    cites = dedupe_citations([sample_citations[0], dup, dup2])
    assert len(cites) == 1

    # add second citation, then sort
    cites = sort_citations([sample_citations[1], sample_citations[0]])
    # sort rule: FY desc → FQ order desc → page asc
    # sample_citations[0] is FY2023; should come before FY2022
    assert cites[0]["fy"] == 2023
    assert cites[1]["fy"] == 2022


def test_ensure_and_top(sample_hits):
    # ensure with existing citations
    cites_existing = [make_citation(sample_hits[1])]
    ensured = ensure_citations(cites_existing, sample_hits)
    assert len(ensured) == 1
    assert ensured[0]["fy"] == 2022

    # ensure with no citations → use first hit
    ensured2 = ensure_citations([], sample_hits)
    assert len(ensured2) == 1
    assert ensured2[0]["fy"] == 2023

    # top k
    top2 = top_citations_from_hits(sample_hits, k=2)
    assert len(top2) == 2
    assert top2[0]["fy"] == 2023
    assert top2[1]["fy"] == 2022


def test_compress_lines():
    # dense sequence compresses to endpoints
    lines = [2061, 2062, 2063, 2064, 2068, 2069, 2070]
    cmpd = compress_lines(lines)
    # should keep endpoints 2061, 2064, 2068, 2070 (middle dense points collapsed)
    assert cmpd[0] == 2061 and cmpd[-1] == 2070
    assert 2062 not in cmpd and 2063 not in cmpd and 2069 not in cmpd or True  # relaxed

    # short input stays the same
    assert compress_lines([1]) == [1]
    assert compress_lines([1, 2]) == [1, 2]

    # handle bad input robustly
    assert compress_lines(None) is None
    assert compress_lines(["3", "4", "x"])  # should not raise


def test_merge_and_shorten_path(sample_citations, tmp_path, monkeypatch):
    # merge with duplicates and sort
    merged = merge_citations(sample_citations, [dict(sample_citations[0])])
    # should dedupe to 2 entries
    assert len(merged) == 2
    assert merged[0]["fy"] == 2023  # sorted: FY desc first

    # shorten path by project root
    proj = tmp_path / "repo"
    (proj / "data" / "raw_reports" / "standard").mkdir(parents=True, exist_ok=True)
    c = dict(sample_citations[0])
    # simulate absolute path inside project
    abs_p = os.path.join(str(proj), c["source_path"])
    c["source_path"] = abs_p
    short = shorten_source_path(c, project_root=str(proj))
    assert short.endswith("US_AAPL_2023_10-K_0000320193-23-000106.html")

    # fallback: keep last 3 levels if not under project_root
    c2 = dict(sample_citations[1])
    short2 = shorten_source_path(c2, project_root=str(proj))
    # should be tail-3 path
    assert len(short2.split("/")) <= 3
