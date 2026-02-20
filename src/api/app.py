"""FastAPI application exposing NER entity query endpoints."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

app = FastAPI(
    title="SEC Filing NER API",
    description="Named Entity Recognition and entity query service for SEC filings.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Lazy-loaded global resources
# ---------------------------------------------------------------------------
_recognizer = None
_graph = None
_timelines = None

DEFAULT_NER_DIR = Path("data/ner")
DEFAULT_GRAPH_PATH = Path("data/ner/entity_graph.json")
DEFAULT_TIMELINES_PATH = Path("data/ner/timelines.json")


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        from src.ner.recognizer import NERRecognizer
        _recognizer = NERRecognizer(device=-1, batch_size=1)
    return _recognizer


def _get_graph():
    global _graph
    if _graph is None:
        from src.ner.graph import EntityGraph
        if DEFAULT_GRAPH_PATH.exists():
            _graph = EntityGraph.from_json(DEFAULT_GRAPH_PATH)
        else:
            _graph = EntityGraph()
    return _graph


def _get_timelines():
    global _timelines
    if _timelines is None:
        from src.ner.timeline import load_timelines
        if DEFAULT_TIMELINES_PATH.exists():
            _timelines = load_timelines(DEFAULT_TIMELINES_PATH)
        else:
            _timelines = {}
    return _timelines


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    text: str = Field(..., description="Text to extract entities from")
    model: str = Field("dslim/bert-base-NER", description="NER model name")
    threshold: float = Field(0.5, ge=0, le=1, description="Confidence threshold")


class EntityItem(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized: str = ""


class ExtractResponse(BaseModel):
    entities: list[EntityItem]
    entity_labels: list[str]


class SearchResult(BaseModel):
    normalized: str
    label: str
    mention_count: int
    tickers: list[str]
    years: list[int]


class TimelineEntry(BaseModel):
    year: int
    tickers: list[str] = []
    forms: list[str] = []
    mention_count: int
    sample_contexts: list[str] = []


class TimelineResponse(BaseModel):
    normalized: str
    label: str
    total_mentions: int
    entries: list[TimelineEntry]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/entities/extract", response_model=ExtractResponse)
def extract_entities(req: ExtractRequest):
    """Extract named entities from the given text."""
    from src.ner.fin_entities import extract_custom_entities
    from src.ner.normalizer import normalize_entity, deduplicate_entities

    recognizer = _get_recognizer()
    model_ents = recognizer.recognize(req.text)
    custom_ents = extract_custom_entities(req.text)
    all_ents = model_ents + custom_ents
    all_ents = [normalize_entity(e) for e in all_ents]
    all_ents = deduplicate_entities(all_ents)
    all_ents.sort(key=lambda e: e["start"])

    labels = sorted(set(e["label"] for e in all_ents))
    return ExtractResponse(
        entities=[EntityItem(**e) for e in all_ents],
        entity_labels=labels,
    )


@app.get("/api/entities/search", response_model=list[SearchResult])
def search_entities(
    label: Optional[str] = Query(None, description="Entity label filter (e.g., ORG, PERSON)"),
    name: Optional[str] = Query(None, description="Entity name (partial match)"),
    ticker: Optional[str] = Query(None, description="Filter by company ticker"),
    year: Optional[int] = Query(None, description="Filter by year"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
):
    """Search entities in the entity graph."""
    graph = _get_graph()
    results = []

    name_upper = name.strip().upper() if name else None

    for nid, node in graph.nodes.items():
        if label and node["label"] != label.upper():
            continue
        if name_upper and name_upper not in node["normalized"]:
            continue
        if ticker and ticker.upper() not in node.get("tickers", set()):
            continue
        if year and year not in node.get("years", set()):
            continue
        results.append(SearchResult(
            normalized=node["normalized"],
            label=node["label"],
            mention_count=node["mention_count"],
            tickers=sorted(node.get("tickers", set())),
            years=sorted(node.get("years", set())),
        ))

    results.sort(key=lambda x: x.mention_count, reverse=True)
    return results[:limit]


@app.get("/api/entities/timeline/{normalized_name}", response_model=TimelineResponse)
def get_entity_timeline(
    normalized_name: str,
    label: str = Query("ORG", description="Entity label"),
):
    """Get the temporal timeline for a specific entity."""
    timelines = _get_timelines()
    key = f"{label.upper()}::{normalized_name.strip().upper()}"
    timeline = timelines.get(key)
    if not timeline:
        raise HTTPException(status_code=404, detail=f"No timeline found for {key}")
    return TimelineResponse(**timeline)


@app.get("/api/entities/graph/{normalized_name}")
def get_entity_graph(
    normalized_name: str,
    label: str = Query("ORG", description="Entity label"),
    depth: int = Query(1, ge=1, le=3, description="Graph traversal depth"),
):
    """Get the relationship subgraph centered on an entity."""
    from src.ner.graph import EntityGraph
    graph = _get_graph()
    node_id = EntityGraph._node_id(normalized_name.strip().upper(), label.upper())

    if node_id not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Entity not found: {node_id}")

    sg = graph.subgraph(node_id, depth=depth)
    return sg


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
