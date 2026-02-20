"""Entity relationship graph: build co-occurrence-based entity graphs from NER output."""
from __future__ import annotations
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Keywords that suggest subsidiary/parent relationship
SUBSIDIARY_RE = re.compile(
    r"\b(subsidiary|subsidiaries|parent|acquired|acquisition|merger|merged|"
    r"wholly[- ]owned|joint venture|affiliate)\b",
    re.IGNORECASE,
)


def _infer_relation(ent_a: dict, ent_b: dict, text: str) -> str:
    """Infer the relationship type between two co-occurring entities."""
    la, lb = ent_a["label"], ent_b["label"]
    # PERSON + ORG => executive_of
    if (la == "PERSON" and lb == "ORG") or (la == "ORG" and lb == "PERSON"):
        return "executive_of"
    # ORG + GPE => located_in
    if (la == "ORG" and lb == "GPE") or (la == "GPE" and lb == "ORG"):
        return "located_in"
    # ORG + ORG with subsidiary keywords => subsidiary_of
    if la == "ORG" and lb == "ORG" and SUBSIDIARY_RE.search(text):
        return "subsidiary_of"
    return "co_mentioned"


class EntityGraph:
    """In-memory entity co-occurrence graph."""

    def __init__(self):
        # node_id -> {normalized, label, mention_count, tickers: set, years: set}
        self.nodes: dict[str, dict[str, Any]] = {}
        # (src_id, tgt_id) -> {relation_type, weight, co_occurrences}
        self.edges: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _node_id(normalized: str, label: str) -> str:
        return f"{label}::{normalized}"

    def add_entity(self, normalized: str, label: str,
                   ticker: str = "", year: int | None = None) -> str:
        nid = self._node_id(normalized, label)
        if nid not in self.nodes:
            self.nodes[nid] = {
                "normalized": normalized,
                "label": label,
                "mention_count": 0,
                "tickers": set(),
                "years": set(),
            }
        node = self.nodes[nid]
        node["mention_count"] += 1
        if ticker:
            node["tickers"].add(ticker)
        if year is not None:
            node["years"].add(year)
        return nid

    def add_edge(self, src_id: str, tgt_id: str, relation: str) -> None:
        key = (min(src_id, tgt_id), max(src_id, tgt_id))
        if key not in self.edges:
            self.edges[key] = {
                "relation_type": relation,
                "weight": 0,
                "co_occurrences": 0,
            }
        edge = self.edges[key]
        edge["weight"] += 1
        edge["co_occurrences"] += 1
        # Upgrade relation if more specific
        if relation != "co_mentioned" and edge["relation_type"] == "co_mentioned":
            edge["relation_type"] = relation

    def neighbors(self, node_id: str) -> list[dict]:
        """Get all neighbors of a node with edge info."""
        results = []
        for (src, tgt), edge in self.edges.items():
            if src == node_id:
                other = tgt
            elif tgt == node_id:
                other = src
            else:
                continue
            if other in self.nodes:
                results.append({
                    **self.nodes[other],
                    "tickers": sorted(self.nodes[other]["tickers"]),
                    "years": sorted(self.nodes[other]["years"]),
                    "edge": edge,
                })
        results.sort(key=lambda x: x["edge"]["weight"], reverse=True)
        return results

    def subgraph(self, node_id: str, depth: int = 1) -> dict:
        """Extract a subgraph centered on node_id up to given depth."""
        visited: set[str] = set()
        frontier = {node_id}
        sg_nodes = {}
        sg_edges = []

        for _ in range(depth + 1):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                if nid in self.nodes:
                    node = self.nodes[nid]
                    sg_nodes[nid] = {
                        **node,
                        "tickers": sorted(node["tickers"]),
                        "years": sorted(node["years"]),
                    }
                for (src, tgt), edge in self.edges.items():
                    if src == nid or tgt == nid:
                        other = tgt if src == nid else src
                        next_frontier.add(other)
                        sg_edges.append({"source": src, "target": tgt, **edge})
            frontier = next_frontier - visited

        # Deduplicate edges
        seen_edges: set[tuple[str, str]] = set()
        unique_edges = []
        for e in sg_edges:
            key = (e["source"], e["target"])
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        return {"nodes": sg_nodes, "edges": unique_edges}

    def summary(self) -> str:
        return (f"EntityGraph: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def to_json(self, path: str | Path) -> None:
        """Serialize graph to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {
                nid: {**info, "tickers": sorted(info["tickers"]),
                       "years": sorted(info["years"])}
                for nid, info in self.nodes.items()
            },
            "edges": [
                {"source": src, "target": tgt, **info}
                for (src, tgt), info in self.edges.items()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("Graph saved to %s (%d nodes, %d edges)",
                 path, len(self.nodes), len(self.edges))

    @classmethod
    def from_json(cls, path: str | Path) -> EntityGraph:
        """Load graph from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        g = cls()
        for nid, info in data.get("nodes", {}).items():
            g.nodes[nid] = {
                **info,
                "tickers": set(info.get("tickers", [])),
                "years": set(info.get("years", [])),
            }
        for edge in data.get("edges", []):
            key = (edge["source"], edge["target"])
            g.edges[key] = {
                "relation_type": edge.get("relation_type", "co_mentioned"),
                "weight": edge.get("weight", 1),
                "co_occurrences": edge.get("co_occurrences", 1),
            }
        return g


def build_graph(ner_dir: str | Path) -> EntityGraph:
    """Build entity co-occurrence graph from NER output directory.

    Reads all text_corpus.jsonl files under ner_dir and builds
    edges between entities that co-occur in the same sentence/row.
    """
    ner_dir = Path(ner_dir)
    graph = EntityGraph()
    files = sorted(ner_dir.rglob("text_corpus.jsonl"))

    if not files:
        log.warning("No text_corpus.jsonl files found under %s", ner_dir)
        return graph

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entities = row.get("entities", [])
                if len(entities) < 2:
                    # Need at least 2 entities for a relationship
                    if entities:
                        ent = entities[0]
                        graph.add_entity(
                            ent.get("normalized", ent["text"]),
                            ent["label"],
                            row.get("ticker", ""),
                            row.get("year"),
                        )
                    continue

                text = row.get("text", "")
                ticker = row.get("ticker", "")
                year = row.get("year")

                # Add all entities as nodes
                node_ids = []
                for ent in entities:
                    nid = graph.add_entity(
                        ent.get("normalized", ent["text"]),
                        ent["label"], ticker, year,
                    )
                    node_ids.append((nid, ent))

                # Build edges for all pairs
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        nid_a, ent_a = node_ids[i]
                        nid_b, ent_b = node_ids[j]
                        if nid_a == nid_b:
                            continue
                        relation = _infer_relation(ent_a, ent_b, text)
                        graph.add_edge(nid_a, nid_b, relation)

    log.info("Built %s from %d files", graph.summary(), len(files))
    return graph
