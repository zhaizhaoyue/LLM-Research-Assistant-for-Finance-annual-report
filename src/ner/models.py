"""Pydantic models for NER entities."""
from __future__ import annotations
from pydantic import BaseModel


class Entity(BaseModel):
    """A single recognized entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0
    normalized: str = ""


class NERResult(BaseModel):
    """NER output for a single sentence."""
    entities: list[Entity] = []
    entity_labels: list[str] = []
