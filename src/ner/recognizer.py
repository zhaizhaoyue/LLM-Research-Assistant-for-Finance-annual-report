"""Core NER engine wrapping HuggingFace token-classification or GLiNER pipeline."""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

# Standard NER labels -> unified label mapping (for dslim/bert-base-NER style models)
LABEL_MAP = {
    "B-PER": "PERSON", "I-PER": "PERSON",
    "B-ORG": "ORG",    "I-ORG": "ORG",
    "B-LOC": "GPE",    "I-LOC": "GPE",
    "B-MISC": "MISC",  "I-MISC": "MISC",
}

# Default entity labels for GLiNER zero-shot extraction
GLINER_DEFAULT_LABELS = [
    "organization", "person", "location", "money",
    "date", "percent", "law",
]

# GLiNER label -> unified label mapping
GLINER_LABEL_MAP = {
    "organization": "ORG",
    "person": "PERSON",
    "location": "GPE",
    "money": "MONEY",
    "date": "DATE",
    "percent": "PERCENT",
    "law": "LAW",
}


class BaseRecognizer(ABC):
    """Abstract base for NER recognizers."""

    @abstractmethod
    def recognize(self, text: str) -> list[dict]:
        ...

    @abstractmethod
    def recognize_batch(self, texts: list[str]) -> list[list[dict]]:
        ...


class HuggingFaceRecognizer(BaseRecognizer):
    """NER via HuggingFace token-classification pipeline (e.g. dslim/bert-base-NER)."""

    def __init__(self, model_name: str, device: int, batch_size: int,
                 confidence_threshold: float):
        from transformers import pipeline
        self.confidence_threshold = confidence_threshold
        log.info("Loading HuggingFace NER model: %s (device=%d)", model_name, device)
        self.pipe = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            device=device,
            aggregation_strategy="simple",
            batch_size=batch_size,
        )

    def recognize(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []
        raw_entities = self.pipe(text[:2048])
        results = []
        for ent in raw_entities:
            score = float(ent["score"])
            if score < self.confidence_threshold:
                continue
            label = LABEL_MAP.get(ent["entity_group"], ent["entity_group"])
            results.append({
                "text": ent["word"],
                "label": label,
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "confidence": round(score, 4),
            })
        return results

    def recognize_batch(self, texts: list[str]) -> list[list[dict]]:
        if not texts:
            return []
        truncated = [t[:2048] if t else "" for t in texts]
        all_results = self.pipe(truncated)

        if texts and isinstance(all_results, list) and all_results \
                and isinstance(all_results[0], dict):
            all_results = [all_results]

        batch_output = []
        for raw_entities in all_results:
            entities = []
            for ent in raw_entities:
                score = float(ent["score"])
                if score < self.confidence_threshold:
                    continue
                label = LABEL_MAP.get(ent["entity_group"], ent["entity_group"])
                entities.append({
                    "text": ent["word"],
                    "label": label,
                    "start": int(ent["start"]),
                    "end": int(ent["end"]),
                    "confidence": round(score, 4),
                })
            batch_output.append(entities)
        return batch_output


class GLiNERRecognizer(BaseRecognizer):
    """NER via GLiNER zero-shot entity recognition."""

    def __init__(self, model_name: str, device: int,
                 confidence_threshold: float,
                 labels: list[str] | None = None):
        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError(
                "GLiNER is not installed. Install with: pip install gliner>=0.2.0"
            )
        self.confidence_threshold = confidence_threshold
        self.labels = labels or GLINER_DEFAULT_LABELS
        device_str = "cpu" if device < 0 else f"cuda:{device}"
        log.info("Loading GLiNER model: %s (device=%s)", model_name, device_str)
        self.model = GLiNER.from_pretrained(model_name)
        if device >= 0:
            self.model = self.model.to(device_str)

    def _convert(self, raw_entities: list) -> list[dict]:
        results = []
        for ent in raw_entities:
            score = float(ent.get("score", 0.0))
            if score < self.confidence_threshold:
                continue
            raw_label = ent.get("label", "MISC").lower()
            label = GLINER_LABEL_MAP.get(raw_label, raw_label.upper())
            results.append({
                "text": ent.get("text", ""),
                "label": label,
                "start": int(ent.get("start", 0)),
                "end": int(ent.get("end", 0)),
                "confidence": round(score, 4),
            })
        return results

    def recognize(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []
        raw = self.model.predict_entities(text[:2048], self.labels)
        return self._convert(raw)

    def recognize_batch(self, texts: list[str]) -> list[list[dict]]:
        if not texts:
            return []
        results = []
        for text in texts:
            results.append(self.recognize(text))
        return results


class NERRecognizer:
    """Unified NER recognizer with pluggable backends.

    Model name formats:
        - "dslim/bert-base-NER" (default) -> HuggingFace pipeline
        - "gliner:urchade/gliner_base"    -> GLiNER zero-shot
        - Any other HuggingFace model      -> HuggingFace pipeline
    """

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        device: int = -1,
        batch_size: int = 32,
        confidence_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        if model_name.startswith("gliner:"):
            actual_model = model_name[len("gliner:"):]
            self._backend = GLiNERRecognizer(
                model_name=actual_model,
                device=device,
                confidence_threshold=confidence_threshold,
            )
        else:
            self._backend = HuggingFaceRecognizer(
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                confidence_threshold=confidence_threshold,
            )

    def recognize(self, text: str) -> list[dict]:
        """Run NER on a single text string."""
        return self._backend.recognize(text)

    def recognize_batch(self, texts: list[str]) -> list[list[dict]]:
        """Run NER on a batch of texts for better throughput."""
        return self._backend.recognize_batch(texts)
