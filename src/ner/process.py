"""NER pipeline stage: reads clean/ JSONL, runs NER, writes to ner/ output."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from .recognizer import NERRecognizer
from .fin_entities import extract_custom_entities
from .normalizer import normalize_entity, deduplicate_entities
from .disambiguator import Disambiguator

log = logging.getLogger(__name__)

# Global recognizer and disambiguator (initialized once per process)
_recognizer: NERRecognizer | None = None
_disambiguator: Disambiguator | None = None


def _init_recognizer(model_name: str, device: int, batch_size: int, threshold: float):
    global _recognizer, _disambiguator
    _recognizer = NERRecognizer(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        confidence_threshold=threshold,
    )
    _disambiguator = Disambiguator()


def process_file(
    input_path: Path,
    output_path: Path,
    model_name: str = "dslim/bert-base-NER",
    device: int = -1,
    batch_size: int = 32,
    confidence_threshold: float = 0.5,
) -> dict:
    """Process a single text_corpus.jsonl file with NER.

    Returns: {"input": str, "output": str, "total": int, "with_entities": int}
    """
    global _recognizer, _disambiguator
    if _recognizer is None:
        _init_recognizer(model_name, device, batch_size, confidence_threshold)

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {"input": str(input_path), "output": str(output_path),
                "total": 0, "with_entities": 0}

    # Batch NER inference
    texts = [r.get("text", "") or "" for r in rows]
    all_model_entities = _recognizer.recognize_batch(texts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with_entities = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for row, model_ents in zip(rows, all_model_entities):
            text = row.get("text", "") or ""

            # Merge model entities + custom financial entities
            custom_ents = extract_custom_entities(text)
            all_ents = model_ents + custom_ents

            # Normalize + disambiguate + deduplicate
            all_ents = [normalize_entity(e) for e in all_ents]
            if _disambiguator:
                doc_ticker = row.get("ticker", "")
                all_ents = [_disambiguator.disambiguate(e, text, doc_ticker) for e in all_ents]
            all_ents = deduplicate_entities(all_ents)

            # Sort by start position
            all_ents.sort(key=lambda e: e["start"])

            row["entities"] = all_ents
            row["entity_labels"] = sorted(set(e["label"] for e in all_ents))

            if all_ents:
                with_entities += 1

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "total": len(rows),
        "with_entities": with_entities,
    }


def _process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    return process_file(*args)


def process_tree(
    input_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "dslim/bert-base-NER",
    device: int = -1,
    batch_size: int = 32,
    confidence_threshold: float = 0.5,
    workers: int = 1,
):
    """Walk input_dir for text_corpus.jsonl files and run NER on each.

    Args:
        input_dir: path to data/clean/
        output_dir: path to data/ner/
        model_name: HuggingFace model for NER
        device: -1 for CPU, 0+ for CUDA GPU
        batch_size: texts per inference batch
        confidence_threshold: min score to keep an entity
        workers: parallel file workers (each loads its own model)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Collect all text_corpus.jsonl files
    files = sorted(input_dir.rglob("text_corpus.jsonl"))
    if not files:
        log.warning("No text_corpus.jsonl files found under %s", input_dir)
        return

    log.info("Found %d files to process", len(files))

    # Build (input, output) pairs
    tasks = []
    for fp in files:
        rel = fp.relative_to(input_dir)
        out_fp = output_dir / rel
        # Skip files already up to date
        if out_fp.exists() and out_fp.stat().st_mtime > fp.stat().st_mtime:
            log.debug("Skipping %s (up to date)", rel)
            continue
        tasks.append((fp, out_fp, model_name, device, batch_size, confidence_threshold))

    if not tasks:
        log.info("All files up to date, nothing to process")
        return

    log.info("Processing %d files", len(tasks))

    if workers <= 1:
        # Single process: share model instance
        _init_recognizer(model_name, device, batch_size, confidence_threshold)
        results = []
        for task in tqdm(tasks, desc="NER"):
            results.append(process_file(*task))
    else:
        # Multi-process
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_recognizer,
            initargs=(model_name, device, batch_size, confidence_threshold),
        ) as pool:
            results = list(tqdm(
                pool.map(_process_file_wrapper, tasks),
                total=len(tasks),
                desc="NER",
            ))

    total_rows = sum(r["total"] for r in results)
    total_with = sum(r["with_entities"] for r in results)
    log.info(
        "NER complete: %d files, %d rows, %d with entities (%.1f%%)",
        len(results), total_rows, total_with,
        100.0 * total_with / max(total_rows, 1),
    )
