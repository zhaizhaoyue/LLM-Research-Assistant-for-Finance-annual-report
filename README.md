# SEC Filing Retrieval Pipeline

A lean, stage-oriented workflow that turns raw SEC filings into a searchable vector index and answers financial questions with hybrid retrieval plus optional LLMs. All stages are scriptable from a single CLI and can be run independently.

## Features
- End-to-end pipeline: download → postprocess → parse → clean → index → chunk → embed.
- Hybrid search: BM25 + dense retrieval with optional cross-encoder rerank, exposed via `python -m src.cli query`.
- Reproducible CLI with per-stage flags and sensible defaults for data locations.
- Windows-friendly; no container required (Dockerfile included for Linux users).
- Optional XBRL numeric pipeline (requires the AIE numeric module; guarded if absent).

## Pipeline Stages
| Stage | Entry point | Input → Output | Purpose |
| --- | --- | --- | --- |
| download | `src/ingest/download.py` | data/companies.csv → data/raw_reports/sec-edgar-filings/ | Fetch latest 10-K/10-Q filings from EDGAR. |
| postprocess | `src/parse/postprocess.py` | raw_reports/sec-edgar-filings/ → raw_reports/standard/ | Normalize folder names, collect HTML/XBRL payloads. |
| parse | `src/parse/text.py` | raw_reports/standard/ → data/processed/ | Extract metadata and produce `text.jsonl`. |
| clean | `src/cleaning/text.py` | data/processed/ → data/clean/ | Sentence-level cleaning with numeric helpers. |
| index | `src/index/schema.py` | data/clean/ → data/silver/ | Validate and emit “silver” records. |
| chunk | `src/chunking/chunk.py` | data/silver/ → data/chunked/ | Build retrieval-ready text chunks with headings. |
| embed | `src/chunking/embedding.py` | data/chunked/ → data/index/ | Encode chunks, build FAISS index, write id/meta maps. |

## Quick Start
1) Install deps  
```bash
python -m venv .venv
.venv\Scripts\activate   # PowerShell on Windows
pip install -r requirements.txt
```
2) Set your SEC email (required for downloads)  
`$env:SEC_EMAIL="your-email@example.com"`
3) Run the full pipeline  
```bash
python -m src.cli run --stages download,postprocess,parse,clean,index,chunk,embed --download-email $env:SEC_EMAIL
```

## CLI Cheat Sheet
- Inspect commands: `python -m src.cli --help`
- Single stage: `python -m src.cli chunk --input data/silver --output data/chunked --chunk-workers 4`
- Rebuild embeddings with a different model: `python -m src.cli embed --embed-model BAAI/bge-small-en-v1.5 --embed-use-title`
- Query the index:  
```bash
python -m src.cli query \
  --query "What was Apple's 2022 revenue?" \
  --index-dir data/index \
  --chunk-dir data/chunked \
  --bm25-topk 400 --dense-topk 400 --ce-candidates 256 --ce-weight 0.7
```
Add `--llm-base-url`, `--llm-model`, and provide an API key when prompted to enable LLM answers.

## Data Flow
`data/companies.csv → data/raw_reports/sec-edgar-filings/ → data/raw_reports/standard/ → data/processed/ → data/clean/ → data/silver/ → data/chunked/ → data/index/`

## Config Notes
- Default paths live in `src/cli.py`; override any stage via CLI flags.
- Downloads respect SEC rate limits; tune `--download-sleep`.
- Chunk/embedding preferences can be steered with `--chunk-max-tokens`, `--chunk-max-chars`, `--embed-model`, and `--embed-prefer-keywords`.

## Tests
Run the retrieval smoke test:  
```bash
pytest tests/test_query_pipeline.py
```

## Optional: XBRL Numeric Pipeline
The XBRL numeric pipeline (`src/xbrl_pipeline/`) depends on `src/aie_for_numeric_retrieval`, which is currently absent. Reinstall that module to enable numeric QA; otherwise the import guard will raise a clear error.
