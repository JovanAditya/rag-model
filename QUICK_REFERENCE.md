# Quick Reference

Common commands for Academic RAG. All scripts use `.env` for defaults.

## Setup

```bash
# 1. Copy env file
cp .env.example .env

# 2. Activate environment
conda activate academic-rag
```

## Data Pipeline

```bash
# Using .env defaults (recommended)
python scripts/prepare_data.py
python scripts/build_indexes.py --verify

# Or with explicit paths
python scripts/prepare_data.py --input ../data/documents --output ../data/processed
python scripts/build_indexes.py --documents ../data/processed/chunks.json --verify
```

## Basic Usage

```python
from rag_model import AcademicRAG

rag = AcademicRAG()
result = rag.query("Apa syarat beasiswa?")
print(result['answer'])
```

## API Server

```bash
cd academic-api
uvicorn api.main:app --reload --port 5001
```

API Docs: http://localhost:5001/docs

## Key .env Variables

```env
# Data paths
DOCUMENTS_DIR=../data/documents
PROCESSED_DIR=../data/processed
CHROMA_PERSIST_DIRECTORY=../data/chroma_db
INDEX_CACHE_DIR=../data/cache

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=indobenchmark/indobert-base-p2

# LLM
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-1.5-flash
```

## Project Structure

```
project/
├── rag-model/         # Core model
├── rag-api/           # REST API
├── rag-web/           # Laravel frontend
└── data/              # Shared data
    ├── documents/     # PDFs (input)
    ├── processed/     # chunks.json
    ├── chroma_db/     # Vector DB
    └── cache/         # BM25 cache
```

## Scripts

| Script | Purpose |
|--------|---------|
| `prepare_data.py` | Chunk PDFs → JSON |
| `build_indexes.py` | Build ChromaDB + BM25 |
| `verify_system.py` | Verify installation |
| `index_manager.py` | Manage indexes |
| `fix_text_preprocessing.py` | Fix OCR issues |

## Troubleshooting

```bash
python -c "from rag_model import AcademicRAG; print('OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
