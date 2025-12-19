# Konfigurasi

Opsi konfigurasi untuk RAG Model.

## Environment Variables

Buat file `.env` di folder `rag-model/`:

```env
# Provider LLM
LLM_PROVIDER=gemini          # gemini, openai, anthropic, ollama

# Google Gemini (Rekomendasi)
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-2.0-flash

# Parameter LLM
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4000

# Ollama (Lokal)
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini

# Anthropic
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_MODEL=claude-3-haiku-20240307

# Qwen (Lokal)
QWEN_MODEL=qwen3:8b

# Llama (Lokal)
LLAMA_MODEL=llama3.2:latest

# Path Data
DATA_PATH=./data
CHROMA_DIR=./data/chroma_db
BM25_CACHE_DIR=./data/bm25_cache
```

## Konfigurasi Pipeline

```python
from rag_model import AcademicRAG

# Mode research (metrik detail)
rag = AcademicRAG(research_mode=True)

# Format response
rag = AcademicRAG(response_format="full")  # simple, full, api
```

## Tipe Pipeline

| Tipe | Deskripsi |
|------|-----------|
| `baseline` | Hanya vector search |
| `advanced` | Hybrid + reranking |

## Konfigurasi Chunking

Via rag-api (port 5001):

```bash
curl -X PUT "http://localhost:5001/api/chunking/config" \
     -H "Content-Type: application/json" \
     -d '{"chunk_size": 1000, "chunk_overlap": 200}'
```

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `chunk_size` | 1000 | Karakter per chunk |
| `chunk_overlap` | 200 | Overlap antar chunk |

## Provider LLM

### Google Gemini (Rekomendasi)

Tersedia tier gratis, kualitas baik.

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
```

### Ollama (Lokal)

Gratis, lokal, tidak perlu API key.

1. Install: https://ollama.ai
2. Pull model: `ollama pull llama2`
3. Konfigurasi:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama2
   ```

### OpenAI

Berbayar, kualitas tinggi.

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Anthropic

Berbayar, sangat baik untuk konteks panjang.

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

### Qwen (Lokal)

Menggunakan Ollama sebagai backend.

```env
LLM_PROVIDER=qwen
QWEN_MODEL=qwen3:8b
# Pastikan: ollama run qwen3:8b
```

### Llama (Lokal)

Menggunakan Ollama sebagai backend.

```env
LLM_PROVIDER=llama
LLAMA_MODEL=llama3.2:latest
# Pastikan: ollama run llama3.2:latest
```

## Parameter Retrieval

Konfigurasi untuk pipeline retrieval:

```env
# Tipe pipeline: baseline (hanya vector) atau advanced (hybrid+rerank)
PIPELINE_TYPE=advanced

# Jumlah hasil retrieval
MAX_RESULTS=5
TOP_K_RETRIEVAL=50
TOP_K_RERANK=5

# Bobot Hybrid Search
BM25_WEIGHT=0.4
VECTOR_WEIGHT=0.6

# Reranking
USE_RERANKING=true
```

## Path Data

Data disimpan di folder `data/`:

```
rag-model/
└── data/
    ├── documents/   # File PDF/DOC
    ├── processed/   # Chunks (JSON)
    ├── chroma_db/   # Vector database
    └── bm25_cache/  # BM25 index cache
```
