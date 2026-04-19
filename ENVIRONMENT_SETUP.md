# Environment Setup

Configure LLM providers and data paths for Academic RAG.

## Quick Setup

Create `.env` file in `rag-model/` folder (copy from `.env.example`):

```env
# LLM Provider: gemini (cloud) atau ollama (lokal)
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your-api-key

# Data directories (relative to rag-model/)
DATA_DIR=../data
DOCUMENTS_DIR=../data/documents
PROCESSED_DIR=../data/processed
CHROMA_PERSIST_DIRECTORY=../data/chroma_db
INDEX_CACHE_DIR=../data/cache
```

## Data Directory Options

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Base data folder | `../data` |
| `DOCUMENTS_DIR` | Source PDFs | `../data/documents` |
| `PROCESSED_DIR` | Processed chunks | `../data/processed` |
| `CHROMA_PERSIST_DIRECTORY` | Vector DB | `../data/chroma_db` |
| `INDEX_CACHE_DIR` | BM25 cache | `../data/cache` |

## LLM Provider Options

| Provider | Cost | Setup |
|----------|------|-------|
| Gemini | Free tier / Pay-as-you-go | API key dari Google AI Studio |
| Ollama | Free | Install lokal |

## Using Gemini (Recommended)

```env
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your-api-key
```

## Using Ollama (Free, Local)

```bash
# Install from https://ollama.ai
ollama pull llama3.2:latest
ollama serve
```

### Konfigurasi `.env`

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434
```

## Verify Configuration

```bash
python -c "from rag_model import AcademicRAG; print(AcademicRAG().health_check())"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama not connecting | Run `ollama serve` |
| API key error | Check `.env` path |
| Model not found | `ollama pull <model>` |

