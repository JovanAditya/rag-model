# RAG Model

Core RAG (Retrieval-Augmented Generation) model dan pipeline untuk sistem tanya-jawab akademik Universitas Mercu Buana.

## 📋 Deskripsi

Repository ini berisi implementasi model RAG dengan:
- **Hybrid Search**: Kombinasi Vector Search (ChromaDB) + BM25 (Sparse)
- **IndoBERT Embedding**: Menggunakan `indobenchmark/indobert-base-p2`
- **Cross-Encoder Reranking**: Untuk meningkatkan akurasi retrieval
- **Multi-LLM Support**: Gemini, OpenAI, Anthropic, Ollama, Qwen, Llama

## 📚 Dokumentasi

Dokumentasi lengkap tersedia di folder [`docs/`](docs/):

| Dokumen | Deskripsi |
|---------|-----------|
| [Getting Started](docs/getting-started.md) | Panduan memulai |
| [Overview](docs/overview.md) | Arsitektur sistem |
| [Configuration](docs/configuration.md) | Opsi konfigurasi |
| [Data Preparation](docs/data-preparation.md) | Persiapan dokumen |
| [Usage](docs/usage.md) | Contoh penggunaan |
| [Scripts](docs/scripts.md) | Referensi script |
| [API Reference](docs/api-reference.md) | Dokumentasi API |
| [Troubleshooting](docs/troubleshooting.md) | Pemecahan masalah |

## 📁 Struktur Direktori

```
rag-model/
├── rag_model/
│   ├── core/           # Pipeline dan konfigurasi
│   ├── indexing/       # ChromaDB + BM25
│   ├── models/         # Reranker, LLM, Context Builder
│   └── utils/          # Helper functions
├── scripts/            # Utility scripts
├── docs/               # Dokumentasi lengkap
├── data/               # Data (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

## ⚙️ Instalasi Cepat

```bash
# Clone repository
git clone https://github.com/JovanAditya/rag-model.git
cd rag-model

# Install menggunakan environment.yml (RECOMMENDED)
conda env create -f environment.yml
conda activate academic-rag

# (Opsional) Install GPU support untuk PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Konfigurasi environment
cp .env.example .env
# Edit .env dengan API key
```

> 📖 Untuk panduan lengkap, lihat [docs/getting-started.md](docs/getting-started.md)

## 🔧 Konfigurasi

Edit file `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
EMBEDDING_MODEL=indobenchmark/indobert-base-p2

# Parameter LLM
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4000
```

> 📖 Untuk opsi lengkap, lihat [docs/configuration.md](docs/configuration.md)

## 🚀 Penggunaan

```python
from rag_model.core.pipeline import AdvancedAcademicRAG
from rag_model.core.config import RAGConfig

config = RAGConfig()
rag = AdvancedAcademicRAG(config)

result = rag.query("Apa visi Universitas Mercu Buana?")
print(result['answer'])
```

> 📖 Untuk contoh lengkap, lihat [docs/usage.md](docs/usage.md)

## 📦 Dependencies

| Package | Versi |
|---------|-------|
| chromadb | >= 0.4.0 |
| sentence-transformers | >= 2.2.0 |
| torch | >= 2.0.0 |
| google-generativeai | >= 0.3.0 |

## 🔗 Repository Terkait

| Repository | Deskripsi |
|------------|-----------|
| [rag-api](https://github.com/JovanAditya/rag-api) | REST API |
| [rag-web](https://github.com/JovanAditya/rag-web) | Laravel Frontend |
| [rag-deploy](https://github.com/JovanAditya/rag-deploy) | Docker Orchestration |

## 📄 Lisensi

MIT License

---

*Bagian dari proyek skripsi Sistem RAG Akademik - Universitas Mercu Buana*
