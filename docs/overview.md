# Overview

RAG Model adalah sistem Retrieval-Augmented Generation untuk konten akademik berbahasa Indonesia.

## Apa itu RAG?

RAG menggabungkan:
1. **Retrieval** - Mencari dokumen yang relevan
2. **Augmentation** - Menambahkan konteks ke prompt
3. **Generation** - Menghasilkan jawaban menggunakan LLM

## Arsitektur

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Query    │────▶│  Retrieval  │────▶│  Generator  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
               ┌───────────┴────────────┐
               ▼                        ▼
       ┌──────────────┐          ┌──────────────┐
       │ Vector Store │          │  BM25 Index  │
       │  (ChromaDB)  │          │   (Python)   │
       └──────────────┘          └──────────────┘
```

## Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Hybrid Search** | Kombinasi vector (semantic) dan BM25 (keyword) search |
| **RRF Fusion** | Reciprocal Rank Fusion untuk kombinasi hasil |
| **Reranking** | Cross-encoder reranking untuk akurasi lebih tinggi |
| **LLM Backend** | Mendukung Gemini (Cloud) dan Ollama (Lokal) |

## Pipeline

### Baseline Pipeline
- Hanya vector search
- Cepat tapi kurang akurat
- Cocok untuk query sederhana

### Advanced Pipeline
- Hybrid search (Vector + BM25)
- RRF fusion
- Cross-encoder reranking
- Akurasi lebih tinggi

## Struktur Proyek (Polyrepo)

```
Repository GitHub:
├── rag-model/          # Core RAG (repository ini)
│   ├── rag_model/      # Implementasi model
│   ├── scripts/        # Script utility
│   └── docs/           # Dokumentasi
├── rag-api/            # REST API (submodule: rag-model)
├── rag-web/            # Laravel frontend
└── rag-deploy/         # Docker orchestration
```

## Komponen Utama

| Komponen | File | Deskripsi |
|----------|------|-----------|
| Pipeline | `core/pipeline.py` | Orchestrator utama |
| Config | `core/config.py` | Konfigurasi sistem |
| Vector Store | `indexing/vector_store.py` | ChromaDB + IndoBERT |
| BM25 Index | `indexing/bm25_index.py` | Sparse index |
| Reranker | `models/reranker.py` | Cross-encoder |
| LLM | `models/llm_generator.py` | Multi-provider LLM |
