# Referensi API

Dokumentasi lengkap API untuk RAG API (`rag-api`).

> **📌 Repository**: `rag-api` berjalan di port 5001

## Base URL

```
http://localhost:5001/api
```

---

## API Query

### POST /api/query

Query sistem RAG.

**Request:**
```json
{
  "question": "Apa syarat beasiswa?",
  "pipeline_type": "advanced",
  "max_results": 5
}
```

**Response:**
```json
{
  "status": "success",
  "answer": "Syarat beasiswa adalah...",
  "confidence": 0.85,
  "sources": [...],
  "query_id": "q_abc123",
  "pipeline_used": "advanced",
  "response_time": 1.23
}
```

---

## API Dokumen

### POST /api/documents/upload

Upload dokumen.

**Request:** `multipart/form-data` dengan `file`

**Response:**
```json
{
  "status": "success",
  "document_id": "doc_abc123",
  "filename": "doc_abc123.pdf",
  "size_bytes": 1024000
}
```

### GET /api/documents

Daftar semua dokumen.

### GET /api/documents/{id}

Detail dokumen.

### DELETE /api/documents/{id}

Hapus dokumen.

---

## API Chunking

### POST /api/chunking/process

Proses dokumen menjadi chunks.

**Request:**
```json
{
  "document_ids": ["doc_abc123"],
  "config": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "auto_index": true
}
```

**Response:**
```json
{
  "status": "success",
  "job_id": "job_xyz789",
  "documents_queued": 1
}
```

### GET /api/chunking/status/{job_id}

Cek status pemrosesan.

---

## API Knowledge Base

### GET /api/kb/stats

Statistik KB.

**Response:**
```json
{
  "status": "success",
  "total_documents": 25,
  "total_chunks": 450,
  "vector_store_status": "healthy",
  "storage_size_mb": 12.5
}
```

### POST /api/kb/reindex

Rebuild index.

### DELETE /api/kb/clear?confirm=true

Hapus semua data.

---

## API Health

### GET /api/health

Health check cepat.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0"
}
```

---

## Python API (rag-model)

### Class AcademicRAG

```python
from rag_model import AcademicRAG

# Inisialisasi
rag = AcademicRAG(
    research_mode=False,
    response_format="simple"
)

# Query
result = rag.query(
    question="Apa syarat beasiswa?",
    pipeline_type="advanced"
)

# Health check
health = rag.health_check()
```

### Class Konfigurasi

```python
from rag_model.core.config import (
    PipelineConfig,
    IndexConfig,
    LLMConfig
)

# Konfigurasi pipeline
config = PipelineConfig(
    pipeline_type="advanced",
    top_k=10,
    rerank_top_k=5
)
```
