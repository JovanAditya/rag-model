# Panduan Penggunaan

Cara menggunakan RAG Model secara efektif.

## Penggunaan Dasar

```python
from rag_model import AcademicRAG

# Inisialisasi
rag = AcademicRAG()

# Query sederhana
result = rag.query("Apa syarat beasiswa prestasi?")
print(result['answer'])
```

## Tipe Pipeline

### Advanced (Default)
```python
result = rag.query(
    "Apa syarat beasiswa?",
    pipeline_type="advanced"
)
```
- Hybrid search (Vector + BM25)
- Cross-encoder reranking
- Akurasi lebih tinggi, lebih lambat

### Baseline
```python
result = rag.query(
    "Apa syarat beasiswa?",
    pipeline_type="baseline"
)
```
- Hanya vector search
- Lebih cepat, kurang akurat

## Format Response

```python
result = rag.query("Apa visi UMB?")

# Field yang tersedia
print(result['answer'])      # Jawaban yang dihasilkan
print(result['confidence'])  # Skor kepercayaan (0-1)
print(result['sources'])     # Dokumen sumber
print(result['metadata'])    # Metadata query
```

## Mode Research

Untuk metrik detail:

```python
rag = AcademicRAG(research_mode=True)
result = rag.query("Apa visi UMB?")

print(result['metadata']['retrieval_time'])
print(result['metadata']['generation_time'])
```

## Health Check

```python
health = rag.health_check()
print(health)
# {'ready': True, 'vector_store': 'ok', 'bm25_index': 'ok'}
```

## Penggunaan via API (rag-api)

### Jalankan Server
```bash
cd rag-api
python api/app.py
# Berjalan di http://localhost:5001
```

### Query via API
```bash
curl -X POST "http://localhost:5001/api/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Apa syarat beasiswa?"}'
```

### Upload Dokumen
```bash
curl -X POST "http://localhost:5001/api/documents/upload" \
     -F "file=@dokumen.pdf"
```

### Proses Dokumen
```bash
curl -X POST "http://localhost:5001/api/chunking/process" \
     -H "Content-Type: application/json" \
     -d '{"document_ids": ["doc_abc123"]}'
```

### Statistik KB
```bash
curl "http://localhost:5001/api/kb/stats"
```

## Contoh

Lihat folder `examples/`:

| File | Deskripsi |
|------|-----------|
| `basic_usage.py` | Contoh penggunaan dasar |
| `api_integration.py` | Integrasi API |
| `evaluation_usage.py` | Panduan evaluasi |

## Best Practices

1. **Gunakan Advanced Pipeline** untuk query penting
2. **Cek Confidence** sebelum menggunakan jawaban
3. **Review Sources** untuk memverifikasi akurasi
4. **Proses Dokumen** setelah upload untuk indexing
