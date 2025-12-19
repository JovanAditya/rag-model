# Persiapan Data

Cara menyiapkan dokumen untuk RAG Model.

## Format yang Didukung

| Format | Ekstensi | Catatan |
|--------|----------|---------|
| PDF | `.pdf` | Support terbaik |
| Word | `.doc`, `.docx` | Support baik |
| Teks | `.txt` | Plain text |

## Alur Kerja

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PDF/DOC    │────▶│prepare_data │────▶│ chunks.json │
│  dokumen    │     │  (chunk)    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐     ┌─────────────┐
                                        │build_indexes│────▶│chroma_db +  │
                                        │  (index)    │     │ BM25 cache  │
                                        └─────────────┘     └─────────────┘
```

## Langkah 1: Tambah Dokumen

Salin file ke `data/documents/`:

```
rag-model/
└── data/
    └── documents/
        ├── handbook.pdf
        ├── peraturan.docx
        └── panduan.txt
```

## Langkah 2: Chunk Dokumen

Ekstrak dan chunk dokumen:

```bash
cd rag-model

# Penggunaan dasar
python scripts/prepare_data.py \
    --input ./data/documents \
    --output ./data/processed

# Dengan pengaturan khusus
python scripts/prepare_data.py \
    --input ./data/documents \
    --output ./data/processed \
    --chunk-size 500 \
    --chunk-overlap 100 \
    --device cuda
```

### Opsi

| Opsi | Deskripsi | Default |
|------|-----------|---------|
| `--input`, `-i` | Folder input dengan PDF | **Wajib** |
| `--output`, `-o` | Folder output untuk chunks | **Wajib** |
| `--chunk-size` | Karakter per chunk | `1000` |
| `--chunk-overlap` | Overlap antar chunk | `200` |
| `--device` | cpu/cuda/auto | `auto` |

### Output

Menghasilkan `chunks.json` dengan struktur:
```json
[
  {
    "text": "Konten chunk...",
    "metadata": {
      "source": "handbook.pdf",
      "page": 1,
      "chunk_id": 0
    }
  }
]
```

## Langkah 3: Build Index

Buat searchable index:

```bash
python scripts/build_indexes.py \
    --documents ./data/processed/chunks.json \
    --chroma-dir ./data/chroma_db \
    --cache-dir ./data/bm25_cache \
    --verify
```

### Opsi

| Opsi | Deskripsi | Default |
|------|-----------|---------|
| `--documents`, `-d` | Path ke chunks.json | `./data/processed/chunks.json` |
| `--chroma-dir` | Output ChromaDB | `./data/chroma_db` |
| `--cache-dir` | BM25 cache | `./data/bm25_cache` |
| `--verify` | Verifikasi setelah build | `false` |
| `--force` | Paksa rebuild | `false` |

## Via rag-api (Alternatif)

Jika rag-api sudah berjalan di port 5001:

```bash
# Upload dokumen
curl -X POST "http://localhost:5001/api/documents/upload" \
     -F "file=@dokumen.pdf"

# Proses (chunk + index)
curl -X POST "http://localhost:5001/api/chunking/process" \
     -H "Content-Type: application/json" \
     -d '{"document_ids": ["doc_abc123"], "auto_index": true}'
```

## Struktur Data Setelah Proses

```
rag-model/
└── data/
    ├── documents/           # File asli
    │   └── handbook.pdf
    ├── processed/           # Chunks
    │   └── chunks.json
    ├── chroma_db/          # Vector database
    │   └── (file ChromaDB)
    └── bm25_cache/         # BM25 cache
        └── (file cache)
```

## Rebuild Index

Jika dokumen berubah:

```bash
# Re-chunk (jika PDF berubah)
python scripts/prepare_data.py -i ./data/documents -o ./data/processed

# Re-index
python scripts/build_indexes.py -d ./data/processed/chunks.json \
    --chroma-dir ./data/chroma_db --force --verify

# Atau via rag-api
curl -X POST "http://localhost:5001/api/kb/reindex"
```

## Best Practices

1. **PDF Bersih** - Pastikan PDF memiliki teks yang bisa diseleksi
2. **Ukuran Chunk** - 500-1000 karakter bekerja dengan baik
3. **Overlap** - 10-20% dari ukuran chunk
4. **Verifikasi** - Selalu gunakan flag `--verify`
5. **Backup** - Simpan salinan dokumen asli
