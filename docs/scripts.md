# Referensi Script

Script utility untuk RAG Model. Semua script membaca default dari file `.env`.

## Konfigurasi

Script otomatis memuat pengaturan dari `.env`. Bisa dijalankan tanpa argumen:

```bash
python scripts/prepare_data.py      # Menggunakan DOCUMENTS_DIR, PROCESSED_DIR dari .env
python scripts/build_indexes.py     # Menggunakan CHROMA_DIR, BM25_CACHE_DIR dari .env
```

Atau override dengan argumen CLI.

---

## Pipeline Utama

### prepare_data.py

Ekstrak dan chunk dokumen dari PDF.

```bash
# Menggunakan default .env (rekomendasi)
python scripts/prepare_data.py

# Dengan path eksplisit
python scripts/prepare_data.py \
    --input ./data/documents \
    --output ./data/processed

# Semua opsi
python scripts/prepare_data.py \
    --chunk-size 1000 \
    --chunk-overlap 200 \
    --device cuda \
    --skip-existing
```

| Opsi | Env Variable | Default |
|------|--------------|---------|
| `--input`, `-i` | `DOCUMENTS_DIR` | `./data/documents` |
| `--output`, `-o` | `PROCESSED_DIR` | `./data/processed` |
| `--chunk-size` | `CHUNK_SIZE` | `1000` |
| `--chunk-overlap` | `CHUNK_OVERLAP` | `200` |
| `--device` | `EMBEDDING_DEVICE` | `auto` |
| `--skip-existing` | - | `false` |
| `--force` | - | `false` |

---

### build_indexes.py

Build index ChromaDB + BM25 dari chunks.

```bash
# Menggunakan default .env (rekomendasi)
python scripts/build_indexes.py --verify

# Dengan path eksplisit
python scripts/build_indexes.py \
    --documents ./data/processed/chunks.json \
    --chroma-dir ./data/chroma_db \
    --verify
```

| Opsi | Env Variable | Default |
|------|--------------|---------|
| `--documents`, `-d` | `PROCESSED_DIR` | `./data/processed/chunks.json` |
| `--chroma-dir` | `CHROMA_DIR` | `./data/chroma_db` |
| `--cache-dir` | `BM25_CACHE_DIR` | `./data/bm25_cache` |
| `--collection` | `COLLECTION_NAME` | `academic_docs` |
| `--verify` | - | `false` |
| `--force` | - | `false` |

---

## Script Utility

### verify_system.py

Verifikasi instalasi dan kesehatan sistem.

```bash
python scripts/verify_system.py
python scripts/verify_system.py --component environment
python scripts/verify_system.py --component data
```

---

### index_manager.py

Kelola index ChromaDB + BM25.

```bash
python scripts/index_manager.py --list
python scripts/index_manager.py --test academic_docs
python scripts/index_manager.py --cleanup academic_docs --confirm
```

---

### fix_text_preprocessing.py

Perbaiki masalah OCR/ekstraksi teks di chunks.

```bash
python scripts/fix_text_preprocessing.py --input ./data/processed/chunks.json --analyze
python scripts/fix_text_preprocessing.py --input ./data/processed/chunks.json --dry-run
```

---

## Alur Kerja

```bash
# 1. Setup .env (salin dan edit)
cp .env.example .env

# 2. Siapkan data (menggunakan .env)
python scripts/prepare_data.py

# 3. Build index (menggunakan .env)
python scripts/build_indexes.py --verify

# 4. Test
python examples/basic_usage.py
```

## Perintah Cepat

```bash
# Setup pertama kali (menggunakan default .env)
python scripts/prepare_data.py
python scripts/build_indexes.py --verify

# Paksa rebuild semua
python scripts/prepare_data.py --force
python scripts/build_indexes.py --force --verify

# Cek kesehatan
python scripts/verify_system.py
```
