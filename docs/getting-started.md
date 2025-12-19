# Memulai

Panduan langkah demi langkah untuk setup dan menggunakan RAG Model.

## Langkah 1: Instalasi

### Windows
```cmd
cd rag-model
install_windows.bat
```

### Linux/Mac
```bash
cd rag-model
chmod +x install.sh
./install.sh
```

Pilih **[1] CUDA** jika memiliki GPU NVIDIA, atau **[2] CPU** jika tidak.

---

## Langkah 2: Aktifkan Environment

```bash
conda activate academic-rag
```

---

## Langkah 3: Konfigurasi LLM

Buat file `.env` di folder `rag-model/`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
```

---

## Langkah 4: Siapkan Dokumen

Letakkan file PDF/DOC di folder `data/documents/`:

```
rag-model/
└── data/
    └── documents/
        ├── handbook.pdf
        ├── peraturan.pdf
        └── panduan.docx
```

---

## Langkah 5: Proses Dokumen

### Chunk Dokumen
```bash
python scripts/prepare_data.py \
    --input ./data/documents \
    --output ./data/processed
```

### Build Index
```bash
python scripts/build_indexes.py \
    --documents ./data/processed/chunks.json \
    --verify
```

---

## Langkah 6: Test Query

```python
from rag_model.core.pipeline import AdvancedAcademicRAG
from rag_model.core.config import RAGConfig

config = RAGConfig()
rag = AdvancedAcademicRAG(config)

result = rag.query("Apa visi Universitas Mercu Buana?")
print(result['answer'])
```

---

## Langkah Selanjutnya

- [Overview](overview.md) - Arsitektur sistem
- [Configuration](configuration.md) - Opsi konfigurasi lengkap
- [Usage](usage.md) - Contoh penggunaan lanjutan
