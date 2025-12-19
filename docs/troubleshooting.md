# Pemecahan Masalah

Masalah umum dan solusinya untuk RAG Model.

## Masalah Instalasi

### "Conda not found"

Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

### "Cannot remove current environment"

Deaktivasi dulu:
```bash
conda deactivate
./install.sh  # atau install_windows.bat
```

### PyTorch DLL Error (Windows)

Reinstall via conda:
```bash
conda activate academic-rag
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### OpenMP Error

Set environment variable:
```bash
# Windows
set KMP_DUPLICATE_LIB_OK=TRUE

# Linux/Mac
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Error Import

### "ModuleNotFoundError: rag_model"

Pastikan environment sudah aktif:
```bash
conda activate academic-rag
```

### "No module named sentence_transformers"

Install package yang hilang:
```bash
pip install sentence-transformers
```

---

## Masalah Runtime

### Hasil Kosong

Proses dan index dokumen dulu:
```bash
python scripts/prepare_data.py -i ./data/documents -o ./data/processed
python scripts/build_indexes.py -d ./data/processed/chunks.json --verify
```

### LLM Tidak Merespon

Cek provider LLM:
```bash
# Gemini - cek API key di .env
cat .env | grep GEMINI

# Ollama
ollama serve
ollama list
```

### CUDA Tidak Tersedia

Cek PyTorch CUDA:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Jika False, reinstall PyTorch dengan CUDA.

---

## Masalah API (rag-api)

### API Tidak Bisa Start

Cek port 5001:
```bash
cd rag-api
python api/app.py
# Seharusnya start di http://localhost:5001
```

### Error CORS

API sudah enable CORS untuk semua origin secara default.

### Upload Gagal

Cek ukuran file (max 50MB).

---

## Masalah Performa

### Query Lambat

- Gunakan pipeline `baseline` untuk kecepatan
- Kurangi `max_results`
- Cek apakah CUDA aktif

### Penggunaan Memori Tinggi

- Kurangi jumlah chunk
- Gunakan embedding model yang lebih kecil
- Tutup aplikasi lain

---

## Mendapatkan Bantuan

1. Cek panduan troubleshooting ini
2. Review pesan error dengan teliti
3. Cek konfigurasi `.env`
4. Verifikasi semua dependencies terinstall
5. Lihat [POLYREPO_SETUP_GUIDE.md](../../POLYREPO_SETUP_GUIDE.md) untuk masalah setup
