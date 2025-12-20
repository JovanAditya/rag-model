# Panduan Setup

Panduan instalasi untuk RAG Model.

## Prasyarat

- **Python 3.11+**
- **Conda** (Miniconda atau Anaconda)
- **NVIDIA GPU** (opsional, untuk CUDA)

## Instalasi Cepat

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

## Instalasi Manual

### Langkah 1: Buat Environment (RECOMMENDED)

Gunakan `environment.yml` untuk instalasi lengkap:

```bash
conda env create -f environment.yml
conda activate academic-rag
```

### Langkah 2: (Opsional) Install PyTorch dengan GPU

Jika ingin menggunakan GPU CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Alternatif: Instalasi Manual Step-by-Step

Jika `environment.yml` gagal, ikuti langkah berikut:

**1. Buat environment kosong:**
```bash
conda create -n academic-rag python=3.11 -y
conda activate academic-rag
```

**2. Install PyTorch:**

- **CUDA (GPU):**
  ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
  ```

- **CPU Only:**
  ```bash
  conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
  ```

- **Mac (MPS):**
  ```bash
  conda install pytorch torchvision torchaudio -c pytorch -y
  ```

**3. Install Packages via requirements.txt:**
```bash
pip install -r requirements.txt
```

Atau manual:
```bash
pip install sentence-transformers chromadb transformers
pip install flask pydantic
pip install pypdf pdfplumber python-docx
pip install rich python-dotenv google-generativeai
```

## Konfigurasi LLM

Buat file `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
```

Lihat [configuration.md](configuration.md) untuk opsi lengkap.

## Verifikasi Instalasi

```bash
conda activate academic-rag
python -c "from rag_model import AcademicRAG; print('OK')"
```

## Build Index

Jika sudah ada dokumen di `data/documents/`:

```bash
python scripts/build_indexes.py --documents ./data/processed/chunks.json --verify
```

## Struktur Polyrepo

| Repository | Port | Deskripsi |
|------------|------|-----------|
| `rag-model` | - | Core RAG (repo ini) |
| `rag-api` | 5001 | REST API |
| `rag-web` | 8000 | Laravel frontend |

## Troubleshooting

Lihat [troubleshooting.md](troubleshooting.md) untuk masalah umum.
