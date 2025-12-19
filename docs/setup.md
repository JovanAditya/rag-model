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

### Langkah 1: Buat Environment

```bash
conda create -n academic-rag python=3.11 -y
conda activate academic-rag
```

### Langkah 2: Install PyTorch

**CUDA (GPU):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

**CPU Only:**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**Mac (MPS):**
```bash
conda install pytorch torchvision torchaudio -c pytorch -y
```

### Langkah 3: Install Packages

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
