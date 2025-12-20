# Windows Setup Guide

Complete installation guide for Windows users.

## Prerequisites

- **Windows 10/11**
- **Miniconda or Anaconda** - [Download](https://docs.conda.io/en/latest/miniconda.html)
- **NVIDIA GPU** (optional, for CUDA support)

## Quick Installation

1. Open **Command Prompt** or **PowerShell**
2. Navigate to the project:
   ```cmd
   cd path\to\rag-model
   ```
3. Run the installer:
   ```cmd
   install_windows.bat
   ```
4. Choose installation type:
   - **[1] CUDA** - If you have NVIDIA GPU
   - **[2] CPU** - If no GPU available

## Manual Installation

Jika script installer gagal, install secara manual:

### Opsi 1: Menggunakan environment.yml (RECOMMENDED)

```cmd
REM Install dari environment.yml
conda env create -f environment.yml
conda activate academic-rag

REM (Opsional) Install PyTorch dengan GPU CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Opsi 2: Instalasi Step-by-Step

Jika `environment.yml` gagal:

```cmd
REM Create environment
conda create -n academic-rag python=3.11 -y
conda activate academic-rag

REM Install PyTorch (pilih salah satu)
REM Untuk CUDA (Windows):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
REM Untuk CPU (Windows):
pip install torch torchvision torchaudio

REM Install packages lain via requirements.txt
pip install -r requirements.txt
```

## Verify Installation

```cmd
conda activate academic-rag
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from rag_model import AcademicRAG; print('OK')"
```

## Common Issues

### "Cannot remove current environment"

Deactivate first:
```cmd
conda deactivate
install_windows.bat
```

### PyTorch DLL Error

Reinstall PyTorch via conda:
```cmd
conda activate academic-rag
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### OpenMP Error

Set environment variable:
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
```

### Import Error

Make sure environment is activated:
```cmd
conda activate academic-rag
```

## GPU Support

Check if CUDA is working:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## Next Steps

1. **Configure LLM** - See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
2. **Test the model** - Run `python examples/basic_usage.py`
3. **Start API** - Run `cd ../rag-api && uvicorn api.main:app --reload --port 5001`
