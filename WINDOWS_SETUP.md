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

If the script fails, install manually:

```cmd
REM Create environment
conda create -n academic-rag python=3.11 -y
conda activate academic-rag

REM Install PyTorch (choose one)
REM For CUDA (Windows):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
REM For CPU (Windows):
pip install torch torchvision torchaudio

REM Install other packages
pip install sentence-transformers chromadb transformers
pip install fastapi uvicorn pydantic
pip install pypdf pdfplumber pillow
pip install rich python-dotenv
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
