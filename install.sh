#!/bin/bash
# Academic RAG - Linux/Mac Installation Script
# Supports both CUDA (GPU) and CPU-only installation

set -e

echo "========================================"
echo "Academic RAG - Linux/Mac Installation"
echo "========================================"
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[1/6] Checking conda installation..."
conda --version
echo

# Detect OS
OS_TYPE="linux"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="mac"
fi

# Ask user for GPU or CPU
echo "========================================"
echo "Select Installation Type:"
echo "========================================"
echo
if [[ "$OS_TYPE" == "mac" ]]; then
    echo "  [1] MPS (Mac GPU) - For Apple Silicon Macs"
    echo "  [2] CPU Only      - For Intel Macs or if MPS not needed"
else
    echo "  [1] CUDA (GPU) - For NVIDIA GPU systems"
    echo "  [2] CPU Only   - For systems without NVIDIA GPU"
fi
echo
read -p "Enter your choice (1 or 2): " INSTALL_TYPE

case $INSTALL_TYPE in
    1)
        if [[ "$OS_TYPE" == "mac" ]]; then
            INSTALL_NAME="MPS (Mac GPU)"
        else
            INSTALL_NAME="CUDA (GPU)"
        fi
        echo "Selected: $INSTALL_NAME installation"
        ;;
    *)
        INSTALL_TYPE=2
        INSTALL_NAME="CPU Only"
        echo "Selected: CPU-only installation"
        ;;
esac
echo

# Deactivate current environment
echo "[2/6] Preparing environment..."
conda deactivate 2>/dev/null || true

# Check if environment exists
if conda env list | grep -q "academic-rag"; then
    echo "Found existing academic-rag environment. Removing..."
    conda env remove -n academic-rag -y
else
    echo "No existing environment found."
fi
echo

# Create conda environment
echo "[3/6] Creating conda environment..."
conda create -n academic-rag python=3.11 -y
echo

# Activate environment
echo "[4/6] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate academic-rag
echo

# Install packages
echo "[5/6] Installing packages ($INSTALL_NAME)..."
echo

echo "Installing PyTorch ($INSTALL_NAME)..."
if [[ "$OS_TYPE" == "mac" ]]; then
    # Mac: pip install torch torchvision (auto detects MPS)
    pip install "torch>=2.9.1" "torchvision" "torchaudio"
elif [[ "$INSTALL_TYPE" == "1" ]]; then
    # Linux CUDA: pip install torch torchvision (defaults to CUDA 12.8)
    pip install "torch>=2.9.1" "torchvision" "torchaudio"
else
    # Linux CPU
    pip install "torch>=2.9.1" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing ML packages..."
pip install "transformers>=4.57.1"
pip install "sentence-transformers>=5.1.2"
pip install "sentencepiece>=0.2.1"
pip install "chromadb>=1.3.4"
pip install "scikit-learn>=1.7.2"

echo "Installing PDF processing..."
pip install "pypdf>=6.2.0" "pymupdf4llm>=1.27.0" "pillow>=12.0.0"

echo "Installing LLM providers..."
pip install "requests>=2.32.5" "google-genai>=1.0.0"

echo "Installing API dependencies..."
pip install "fastapi>=0.104.0" "uvicorn>=0.24.0" "pydantic>=2.0.3" "python-multipart>=0.0.6"

echo "Installing utilities..."
pip install "rich>=14.2.0" "python-dotenv>=0.8.2" "numpy>=2.3.4" "pandas>=2.3.3"
pip install "tenacity>=9.1.2" "httpx>=0.23.1" "pyyaml>=6.0.3" "orjson>=3.11.4" "cachetools>=6.2.2"

echo "Installing visualization..."
pip install "matplotlib>=3.10.8" "seaborn>=0.11.1" "plotly>=6.4.0"

echo "Installing evaluation packages..."
pip install "accelerate>=1.12.0" "datasets>=4.4.1" "evaluate>=0.4.6" "ragas>=0.4.0" "statsmodels>=0.14.5"

echo

# Verify installation
echo "[6/6] Verifying installation..."
echo

python -c "import torch; print('OK: PyTorch', torch.__version__); print('    CUDA available:', torch.cuda.is_available()); print('    MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
python -c "import sentence_transformers; print('OK: sentence-transformers', sentence_transformers.__version__)"
python -c "import chromadb; print('OK: chromadb', chromadb.__version__)"
python -c "import sklearn; print('OK: scikit-learn', sklearn.__version__)"
python -c "from rag_model import AcademicRAG; print('OK: AcademicRAG import successful')"

if [ $? -eq 0 ]; then
    echo
    echo "========================================"
    echo "Installation completed successfully!"
    echo "Installation type: $INSTALL_NAME"
    echo "========================================"
    echo
    echo "To activate the environment:"
    echo "    conda activate academic-rag"
    echo
    echo "To test the system:"
    echo "    python examples/basic_usage.py"
    echo
    echo "To start the API:"
    echo "    cd ../academic-api"
    echo "    uvicorn api.main:app --reload --port 8000"
    echo
else
    echo
    echo "========================================"
    echo "Installation completed with warnings"
    echo "========================================"
    echo
    echo "Some packages may need manual fixing."
    echo "See README.md for troubleshooting."
    echo
fi
