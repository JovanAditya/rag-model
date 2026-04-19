@echo off
setlocal enabledelayedexpansion
REM Academic RAG - Windows Installation Script
REM Supports both CUDA (GPU) and CPU-only installation

echo ========================================
echo Academic RAG - Windows Installation
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/6] Checking conda installation...
call conda --version
echo.

REM Ask user for GPU or CPU
echo ========================================
echo Select Installation Type:
echo ========================================
echo.
echo   [1] CUDA (GPU) - Recommended if you have NVIDIA GPU
echo   [2] CPU Only   - For computers without NVIDIA GPU
echo.
choice /c 12 /n /m "Enter your choice (1 or 2): "
set INSTALL_TYPE=%ERRORLEVEL%

if %INSTALL_TYPE%==1 (
    set INSTALL_NAME=CUDA GPU
    echo.
    echo Selected: CUDA installation
) else (
    set INSTALL_NAME=CPU Only
    echo.
    echo Selected: CPU-only installation
)
echo.

REM Deactivate current environment first
echo [2/6] Preparing environment...
call conda deactivate 2>nul

REM Check if environment exists and remove it
call conda env list | findstr /C:"academic-rag" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Found existing academic-rag environment. Removing...
    call conda env remove -n academic-rag -y
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Could not remove existing environment.
        echo Please close all Python processes and try again.
        pause
        exit /b 1
    )
) else (
    echo No existing environment found.
)
echo.

REM Create conda environment
echo [3/6] Creating conda environment...
call conda create -n academic-rag python=3.11 -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)
echo.

REM Activate environment
echo [4/6] Activating environment...
call conda activate academic-rag
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate environment
    pause
    exit /b 1
)
echo.

REM Install PyTorch
echo [5/6] Installing packages (%INSTALL_NAME%)...
echo.

echo Installing PyTorch (%INSTALL_NAME%)...
if %INSTALL_TYPE%==1 (
    echo Installing CUDA version...
    pip install "torch>=2.9.1" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu128
) else (
    echo Installing CPU version...
    pip install "torch>=2.9.1" "torchvision" "torchaudio"
)

echo Installing ML packages...
pip install "transformers>=4.57.1"
pip install "sentence-transformers>=5.1.2"
pip install "sentencepiece>=0.2.1"
pip install "chromadb>=1.3.4"
pip install "scikit-learn>=1.7.2"

echo Installing PDF processing...
pip install "pypdf>=6.2.0" "pdfplumber>=0.11.8" "pillow>=12.0.0"

echo Installing LLM providers...
pip install "requests>=2.32.5" "google-genai>=1.0.0"

echo Installing API dependencies...
pip install "fastapi>=0.104.0" "uvicorn>=0.24.0" "pydantic>=2.0.3" "python-multipart>=0.0.6"

echo Installing utilities...
pip install "rich>=14.2.0" "python-dotenv>=0.8.2" "numpy>=2.3.4" "pandas>=2.3.3"
pip install "tenacity>=9.1.2" "httpx>=0.23.1" "pyyaml>=6.0.3" "orjson>=3.11.4" "cachetools>=6.2.2"

echo Installing visualization...
pip install "matplotlib>=3.10.8" "seaborn>=0.11.1" "plotly>=6.4.0"

echo Installing evaluation packages...
pip install "accelerate>=1.12.0" "datasets>=4.4.1" "evaluate>=0.4.6" "ragas>=0.4.0" "statsmodels>=0.14.5"

echo.

REM Verify installation
echo [6/6] Verifying installation...
echo.

python -c "import torch; print('OK: PyTorch', torch.__version__); print('    CUDA available:', torch.cuda.is_available())"
python -c "import sentence_transformers; print('OK: sentence-transformers', sentence_transformers.__version__)"
python -c "import chromadb; print('OK: chromadb', chromadb.__version__)"
python -c "import sklearn; print('OK: scikit-learn', sklearn.__version__)"
python -c "from rag_model import AcademicRAG; print('OK: AcademicRAG import successful')"

echo.
echo ========================================
echo Installation completed successfully!
echo Installation type: %INSTALL_NAME%
echo ========================================
echo.
echo To activate the environment:
echo     conda activate academic-rag
echo.
echo To test the system:
echo     python examples/basic_usage.py
echo.
echo To start the API:
echo     cd ../academic-api
echo     uvicorn api.main:app --reload --port 8000
echo.

pause
