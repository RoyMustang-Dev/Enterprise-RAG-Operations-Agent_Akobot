#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$DIR")"

echo "[BOOTSTRAP] Repo: $REPO_ROOT"
cd "$REPO_ROOT"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi
PY=$(which python)
echo "[BOOTSTRAP] Using Python: $PY"

if [ -f ".env" ]; then
    echo "[BOOTSTRAP] Loading .env variables..."
    export $(grep -v '^#' .env | xargs)
fi

echo "[BOOTSTRAP] Installing requirements..."
$PY -m pip install -r requirements.txt

test_import() {
    local module=$1
    local label=$2
    if $PY -c "import $module" &> /dev/null; then
        echo "[BOOTSTRAP] ${label}: OK"
        return 0
    else
        echo "[BOOTSTRAP] ${label}: FAIL"
        return 1
    fi
}

HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
    fi
fi

if [ "$HAS_GPU" = true ]; then
    echo "[BOOTSTRAP] NVIDIA GPU Detected via nvidia-smi."
else
    echo "[BOOTSTRAP] No NVIDIA GPU detected via nvidia-smi."
fi

NEEDS_REINSTALL=false

if $PY -c "import torch" &> /dev/null; then
    CUDA_AVAILABLE=$($PY -c "import torch; print(str(torch.cuda.is_available()).lower())")
    if [ "$CUDA_AVAILABLE" = "true" ] && [ "$HAS_GPU" = true ]; then
        echo "[BOOTSTRAP] PyTorch CUDA integration check PASSED."
    elif [ "$CUDA_AVAILABLE" = "false" ] && [ "$HAS_GPU" = true ]; then
        echo "[BOOTSTRAP] GPU is present but installed PyTorch cannot use CUDA. Needs reinstall."
        NEEDS_REINSTALL=true
    else
        echo "[BOOTSTRAP] PyTorch CPU/MPS load ok."
    fi
else
    echo "[BOOTSTRAP] PyTorch native dependencies failed to load. Needs reinstall."
    NEEDS_REINSTALL=true
fi

if [ "$NEEDS_REINSTALL" = true ]; then
    if [ "$AUTO_FIX_SYSTEM_DEPS" = "true" ]; then
        echo "[BOOTSTRAP] AUTO_FIX_SYSTEM_DEPS enabled. Rebuilding PyTorch environment cleanly..."
        $PY -m pip uninstall -y torch torchvision torchaudio > /dev/null 2>&1
        
        if [ "$HAS_GPU" = true ]; then
            echo "[BOOTSTRAP] Installing clean PyTorch stack (cu121)..."
            $PY -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
        else
            echo "[BOOTSTRAP] Installing clean PyTorch stack (cpu)..."
            $PY -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        echo "[BOOTSTRAP] Fix: reinstall torch with correct index to match your architecture."
    fi
fi

OK=true
test_import "torch" "PyTorch" || OK=false
test_import "transformers" "Transformers" || OK=false
test_import "sentence_transformers" "SentenceTransformers" || OK=false
test_import "paddleocr" "PaddleOCR" || OK=false
test_import "qdrant_client" "Qdrant Client" || OK=false

echo "[BOOTSTRAP] Ensuring Paddle runtime (auto-install if enabled)..."
$PY -c "from app.infra.model_bootstrap import ensure_paddle_runtime; ensure_paddle_runtime()" || true

echo "[BOOTSTRAP] Pre-downloading AI models (if enabled) to prevent API latency later..."
$PY -c "from app.infra.model_bootstrap import preload_models; preload_models()" || true

if [ "$OK" = true ]; then
    echo "Success Everything is Up & Running - Start using the Product now!!"
    exit 0
else
    echo "[BOOTSTRAP] One or more checks failed. See logs above."
    exit 1
fi
