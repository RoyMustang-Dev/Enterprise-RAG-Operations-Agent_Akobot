#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$DIR")"

echo "[RESET] Stopping running API/Celery processes..."
pkill -f "uvicorn.*app.main:app" || true
pkill -f "celery.*app.infra.celery_app" || true
pkill -f "python.*$REPO_ROOT" || true

echo "[RESET] Removing venv and caches..."
rm -rf "$REPO_ROOT/venv"
rm -rf "$REPO_ROOT/model-cache"
rm -rf "$REPO_ROOT/data/qdrant_storage"
rm -rf "$REPO_ROOT/.pytest_cache"

find "$REPO_ROOT" -type d -name "__pycache__" -exec rm -rf {} + || true

echo "[RESET] Clearing pip cache..."
python3 -m pip cache purge > /dev/null 2>&1 || true

echo "[RESET] Recreating venv..."
cd "$REPO_ROOT"
python3 -m venv venv
source venv/bin/activate
PY=$(which python)

echo "[RESET] Installing requirements..."
$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt

echo "[RESET] Running bootstrap with auto-fix..."
export AUTO_FIX_SYSTEM_DEPS="true"
bash "$DIR/bootstrap_env.sh"

echo "[RESET] Starting stack..."
export CELERY_AUTOSTART="true"
bash "$DIR/start_stack.sh"
