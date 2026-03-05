#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$DIR")"

echo "[API] Starting FastAPI..."
cd "$REPO_ROOT"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONEXECUTABLE=$(which python)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
