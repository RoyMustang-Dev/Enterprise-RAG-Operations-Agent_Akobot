#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$DIR")"

echo "[CELERY] Starting worker..."
cd "$REPO_ROOT"

# Load .env variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONEXECUTABLE=$(which python)
python -m celery -A app.infra.celery_app.celery_app worker --loglevel=info --concurrency=1
