#!/usr/bin/env bash
set -euo pipefail

echo "[CELERY][PROD] Starting worker (prefork)..."

# Load env if present
if [[ -f ".env" ]]; then
  export $(grep -v '^#' .env | xargs)
fi

CELERY_CONCURRENCY="${CELERY_CONCURRENCY:-4}"

celery -A app.infra.celery_app.celery_app worker \
  --loglevel=info \
  --concurrency="$CELERY_CONCURRENCY"

