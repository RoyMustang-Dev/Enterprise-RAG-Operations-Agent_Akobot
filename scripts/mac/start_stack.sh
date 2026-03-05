#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$DIR")"

if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
fi

echo "[STACK] Starting API on port 8000..."
bash "$DIR/start_api.sh" &
API_PID=$!

if [ "$CELERY_AUTOSTART" = "true" ]; then
  echo "[STACK] Starting Celery worker..."
  bash "$DIR/start_celery_worker.sh" &
  CELERY_PID=$!
else
  echo "[STACK] CELERY_AUTOSTART is false; not starting worker."
fi

# Wait for both background processes
wait $API_PID
if [ -n "$CELERY_PID" ]; then
  wait $CELERY_PID
fi
