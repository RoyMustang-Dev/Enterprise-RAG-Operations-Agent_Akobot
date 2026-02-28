# Production Environment Guide

This guide explains how to configure `.env` for production. Use `.env-copy` as a template.

## 1. Secrets & Keys
- Store API keys in a secret manager (Vault, AWS Secrets Manager, GCP Secret Manager).
- Never commit real keys to git.

## 2. Qdrant (Vector Store)
- For production, set `QDRANT_URL` and `QDRANT_API_KEY`.
- Leave these empty for local storage.

## 3. Model Cache
- Set `MODEL_CACHE_DIR` to a persistent volume so models are reused across restarts.
- Example: `/var/cache/enterprise-rag` or `D:\model-cache`.

## 4. Preload Models
- `PRELOAD_MODELS=true` downloads models on boot, reducing first-request latency.
- Use this for production services.

## 5. OCR Runtime (Paddle)
- If you want auto-install, set:
  - `PADDLE_AUTO_INSTALL=true`
  - `PADDLE_PIP_SPEC_GPU=paddlepaddle-gpu`
  - `PADDLE_PIP_SPEC_CPU=paddlepaddle`
- Recommended: bake the correct paddle wheel into your image instead of runtime install.

## 6. Background Jobs (Celery)
- Set:
  - `CELERY_ENABLED=true`
  - `CELERY_BROKER_URL=redis://:password@redis-host:6379/0`
  - `CELERY_RESULT_BACKEND=redis://:password@redis-host:6379/0`
- Run a separate worker process:
  - `celery -A app.infra.celery_app.celery_app worker --loglevel=info`

## 7. Security & CORS
- Use explicit origins if `CORS_ALLOW_CREDENTIALS=true`.
- Example: `CORS_ALLOW_ORIGINS=https://app.example.com`

## 8. Ephemeral Session TTL
- `EPHEMERAL_TTL_HOURS=24` is recommended for uploaded files.
- Increase only if you need longer retention.
