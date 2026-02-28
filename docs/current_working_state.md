# Current Working State Notes (2026-02-28)

This file is a snapshot of the **known-working implementation** so we can avoid breaking changes later.
It is intentionally concise and focused on what is currently in place.

## Active Features (Confirmed in Code)
- Unified `/chat` endpoint supports **JSON** and **multipart/form-data** (text + optional files + optional images).
- Multi-file session reuse (ephemeral session collections with TTL) for file uploads.
- OCR path + Vision path for images (`image_mode=auto|ocr|vision`).
- Streaming output for `/chat` (SSE when `stream=true`).
- Audio transcription endpoint (`/transcribe`) using faster-whisper.
- TTS endpoint (`/tts`) using Coqui TTS.
- Provider auto-routing when `PROVIDER_AUTO_ROUTING=true` and `model_provider=auto`.
- Reranker profiles: `auto|accurate|fast|off` with override `reranker_model_name`.
- Background cleanup timer for ephemeral collections.

## Key New Modules Added
- `app/multimodal/` (file parsing, session vectors, multimodal router, TTS, vision)
- `app/infra/model_bootstrap.py` (preload models, cache, paddle auto-install)
- `app/infra/provider_router.py` (provider auto-selection)
- `app/infra/celery_app.py`, `app/infra/celery_tasks.py`, `app/infra/job_tracker.py`
- `app/infra/qdrant_patch.py` (gRPC patch guard)

## Key File Modifications (High-Level)
- `app/api/routes.py`: unified chat, multipart handling, streaming, Swagger examples, feedback, ingestion, transcription, TTS.
- `app/supervisor/router.py`: provider routing hook + reranker profile handling.
- `app/retrieval/*`: reranker profile support + embeddings tweaks + Qdrant cloud support.
- `app/infra/hardware.py`: hardware probe + DLL path handling on Windows.
- `app/main.py`: startup hooks, background cleanup, CORS config.
- `frontend/app.py`: updated UI flow to match unified `/chat` and streaming.

## Operational Scripts (Windows)
- `scripts/bootstrap_env.ps1`: automated dependency repair + model runtime fixes.
- `scripts/start_api.ps1`: starts FastAPI using venv python with env load.
- `scripts/start_celery_worker.ps1`: starts Celery worker (Windows-safe pool).
- `scripts/start_stack.ps1`: starts API + Celery.
- `scripts/full_reset_and_bootstrap.ps1`: full reset + rebuild + start stack.

## Current Env Flags in Use (See `.env` / `.env-copy`)
- `MODEL_CACHE_DIR`, `PRELOAD_MODELS`, `PADDLE_AUTO_INSTALL`, `PADDLE_PIP_*`
- `VISION_MODEL_NAME`, `VISION_FALLBACK_MODEL`, `VISION_ALLOW_FALLBACK`
- `HYBRID_SEARCH`, `RERANKER_ENABLED`, `RERANKER_MODEL_NAME`, `RERANK_TOP_K`
- `CELERY_ENABLED`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `CELERY_AUTOSTART`
- `EPHEMERAL_TTL_HOURS`, `EPHEMERAL_CLEANUP_INTERVAL_MINUTES`, `MAX_UPLOAD_MB`

## Notes About Known-Working Behavior
- `/chat` supports multi-file uploads and session reuse across turns.
- Vision uses LLaVA by default with BLIP fallback.
- OCR uses Paddle-based pipeline (auto-installs runtime when enabled).
- Streaming responses are SSE on `/chat` when `stream=true`.

## Guardrails
- Avoid changing the above modules/flags unless we confirm tests still pass.
- If we change model/provider logic, update `docs/configuration.md` and this file.

