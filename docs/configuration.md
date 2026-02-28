# Configuration Flags

## Core
- `GROQ_API_KEY`: Required for inference calls.
- `OPENAI_API_KEY`: Optional for embedding fallback.
- `SARVAM_API_KEY`: Optional for verifier independence.

## CORS
- `CORS_ALLOW_ORIGINS`: Comma-separated list of allowed origins.
- `CORS_ALLOW_CREDENTIALS`: `true|false`. If true, origins must be explicit.

## Observability
- `OTEL_ENABLED`: `true|false` to enable OpenTelemetry instrumentation.

## Model Cache + Preload
- `MODEL_CACHE_DIR`: Path for HuggingFace/Transformers/SentenceTransformers cache.
- `PRELOAD_MODELS`: `true|false` to preload local models on startup.

## Vision
- `VISION_BACKEND`: `blip` (default) or `llava`.
- `VISION_MODEL_NAME`: Model id for the selected backend.
- `VISION_FALLBACK_MODEL`: Fallback caption model.
- `VISION_ALLOW_FALLBACK`: `true|false` to allow fallback on load errors.

## OCR (EasyOCR)
- `OCR_ENGINE`: Set to `easyocr` (default).
- `OCR_LANG`: OCR language (default `en`).
- `PDF_OCR_FALLBACK`: `true|false` to run OCR when PDFs have little/no extractable text.
- `PDF_OCR_MIN_CHARS`: Minimum extracted characters before OCR fallback triggers.

## Retrieval
- `QDRANT_URL`, `QDRANT_API_KEY`: Enable Qdrant Cloud.
- `QDRANT_COLLECTION`: Override collection name.
- `QDRANT_MULTI_TENANT`: `true|false` to use `tenant_id` filter.
- `HYBRID_SEARCH`: `true|false` to enable lexical rerank.
- `RERANKER_ENABLED`: `true|false` to enable cross-encoder reranking.
- `RERANKER_MODEL_NAME`: Override reranker model.
- `RERANK_TOP_K`: Override final top-k after rerank.

## Background Jobs
- `CELERY_ENABLED`: `true|false` to use Celery workers for ingestion/crawl.
- `CELERY_BROKER_URL`: Redis/AMQP broker URL.
- `CELERY_RESULT_BACKEND`: Redis/AMQP backend URL.

## Ephemeral Sessions
- `EPHEMERAL_TTL_HOURS`: Session file collection TTL.
- `EPHEMERAL_CLEANUP_INTERVAL_MINUTES`: Background cleanup interval.
- `MAX_UPLOAD_MB`: Maximum upload size in MB.

## Server
- `GUNICORN_WORKERS`: Worker count for Gunicorn.
- `GUNICORN_BIND`: Bind address.
- `GUNICORN_TIMEOUT`: Request timeout.
