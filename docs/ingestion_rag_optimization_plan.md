# Ingestion + RAG Latency Optimization Plan (No Architecture Changes)

## Phase 0: Baseline Harness (Start Here)

**Goal:** Establish reliable, repeatable baseline metrics for ingestion and RAG latency before any optimizations.

### 0.1 Baseline Inputs (Keep Stable)
- Use the same input files and URLs for every run.
- Keep current `.env` settings unchanged.
- Use the same hardware environment.

### 0.2 Baseline Metrics to Capture
From telemetry / response / logs:
- `latency_ms` (end-to-end)
- `retrieval_time_ms`
- `rerank_time_ms`
- `llm_time_ms`
- `tokens_input`
- `tokens_output`
- `confidence`
- `verifier_verdict`

From ingestion job logs:
- Total chunks generated
- Batch size used
- Time to embed batches
- Time to Qdrant insert per batch (if logged)

### 0.3 Baseline Procedure
Run each benchmark **3 times** and record **median**:
1. **Crawler benchmark**: `benchmark_crawler.py` (already standardized).
2. **Ingestion benchmark**:
   - Files: use the same file set for every run.
   - Capture total chunks and total time.
3. **RAG response benchmark**:
   - Use a fixed test query list (same session, same tenant).
   - Record latency fields from telemetry logs.

### 0.4 Baseline Storage
Record results in:
- `crawler_benchmark_results/benchmark_log.md`
- `docs/ingestion_rag_optimization_baseline.md` (to be created after first run)

### 0.5 Acceptance Gate (Before Any Changes)
Baseline is accepted if:
- 3 runs complete without errors.
- Median times are stable (variance < 20%).
- Logs show no missing telemetry fields.

---

## Phase 1: Ingestion Optimization (No Architecture Changes)
**Focus:** Batch sizing, indexing, and payload tuning without altering ingestion flow.

Planned actions:
1. Batch size tuning (controlled sweep).
2. Qdrant payload index audit (only for actively filtered fields).
3. HNSW config review (index-time vs query-time tradeoff).

**Gate:** Improved ingestion throughput with no regression in errors.

---

## Phase 2: Retrieval + Reranking Optimization (No Architecture Changes)
**Focus:** Reduce retrieval volume and reranker input size.

Planned actions:
1. Top-k reduction experiment.
2. Qdrant search param tuning (`hnsw_ef`).
3. Filter strictness audit.

**Gate:** Lower `retrieval_time_ms` and `rerank_time_ms` without quality loss.

---

## Phase 3: Synthesis + Routing Optimization (No Architecture Changes)
**Focus:** Reduce synthesis prompt size and avoid redundant calls.

Planned actions:
1. Prompt budget control for synthesis.
2. Early truncation of low-signal chunks.
3. Parallelize non-dependent calls (where already safe).

**Gate:** Lower `llm_time_ms` without hallucination increases.

---

## Phase 4: End-to-End Latency Strategy
**Focus:** Overlap non-critical work and enable caching.

Planned actions:
1. Cache reranked results for repeated queries.
2. Overlap retrieval + reasoning steps.

**Gate:** Lower `latency_ms` with no correctness regression.
