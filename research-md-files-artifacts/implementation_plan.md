# Root Cause Analysis & Implementation Plan

## 1. Goal Description
The current Enterprise RAG pipeline is failing at 4 critical junctures during chat execution. This plan addresses the root causes discovered during the codebase analysis and outlines the precise fixes required to stabilize the architecture without modifying the seamlessly working ingestion pipeline. 

It also includes **Part 2**, a dynamic model routing strategy to optimize both cost and latency during heavy concurrent loads.

## 2. Root Cause Analysis (RCA)

### A. Empty Chat History
- **Root Cause**: The API endpoint (`app/api/routes.py`) and the orchestration layer (`app/supervisor/router.py`) are entirely stateless. While `request.chat_history` is accepted, the system never appends the current user query and the generated assistant answer to it. 
- **Impact**: The LLM lacks contextual awareness of previous conversational turns.

### B. Metadata Extraction Failure (Reverting to semantic search)
- **Root Cause**: `app/retrieval/metadata_extractor.py` specifies `model_id = "qwen-2.5-32b"` and enforces `response_format={"type": "json_object"}`. Groq API strictly limits native JSON mode to specific Llama/GPT-OSS models. Using it on Qwen often triggers payload rejections, causing the code to catch the exception and bypass metadata filtering.
- **Impact**: No metadata bounds are applied to Qdrant searches.

### C. Reranker Returning Zero Chunks
- **Root Cause**: `app/retrieval/reranker.py` utilizes the HuggingFace `CrossEncoder`. The `.predict()` method natively outputs **un-normalized logits** (e.g., -5.2 to +6.1). The code arbitrarily checks `if score > 0.35:`, discarding perfectly valid semantic chunks because their logits weren't mathematically processed.
- **Impact**: The 70B generation engine receives `[]` (zero chunks).

### D. Query/Prompt Rewriter Not Working
- **Root Cause**: The MoE rewriter was crashing. We previously theorized `max_completion_tokens` was invalid; however, Groq documentation confirms `max_completion_tokens` is the **current standard** and `max_tokens` is deprecated. The true root cause was model incompatibility with JSON mode combined with a lack of defensive JSON parsing.
- **Impact**: The MoE fails to distill the prompts, defaulting to the failsafe raw query.

---

## 3. Proposed Changes (PART 1 - Core Stabilization)

### Target Optimized Architecture Stack
We will align all active components to this strict execution stack:

| Phase | Model |
| --- | --- |
| Guard | `llama-prompt-guard-2-86m` |
| Intent | `llama-3.1-8b-instant` |
| Prompt Rewriter | `openai/gpt-oss-120b` |
| Metadata Extractor | `llama-3.1-8b-instant` |
| Reranker | BAAI local |
| Core Brain | `llama-3.3-70b-versatile` |
| Verifier (L1) | `llama-3.1-8b-instant` |
| Verifier (L2 Escalation) | `openai/gpt-oss-120b` |
| RLHF Selection | `openai/gpt-oss-120b` |
| Code Agent | `qwen/qwen3-32b` |
| STT | `whisper-large-v3-turbo` |

### `app/api/routes.py` & `app/supervisor/router.py` (Stateless Fix)
- Modify the `ChatResponse` schema to include `chat_history: List[Dict[str, Any]]`.
- Update the `invoke` method in Router to append `{"role": "user", "content": query}` prior to execution, and `{"role": "assistant", "content": state["answer"]}` upon completion. The route will return this updated state.

### `app/retrieval/metadata_extractor.py` & `app/prompt_engine/rewriter.py` (JSON Stability Fix)
- Apply the updated models from the Target Stack table.
- Use strictly `max_completion_tokens` (do not mix with `max_tokens`).
- Add defensive JSON extraction logic directly beneath the API call:
```python
raw = response_json["choices"][0]["message"]["content"]
try:
    parsed = json.loads(raw)
except json.JSONDecodeError:
    # Explicit fallback extraction if the model hallucinates markdown backticks around the JSON
    parsed = extract_json_from_markdown(raw)
```

### `app/retrieval/reranker.py` (Logit Normalization Fix)
- Import `import math`.
- Apply a Sigmoid activation to convert logits into probabilities: `probability = 1 / (1 + math.exp(-score))`.
- Update the threshold condition to check `if probability > 0.35:`.

---

## 4. Proposed Changes (PART 2 - Enterprise Dynamic Routing Strategy)
*Note: This will be implemented ONLY after Part 1 is stabilized.*

### Goal
Implement a Reasoning Effort Controller to route standard queries to `llama-3.3-70b-versatile` and escalate complex reasoning queries to `openai/gpt-oss-120b`, saving compute costs and lowering latency under high concurrency.

### `app/core/types.py`
- Add `"reasoning_effort": str` ("low", "medium", "high") back to the `AgentState`.
- Add `"latency_optimizations": Dict` to return debugging metrics.

### `app/agents/rag.py` & `app/reasoning/synthesis.py` (Complexity Analyzer)
- After retrieval, insert a Complexity Analyzer node/function to score the query based on heuristics:
  - **Low Effort**: Short queries, few chunks.
  - **Medium Effort**: Summaries, multiple chunks (`len(context_chunks) > 4`).
  - **High Effort**: Code logic, math patterns, heavily convoluted language (`len(query.split()) > 40`), contradiction resolution.
- Modify the `synthesis_engine` inference payload to utilize the correct model dynamically:
  - **Low**: `llama-3.3-70b-versatile`
  - **Medium**: `llama-3.3-70b-versatile` (with higher temperature context mapping)
  - **High**: `openai/gpt-oss-120b`

## 5. Verification Plan
- **Automated Verification:** Run the `run_swagger_tests.py` script locally.
  - The script will be modified to safely load `GROQ_API_KEY` from the local `.env` file to prevent auth rejections.
  - The script will search for any active `uvicorn` processes running on port `8000` and explicitly kill them to prevent Qdrant locking/concurrency faults.
  - The script will programmatically spawn a fresh instance of the Uvicorn server in a subprocess before running the concurrent workflows, and gracefully terminate it upon completion.
  - Ensure the script outputs the newly added `latency_optimizations` metrics during Scenario A-E evaluations.
- **Expected Outcome:** 
  - File/Crawler pipelines remain unbothered.
  - Scenarios D and E return verified RAG chunks (fixing Reranker).
  - Terminal logs no longer display `[METADATA]` JSON schema fault errors.
  - `latency_optimizations` metrics in ChatResponse demonstrate the successful escalation to the proper LLM based on query complexity.
