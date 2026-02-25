# Task Breakdown

## PART 1: Core System Stabilization

### 1. Chat History Fix
- [x] Update `app/supervisor/router.py` to append user queries and assistant responses to the runtime state.
- [x] Update `ChatResponse` schema in `app/api/routes.py` to return the updated history.

### 2. JSON Mode Stability (Metadata & Rewriter)
- [x] Update `app/retrieval/metadata_extractor.py` to use `llama-3.1-8b-instant`.
- [x] Update `app/prompt_engine/rewriter.py` to use `openai/gpt-oss-120b`.
- [x] Ensure both files explicitly utilize `max_completion_tokens`.
- [x] Add defensive markdown-to-JSON fallback extraction logic using `try/except`.

### 3. Reranker Fix
- [x] Apply Sigmoid function `1 / (1 + math.exp(-score))` to CrossEncoder logit outputs before the `0.35` threshold check in `app/retrieval/reranker.py`.

### 4. Stack Re-Alignment
- [x] Verify Intent Classifier uses `llama-3.1-8b-instant`.
- [x] Verify Guard uses `llama-prompt-guard-2-86m`.
- [x] Verify Code Agent uses `qwen/qwen3-32b` or equivalent.
- [x] Verify `synthesis.py` and verifiers utilize the corrected stack models as defined.

## PART 2: Enterprise Dynamic Routing Strategy

### 1. State Updates
- [x] Add `reasoning_effort` and `latency_optimizations` to `AgentState` in `app/core/types.py`.
- [x] Expose `latency_optimizations` in the `ChatResponse` model in `api/routes.py`.

### 2. Complexity Analyzer Integration
- [x] Introduce heuristic tracking within `app/reasoning/synthesis.py` or as a distinct LangGraph node.
- [x] Assess query length, chunk count, and semantic patterns to assign `low`, `medium`, or `high`.
- [x] Dynamically switch the endpoint invocation between `llama-3.3-70b-versatile` and `openai/gpt-oss-120b` based on the computed effort logic.

## Verification
- [x] Refactor `run_swagger_tests.py` to auto-kill existing instances on port 8000.
- [x] Modify `run_swagger_tests.py` to spawn the Uvicorn server in a subprocess autonomously.
- [x] Modify `run_swagger_tests.py` to assert `GROQ_API_KEY` before execution.
- [x] Modify `run_swagger_tests.py` to print `latency_optimizations` and `chat_history` metrics.
- [x] Execute `run_swagger_tests.py` and verify all endpoints succeed natively without throwing Uvicorn JSON extraction faults.

## PART 3: Post-Deployment Tidy Up

### 1. Test Suite Relocation
- [ ] Create the `/tests/` directory if it does not exist.
- [ ] Verify all root-level python files that are meant for testing (e.g., `run_swagger_tests.py` and any other ad-hoc test files) and move them securely into the `/tests/` folder.
- [ ] Refactor any necessary import/sys paths inside those files if the relocation breaks relative pathing.

### 2. Artifact Consolidation
- [ ] Create a new root directory named `/research-md-files-artifacts/`.
- [ ] Identify all loose markdown files within the root directory (excluding the official `README.md`) generated during the research phases (e.g., `groq_comparison.md`, etc.).
- [ ] Identify any `.gemini` framework generated artifact documents spanning the lifecycle of this application buildup.
- [ ] Copy or move all identified research artifacts into the new `/research-md-files-artifacts/` folder.
