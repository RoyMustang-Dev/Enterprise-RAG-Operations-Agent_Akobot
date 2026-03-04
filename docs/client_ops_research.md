# Client Ops and Scaling Research Notes

Date: 2026-03-04

This document summarizes guidance for:
1) benchmarking the crawler on the client VPS,
2) clarifying "endpoints" and deployment expectations,
3) concurrency and isolation behavior under load,
4) questions to clarify with the client.

It includes references to authoritative sources for deployment, Celery concurrency, Qdrant multitenancy, and proxy vendors.

---

## 1) Crawler benchmark on client VPS (akobot-agent@server.akobot.ai)

### Goal
Run the same crawler benchmark you already use locally (benchmark_crawler.py) on the client VPS, compare results, and capture logs and metrics.

### Steps (safe and repeatable)
1. SSH login:
   - ssh akobot-agent@server.akobot.ai
2. Clone or update the repo on the VPS:
   - git clone <repo> or git pull
3. Create and activate venv:
   - python3 -m venv venv
   - source venv/bin/activate
4. Install deps:
   - pip install -r requirements.txt
5. Set environment:
   - copy your .env (or create a minimal .env with Qdrant, API keys, and crawler settings)
6. Run benchmark in isolation:
   - python benchmark_crawler.py
7. Capture metrics:
   - save crawler_benchmark_results/ and terminal output

Notes:
- Keep benchmark URL and depth identical to local for a fair comparison.
- Measure CPU, RAM, and IO utilization during the run to locate bottlenecks.

---

## 2) "Give me the endpoints & I will deploy them" - what this means

When clients say this, they usually want:
- A public base URL of the API (not localhost).
- A deployment method for exposing endpoints securely.

Expected deliverables:
1. Public base URL (e.g., https://api.client.com)
2. OpenAPI/Swagger URL (e.g., https://api.client.com/docs)
3. List of endpoints they will call from the frontend

Deployment reality:
- Localhost endpoints are not reachable externally.
- Use a production ASGI deployment and a reverse proxy.

Reference:
- Uvicorn deployment guidance (production and Nginx): https://www.uvicorn.org/deployment/ (see production guidance and proxy notes).

---

## 3) "100 users hit chat at once - what happens?"

### Current behavior (today)
- FastAPI/Uvicorn handles concurrent requests but is limited by CPU/GPU, worker count, and LLM latency.
- LLM generation dominates latency in the current pipeline.
- Celery handles ingestion and crawler tasks, not live /chat responses.
- Qdrant can support multi-tenant isolation if payload filters and ingestion are correct.

### Risks today
1. Throughput bottleneck: generation dominates wall time.
2. Queue buildup: concurrent requests can stack and increase p95 latency.
3. Tenant isolation risk: if tenant metadata is missing on ingest, tenant queries can see empty or wrong results.
4. Session leakage risk: if session IDs are reused or validated poorly, session collections can be queried across users.

### What is missing for true multi-user concurrency
- Horizontal scaling of API workers
- Rate limiting per tenant and user
- Backpressure when LLM latency spikes
- Strict tenant-aware ingestion to prevent cross-tenant retrieval

### How large providers handle it (high level)
- Strict isolation per user or tenant
- Autoscaled worker pools to absorb bursts
- Queueing and load shedding under saturation
- Caching and retrieval pruning to reduce LLM tokens

References:
- Celery concurrency overview (prefork default, other pools): https://docs.celeryq.dev/en/stable/userguide/concurrency/index.html
- Qdrant multitenancy guidance (payload partitioning over many collections): https://qdrant.tech/documentation/guides/multitenancy/

---

## 4) Proxy URLs for CRAWLER_PROXY_URLS

Important: you cannot invent proxy URLs. They must come from a paid proxy provider and include credentials.

Recommended vendors (rotating proxies):
- Bright Data
- Oxylabs
- Smartproxy/Decodo

Reference comparisons:
- Proxy market research (Proxyway 2024): https://proxyway.com/research/proxy-market-research-2024

Expected format example:
CRAWLER_PROXY_URLS=http://user:pass@proxy.provider.com:port, http://user:pass@proxy.provider.com:port

Action:
- Client must purchase a proxy plan and supply the URLs. We should not hardcode these into code or .env.

---

## 5) Clarifications to ask the client

1. Deployment scope:
   - Self-hosted on their VPS or managed by us?
2. Endpoint exposure:
   - Which endpoints are public vs internal?
3. Concurrency SLO:
   - Target p50/p95 latency for /chat?
   - Expected concurrency peaks?
4. Data isolation:
   - Is x-tenant-id the primary boundary? Any cross-tenant access allowed?
5. Proxy usage:
   - Do they want to supply proxies or run crawler without them?

---

## 6) Reducing .env surface (design direction)

Keep .env for secrets and hard system settings only. Move dynamic decisions into code:
- Crawler: detect SPA and decide scroll/wait dynamically per site.
- Provider routing: select model based on latency budget and query complexity (no static env toggles).
- Vision: auto-fallback in code.

---

## 7) Summary of immediate next steps

1. Run crawler benchmark on VPS using the same script and parameters.
2. Provide client with public API URL, docs URL, and endpoint list.
3. Define concurrency targets (p95 latency and max concurrent users).
4. Decide on proxy provider and receive actual proxy URLs.
