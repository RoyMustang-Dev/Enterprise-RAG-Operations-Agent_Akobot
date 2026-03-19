# Stress Test & Persona Report

Generated: 2026-03-15T05:45:18

## 1) Persona Handling (How It Works)
- Persona is created via `POST /api/v1/agents` and persisted in `data/agent_profiles.db`.
- `PersonaCacheManager` loads the most recent persona into memory and injects it into prompts via `get_compiled_prompt()`.
- Injection mode is controlled by `PERSONA_INJECTION_MODE` in `.env`:
  - `strict`: inject into `rag_synthesis`, `smalltalk_agent`, `multimodal_voice` and partial into `intent_classifier`, `coder_agent`.
  - `disabled`: no persona injected (used for A/B comparison below).
- Persona fields used for this test:
  - `bot_name`: ako-test-bot
  - `brand_details`: Akobot.ai is one of the best leading AI SaaS companies providing various services like Image Generation, Video Generation, Custom AI RAG Agents, etc. So what are you waiting for, come join us now & have fun!!!
  - `welcome_message`: Welcome to Akobot.ai we hlep our customers build AI Agents, Custom RAG Agents, with other plethora of services. What are you interested in, let us know & we will help out.
  - `prompt_instructions`: You are very intelligent & professional Support giver agent, you help customers with their various queries ranging from normal summarizations to most advanced RAG-requests. You have multi-modal capabilities so that you can better help the customers.

## 2) Persona A/B Evidence (Persona ON vs OFF)
## Persona ON (strict)

### Greeting Query
- query: hii how can you help me
- status: 200
- latency_s: 15.21
```json
{
  "session_id": "f5bf2449-9a4c-4050-9697-c1e6d4bfab55",
  "answer": "Hi! I'm ako-test-bot. How can I help you today?",
  "sources": [],
  "confidence": 0.99,
  "verifier_verdict": "PENDING",
  "is_hallucinated": false,
  "optimizations": {
    "model_provider": "modelslab",
    "reranker_profile": "llm_judge",
    "reranker_model_name": "llama-3.1-8b-instant",
    "original_query": "hii how can you help me",
    "rewritten_query": "Hi! How can you help me?",
    "complexity_score": 0.0,
    "retrieval_scope": "both",
    "agent_routed": "smalltalk"
  },
  "chat_history": [
    {
      "role": "user",
      "content": "hii how can you help me"
    },
    {
      "role": "assistant",
      "content": "Hi! I'm ako-test-bot. How can I help you today?"
    }
  ],
  "latency_optimizations": {},
  "active_persona": "ako-test-bot"
}
```

### RAG Query
- query: summarize all the data that we have in the sources & explain them properly
- status: 200
- latency_s: 28.83
```json
{
  "session_id": "2e657d21-aa9d-417d-aba5-3d5eb567d76b",
  "answer": "## Summary of Sources\nThe provided sources offer an overview of the services and capabilities of Akobot.ai, a leading AI SaaS company. The main findings can be summarized as follows:\n### Company Overview\nAkobot.ai is an all-in-one AI platform that provides tools for chat, images, and videos, powered by the world's best models. The company enables the creation of custom AI agents with multi-flow intelligence for any use case [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n### Video Tools\nThe platform offers a range of video tools, including:\n* T2Video: generates stunning cinematic videos from simple text prompts\n* AI Editing: transforms and modifies existing video clips with AI precision\n* Video Upscale: enhances video resolution and clarity up to 4K quality\n* Motion Synthesis: adds realistic motion and fluid animations to static scenes\n* Style Transfer: applies artistic styles and cinematic filters to footage\n* Frame Interpolation: creates buttery smooth slow-motion by generating extra frames\n* Video2Video: transforms one video into another with AI-powered conversion\n* Video2Image: converts video clips into stunning high-res images\n* Image2Video: brings images to life by transforming them into dynamic videos [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n### Image Tools\nThe platform also offers a range of image tools, including:\n* Text2Image: generates stunning images from text prompts using advanced AI models\n* Image2Image: transforms and modifies existing images with AI-powered conversion\n* Background Removal: instantly removes backgrounds from images with precision and accuracy\n* Avatar Generation: creates unique AI-generated avatars and profile pictures in various styles\n* Upscale: enhances image resolution and quality with AI-powered upscaling\n* Object Removal: removes unwanted elements from images seamlessly with AI filling [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n### API and Integration\nThe platform provides a powerful REST API for developers to integrate AI generation into their applications. The API offers fast inference, simple authentication, and scalable infrastructure. It also provides SDKs in various programming languages, including Python, Node.js, and Go [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n### Benefits for Developers\nThe platform offers several benefits for developers, including:\n* Fast inference with sub-second response times\n* 99.9% uptime with enterprise-grade reliability and SLA\n* Secure and private with end-to-end encryption and data protection\n* Real-time capabilities with WebSocket support for streaming responses\n* Global scale with multi-region deployment for low latency [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\nIn summary, Akobot.ai provides a comprehensive platform for creating custom AI agents, generating images and videos, and integrating AI capabilities into applications. The platform offers a range of tools and services that cater to the needs of developers, creators, and businesses.",
  "sources": [
    {
      "source": "0104ec25fee1014adcbfb9dae09da25e.txt",
      "score": 0.42875072,
      "text": "H3: Video Tools\nH3: Video Models\nH3: Image Models\nH3: Image Tools\nH3: Company\nAKOBOT.ai - Create Images & Videos with AI\nCreate Custom AI Agents with Multi-Flow Intelligence for Any Use Case\n-------------------------------------------------"
    },
    {
      "source": "0104ec25fee1014adcbfb9dae09da25e.txt",
      "score": 0.49726543,
      "text": "### Content Creation\nCreate engaging visual content for social media and marketing.\nVideo Tool Showcase\n-------------------\nComing Soon\nSlow-scrolling cinematic previews of our core AI video capabilities.\n[](/feeds/video1.mp4)\n### T2Video\nG"
    },
    {
      "source": "0104ec25fee1014adcbfb9dae09da25e.txt",
      "score": 0.48579198,
      "text": "Discover how cutting-edge AI transforms your creative workflow.\n![Text2Image](/fe
```

### BA Query
- query: Identify the future trends of all products based on current data for every region individually.
- status: 200
- latency_s: 12.46
```json
{
  "status": "success",
  "agent": "BUSINESS_ANALYST",
  "data": {
    "summary_paragraph": "Analyzed 1200 rows, date range 2023-01-01 to 2026-04-14, 4 regions, 5 products. Forecasting used xgboost for near-term trend projection. Overall regional trend: down in 2/4, up in 2/4, flat in 0/4. Top movers by region: Asia - Keyboard (down, -21.51%), Smartphone (down, -10.07%); Europe - Monitor (down, -6.61%), Tablet (down, -6.46%); North America - Monitor (up, 9.53%), Laptop (up, 7.00%); South America - Monitor (up, 10.74%), Smartphone (up, 10.74%).",
    "per_region_summaries": [
      {
        "region": "Asia",
        "top_products": [
          {
            "product": "Keyboard",
            "forecast_delta_percent": -21.514718163621556,
            "direction": "down",
            "reason": "Forecast moved from 272.20 to 213.64."
          },
          {
            "product": "Smartphone",
            "forecast_delta_percent": -10.070277077227564,
            "direction": "down",
            "reason": "Forecast moved from 308.23 to 277.19."
          },
          {
            "product": "Monitor",
            "forecast_delta_percent": -9.006048844166411,
            "direction": "down",
            "reason": "Forecast moved from 290.73 to 264.55."
          }
        ],
        "overall_region_trend": "down",
        "notes": "3 products evaluated."
      },
      {
        "region": "Europe",
        "top_products": [
          {
            "product": "Monitor",
            "forecast_delta_percent": -6.6104190378435765,
            "direction": "down",
            "reason": "Forecast moved from 282.26 to 263.60."
          },
          {
            "product": "Tablet",
            "forecast_delta_percent": -6.457628983172135,
            "direction": "down",
            "reason": "Forecast moved from 229.71 to 214.88."
          },
          {
            "product": "Laptop",
            "forecast_delta_percent": -4.796997848769455,
            "direction": "down",
            "reason": "Forecast moved from 278.95 to 265.57."
          }
        ],
        "overall_region_trend": "down",
        "notes": "3 products evaluated."
      },
      {
        "region": "North America",
        "top_products": [
          {
            "product": "Monitor",
            "forecast_delta_percent": 9.525722032669655,
            "direction": "up",
            "reason": "Forecast moved from 215.20 to 235.70."
          },
          {
            "product": "Laptop",
            "forecast_delta_percent": 7.004778390788985,
            "direction": "up",
            "reason": "Forecast moved from 251.03 to 268.62."
          },
          {
            "product": "Tablet",
            "forecast_delta_percent": 6.543154110848544,
            "direction": "up",
            "reason": "Forecast moved from 228.29 to 243.23."
          }
        ],
        "overall_region_trend": "up",
        "notes": "3 products evaluated."
      },
      {
        "region": "South America",
        "top_products": [
          {
            "product": "Monitor",
            "forecast_delta_percent": 10.744513099825424,
            "direction": "up",
            "reason": "Forecast moved from 293.08 to 324.57."
          },
          {
            "product": "Smartphone",
            "forecast_delta_percent": 10.744513099825424,
            "direction": "up",
            "reason": "Forecast moved from 293.08 to 324.57."
          },
          {
            "product": "Tablet",
            "forecast_delta_percent": 10.744513099825424,
            "direction": "up",
            "reason": "Forecast moved from 293.08 to 324.57."
          }
        ],
        "overall_region_trend": "up",
        "notes": "3 products evaluated."
      }
    ],
    "forecast_table": [
      {
        "region": "Asia",
        "product": "Keyboard",
        "date": "2026-04-15",
        "metric": "units_sold",
        "forecast_value": 272.19836,
        "lower_ci": null,
       
```

## Persona OFF (disabled)

### Greeting Query
- query: hii how can you help me
- status: 200
- latency_s: 15.47
```json
{
  "session_id": "3fad42df-55be-448d-ad7d-bb2aa23f1cf5",
  "answer": "Hi! I'm ako-test-bot. How can I help you today?",
  "sources": [],
  "confidence": 0.99,
  "verifier_verdict": "PENDING",
  "is_hallucinated": false,
  "optimizations": {
    "model_provider": "modelslab",
    "reranker_profile": "llm_judge",
    "reranker_model_name": "llama-3.1-8b-instant",
    "original_query": "hii how can you help me",
    "rewritten_query": "Hi! How can you help me?",
    "complexity_score": 0.0,
    "retrieval_scope": "both",
    "agent_routed": "smalltalk"
  },
  "chat_history": [
    {
      "role": "user",
      "content": "hii how can you help me"
    },
    {
      "role": "assistant",
      "content": "Hi! I'm ako-test-bot. How can I help you today?"
    }
  ],
  "latency_optimizations": {},
  "active_persona": "ako-test-bot"
}
```

### RAG Query
- query: summarize all the data that we have in the sources & explain them properly
- status: 200
- latency_s: 30.41
```json
{
  "session_id": "161ee3db-859b-4521-9315-55cb0bb1a640",
  "answer": "## Summary of Sources\nThe provided sources appear to be related to AKOBOT.ai, a platform that utilizes AI to create images and videos. The main findings can be summarized as follows:\n### Key Features\n* **Automate 80%+ of interactions with AI agents**: AKOBOT.ai allows users to create custom AI agents with multi-flow intelligence for various use cases [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n* **Image and Video Creation**: The platform provides tools for generating stunning images from text prompts, transforming existing images, and creating cinematic videos [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n* **AI Editing and Enhancement**: AKOBOT.ai offers features for editing and enhancing videos, including upscaling, motion synthesis, and style transfer [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n### Tools and Models\n* **Text2Image**: Generates images from text prompts using advanced AI models [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n* **Image2Image**: Transforms and modifies existing images with AI-powered conversion [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n* **Video Tools**: Includes features like video upscaling, motion synthesis, and style transfer [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n### Target Audience\n* **Creators**: The platform is designed for creators, including designers, artists, and developers, to bring their creative visions to life [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n* **Developers**: AKOBOT.ai is also geared towards developers, providing them with the tools and features needed to build powerful AI agents [Doc 0104ec25fee1014adcbfb9dae09da25e.txt].\n## Conclusion\nIn summary, the sources provide an overview of AKOBOT.ai's features, tools, and target audience. The platform offers a range of AI-powered tools for creating and editing images and videos, as well as building custom AI agents. With its focus on creators and developers, AKOBOT.ai aims to provide a comprehensive platform for bringing creative visions to life.",
  "sources": [
    {
      "source": "0104ec25fee1014adcbfb9dae09da25e.txt",
      "score": 0.4243164,
      "text": "H3: Video Tools\nH3: Video Models\nH3: Image Models\nH3: Image Tools\nH3: Company\nAKOBOT.ai - Create Images & Videos with AI\nCreate Custom AI Agents with Multi-Flow Intelligence for Any Use Case\n-------------------------------------------------"
    },
    {
      "source": "0104ec25fee1014adcbfb9dae09da25e.txt",
      "score": 0.42653757,
      "text": "== AKOBOT.ai - Create Images & Videos with AI ==\nH2: Create Custom AI Agents withMulti-Flow Intelligencefor Any Use Case\nH2: ALL the Great AI Video & Image Modelsin ONE Place!\nH2: Agent ToolShowcase\nH2: Image ToolShowcase\nH2: Video ToolShow"
    },
    {
      "source": "0104ec25fee1014adcbfb9dae09da25e.txt",
      "score": 0.49180347,
      "text": "### Content Creation\nCreate engaging visual content for social media and marketing.\nVideo Tool Showcase\n-------------------\nComing Soon\nSlow-scrolling cinematic previews of our core AI video capabilities.\n[](/feeds/video1.mp4)\n### T2Video\nG"
    },
    {
      "source": "0104ec25fee1014adcbfb9dae09da25e.txt",
      "score": 0.48250666,
      "text": "Discover how cutting-edge AI transforms your creative workflow.\n![Text2Image](/feeds/image1.jpg)\n### Text2Image\nGenerate stunning images from text prompts using advanced AI models.\n![Image2Image](/feeds/image2.jpg)\n### Image2Image\nTransform"
    },
    {
      "source": "0104ec25fee1014adcbfb9dae09da25e.txt",
      "score": 0.41482067,
      "text": "[](/feeds/video6.mp4)\n### Frame Interpolation\nCreate buttery smooth slow-motion by generating extra frames.\n[](/feeds/video7.mp4)\n### Video2Video\nTransform one video into another with AI-powered conversion.\n[](/feeds/video8.mp4)\n### Video2I"
    }
  ],
  "confidence": 0.0,
  "verifier_verdict": "UNVERIFIED",
  "is_hallucinated": false,
  "optimizations": {
    "mo
```

### BA Query
- query: Identify the future trends of all products based on current data for every region individually.
- status: 200
- latency_s: 12.91
```json
{
  "status": "success",
  "agent": "BUSINESS_ANALYST",
  "data": {
    "summary_paragraph": "Analyzed 1200 rows, date range 2023-01-01 to 2026-04-14, 4 regions, 5 products. Forecasting used xgboost for near-term trend projection. Overall regional trend: down in 2/4, up in 2/4, flat in 0/4. Top movers by region: Asia - Keyboard (down, -21.51%), Smartphone (down, -10.07%); Europe - Monitor (down, -6.61%), Tablet (down, -6.46%); North America - Monitor (up, 9.53%), Laptop (up, 7.00%); South America - Monitor (up, 10.74%), Smartphone (up, 10.74%).",
    "per_region_summaries": [
      {
        "region": "Asia",
        "top_products": [
          {
            "product": "Keyboard",
            "forecast_delta_percent": -21.514718163621556,
            "direction": "down",
            "reason": "Forecast moved from 272.20 to 213.64."
          },
          {
            "product": "Smartphone",
            "forecast_delta_percent": -10.070277077227564,
            "direction": "down",
            "reason": "Forecast moved from 308.23 to 277.19."
          },
          {
            "product": "Monitor",
            "forecast_delta_percent": -9.006048844166411,
            "direction": "down",
            "reason": "Forecast moved from 290.73 to 264.55."
          }
        ],
        "overall_region_trend": "down",
        "notes": "3 products evaluated."
      },
      {
        "region": "Europe",
        "top_products": [
          {
            "product": "Monitor",
            "forecast_delta_percent": -6.6104190378435765,
            "direction": "down",
            "reason": "Forecast moved from 282.26 to 263.60."
          },
          {
            "product": "Tablet",
            "forecast_delta_percent": -6.457628983172135,
            "direction": "down",
            "reason": "Forecast moved from 229.71 to 214.88."
          },
          {
            "product": "Laptop",
            "forecast_delta_percent": -4.796997848769455,
            "direction": "down",
            "reason": "Forecast moved from 278.95 to 265.57."
          }
        ],
        "overall_region_trend": "down",
        "notes": "3 products evaluated."
      },
      {
        "region": "North America",
        "top_products": [
          {
            "product": "Monitor",
            "forecast_delta_percent": 9.525722032669655,
            "direction": "up",
            "reason": "Forecast moved from 215.20 to 235.70."
          },
          {
            "product": "Laptop",
            "forecast_delta_percent": 7.004778390788985,
            "direction": "up",
            "reason": "Forecast moved from 251.03 to 268.62."
          },
          {
            "product": "Tablet",
            "forecast_delta_percent": 6.543154110848544,
            "direction": "up",
            "reason": "Forecast moved from 228.29 to 243.23."
          }
        ],
        "overall_region_trend": "up",
        "notes": "3 products evaluated."
      },
      {
        "region": "South America",
        "top_products": [
          {
            "product": "Monitor",
            "forecast_delta_percent": 10.744513099825424,
            "direction": "up",
            "reason": "Forecast moved from 293.08 to 324.57."
          },
          {
            "product": "Smartphone",
            "forecast_delta_percent": 10.744513099825424,
            "direction": "up",
            "reason": "Forecast moved from 293.08 to 324.57."
          },
          {
            "product": "Tablet",
            "forecast_delta_percent": 10.744513099825424,
            "direction": "up",
            "reason": "Forecast moved from 293.08 to 324.57."
          }
        ],
        "overall_region_trend": "up",
        "notes": "3 products evaluated."
      }
    ],
    "forecast_table": [
      {
        "region": "Asia",
        "product": "Keyboard",
        "date": "2026-04-15",
        "metric": "units_sold",
        "forecast_value": 272.19836,
        "lower_ci": null,
       
```



## 3) Current Persona Visibility in Responses
- `active_persona` field appears in response metadata when smalltalk is used.
- For RAG/BA, persona affects prompt style and sometimes phrasing; it will be visible if the prompt is constructed to echo greeting/brand voice.

## 4) Groq Rate-Limit Backoff (Reward Model / LLM Calls)
- Added exponential backoff for Groq in `app/infra/llm_client.py`.
- Env vars:
  - `GROQ_RETRY_MAX` (default 3)
  - `GROQ_RETRY_BACKOFF_SEC` (default 1.5)

## 5) Composio Plan (High-Level)
- Use Composio sessions to dynamically discover tools, authenticate, and execute actions with user context.
- Use MCP endpoints for framework-agnostic tool access; use native tool providers where tighter integration is needed.
- Use Composio Workbench for heavy data operations and large tool outputs to avoid context overflow.

## 6) Langfuse (RAG + BA)
- RAG: Langfuse callbacks are initialized in `app/agents/rag.py`.
- BA: Langfuse callbacks are initialized in `app/agents/data_analytics/supervisor.py`.
- Verify in Langfuse UI by searching for traces containing your query text and session_id.

## 7) Crawling + Ingestion Lifecycle (Control-Flow Detail)
1. User calls `/api/v1/ingest/crawler` or `/api/v1/ingest/files`.
2. API creates a `JobTracker` entry in SQLite and returns `job_id`.
3. Celery receives job and loads tenant DB + crawler DB.
4. Hardware probe decides batch sizes & worker counts.
5. Crawler (if enabled) launches Chromium workers and writes crawled documents into `data/crawled_docs/...`.
6. Ingestion pipeline loads files, chunks them, applies truncation safeguards, and generates embeddings.
7. Embeddings are written to Qdrant collections (per-tenant or global based on config).
8. Job status and timings are persisted and visible via `/api/v1/progress/{job_id}`.

## 8) Streaming Hybrid Approach (Preferred Native Streaming)
- Preferred: provider-native streaming (Groq/Gemini) in synthesis.
- Fallback: SSE chunked responses for models without native streaming.
- Current implementation: native streaming in synthesis ? SSE wrapper in API for UI.

## 9) Stress Test Queries & Data Sources
### RAG Stress Queries
- Multi-domain synthesis + citation: ?Summarize product, API, and compliance policy; compare versions; provide rollout risks with citations.?
- Multi-hop troubleshooting: ?Diagnose latency regressions across ingestion, retrieval, and LLM calls; propose fixes with evidence.?

### BA Stress Queries
- ?Perform region-level forecasting with confidence intervals; detect anomalies; provide causal hypotheses and sensitivity analysis.?
- ?Segment by product-category and region; build scenario simulations for +10% marketing spend and report revenue uplift.?

### Suggested RAG Crawl Targets
- Your internal docs + Akobot.ai website content.
- Composio docs (toolkits, workbench, how it works).

### Suggested BA Datasets
- UCI Online Retail dataset.
- NYC TLC Trip Record Data (yellow/green/FHV trips) + data dictionary.

Place all stress test assets (crawled docs + datasets) inside `Stress Test/`.


## 10) Support Data Analytics Agent (Skeleton)
- Implemented in `app/agents/support_data_analytics.py` as a non-wired blueprint.
- Contains tools: prompt guard, prompt rewriter, smalltalk, RAG, BA, auto-email.
- Currently safe and inert (not routed by default).

## 11) Composio Tool Skeletons
- `app/integrations/composio_mcp.py` provides MCP wrapper.
- `app/integrations/composio_tools.py` provides discovery/augmentation stubs for RAG/BA.
- `app/tools/emailer.py` provides `send_email_via_composio` tool.


## 12) Composio Auto Email Support
- Composio Gmail toolkit includes a **Send Email** tool that can send emails via Gmail for the authenticated user.
- This can be wired via MCP or native tools once COMPOSIO_API_KEY + MCP URL are configured.
- Reference: https://composio.dev/toolkits/gmail

## 13) Enterprise-Readiness Implementation Details
- Redis rate limiter: replaces in-memory token bucket; enforces limits across multiple workers/instances.
- Distributed locks: prevents duplicate ingestion or double tool execution across workers.
- Multi-worker Uvicorn + autoscaling: uses multiple Uvicorn workers per host + scale-out with load balancer.
- AuthN/AuthZ: tenant isolation at API gateway and in app logic with scoped access tokens.
- WAF/API Gateway: blocks malicious traffic, enforces auth, rate limits, and request validation at edge.
- Circuit breakers/bulkheads: isolate LLM failures and prevent cascading latency spikes.

## 14) Celery Concurrency vs Worker Count
- **Concurrency** = number of task execution slots in a worker.
- **Worker count** = number of separate worker processes/containers.
- Current setup uses `--pool=solo --concurrency=1` in `scripts/start_celery_worker.ps1` for Windows stability.
- Hardware probe affects batch sizes and crawler workers inside ingestion, not Celery concurrency itself.

## 15) Hybrid Streaming Mapping
- RAG synthesis uses native provider streaming (Groq/Gemini) when available.
- API always streams to the frontend over SSE.
- If native streaming is not available, API falls back to chunked SSE using final answer.

## 16) Tracked Implementation Plan and Checklist
Generated: 2026-03-15T06:02:00

### 16.1 AuthN/AuthZ Tenant Isolation Without Client Tokens
Goal: Work with or without client auth by supporting two modes.
Mode A: Client brings identity token.
Mode B: Platform issues its own tenant-scoped access token.

Plan:
1. Add configuration `AUTH_MODE` with values `client_token` or `platform_token`.
2. Implement middleware to extract `tenant_id`, `user_id`, `roles`, and `auth_token`.
3. If `AUTH_MODE=client_token`, validate incoming client token and map to tenant.
4. If `AUTH_MODE=platform_token`, issue a tenant-scoped JWT on login or API key creation.
5. Add tenant enforcement to SQL queries, Qdrant collections or payload filters, and file paths for uploads and crawled docs.
6. Add audit logs for deny events.

Checklist:
- [ ] Add `AUTH_MODE` to config
- [ ] Implement auth middleware with two modes
- [ ] Token validator for client JWT or API key
- [ ] Tenant mapping table and lookup
- [ ] Enforce tenant filters in SQL
- [ ] Enforce tenant filters in Qdrant
- [ ] Enforce tenant filters in filesystem paths
- [ ] Add deny audit logs
- [ ] Add tests for cross-tenant access

### 16.2 RBAC vs Payment Tiers
Goal: Keep billing tiers as entitlement, add RBAC for operational safety.

Plan:
1. Keep payment tiers for usage limits and features.
2. Add RBAC for permissions such as ingestion, crawler, email sending, and admin actions.
3. Map tiers to default roles.
4. Allow overrides for enterprise accounts.

Checklist:
- [ ] Define roles and permissions
- [ ] Map tiers to roles
- [ ] Add role checks to sensitive endpoints
- [ ] Add admin override support
- [ ] Add tests for role enforcement

### 16.3 Redis Rate Limiter and Distributed Locks
Plan:
1. Add Redis client with pooling.
2. Replace in-memory limiter with Redis token bucket.
3. Add distributed lock for ingestion, crawler, and tool calls.
4. Set TTL and retry policy.
5. Add metrics for lock wait and reject rates.

Checklist:
- [ ] Redis client + config
- [ ] Redis limiter implementation
- [ ] Replace in-memory limiter
- [ ] Add distributed locks
- [ ] Add metrics
- [ ] Add tests for rate limit and lock contention

### 16.4 Multi-worker Uvicorn + Autoscaling
Plan:
1. Run `uvicorn` with multiple workers.
2. Ensure all shared state is externalized to Redis or DB.
3. Add HPA based on CPU and latency.
4. Add readiness and liveness probes.

Checklist:
- [ ] Multi-worker config
- [ ] Externalize state
- [ ] Autoscaling policy
- [ ] Health checks

### 16.5 WAF or API Gateway
Plan:
1. Choose gateway.
2. Enforce auth, rate limits, and schema validation.
3. Add bot or abuse protection.
4. Add allowlists for internal endpoints.

Checklist:
- [ ] Gateway selected
- [ ] Auth and validation rules
- [ ] Rate limit policies
- [ ] Bot protection
- [ ] Internal endpoint allowlists

### 16.6 Circuit Breakers and Bulkheads for LLM Calls
Plan:
1. Add circuit breaker per provider.
2. Add per-provider concurrency limits.
3. Add fallback ladder.
4. Add metrics for opens and retries.

Checklist:
- [ ] Circuit breaker implementation
- [ ] Concurrency caps
- [ ] Fallback ladder
- [ ] Metrics and dashboards

### 16.7 Composio Auto-Send Email
Plan:
1. Wire Composio MCP client.
2. Add `send_email` tool using Gmail toolkit.
3. Register tool with RAG, BA, Support-DA.
4. Add permission policy and allowlist.
5. Add audit logs.

Checklist:
- [ ] MCP client configured
- [ ] `send_email` tool implemented
- [ ] Tool registered in RAG
- [ ] Tool registered in BA
- [ ] Tool registered in Support-DA
- [ ] Permissions and allowlists
- [ ] Audit logs

### 16.8 Celery Concurrency
Plan:
1. Identify explicit concurrency overrides.
2. Allow dynamic concurrency from CPU cores.
3. Enable autoscale for workers.
4. Ensure task idempotency + locks.

Checklist:
- [ ] Locate overrides
- [ ] Dynamic concurrency setting
- [ ] Autoscale enabled
- [ ] Task idempotency review

### 16.9 Hybrid Streaming Native First, SSE Fallback
Plan:
1. Add provider capability map for streaming.
2. Use native streaming where supported.
3. SSE chunked fallback for non-streaming.
4. Add stream mode metrics.

Checklist:
- [ ] Capability map
- [ ] Native streaming enforced
- [ ] SSE fallback path
- [ ] Stream mode metrics

### 16.10 Support Data Analytics Agent + Composio
Plan:
1. Add Support-DA agent in routing registry.
2. Add tool registry for prompt guard, rewrite, smalltalk, RAG, BA, and email.
3. Add Langfuse traces per tool call.
4. Add tests for tool routing.

Checklist:
- [ ] Agent registered
- [ ] Tools wired
- [ ] Langfuse tracing
- [ ] Tool routing tests

### 16.11 Composio for RAG + BA Without CRM
Plan:
1. Use non-CRM toolkits only.
2. Add safe tool discovery and allowlist.
3. Enforce per-tenant permissions.
4. Add tool execution audit logs.

Checklist:
- [ ] Non-CRM toolkit allowlist
- [ ] Tool discovery
- [ ] Permission checks
- [ ] Audit logs
