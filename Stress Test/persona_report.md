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

