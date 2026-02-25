import os
import subprocess

diagrams = {
    "step01_http_api_ingestion": """graph LR
  A([Input: Client JSON Payload]) --> B[FastAPI /api/v1/chat]
  B --> C([Output: Validated Request])
  style A fill:#e1f5fe,stroke:#0288d1
  style C fill:#fce4ec,stroke:#c2185b""",
    
    "step02_prompt_guard_security": """graph LR
  A([Input: Query String]) --> B{Prompt Guard Layer}
  B -->|Malicious| C([Output: HTTP 403 Security Exception])
  B -->|Safe| D([Output: Safe String])
  style B fill:#ffebee,stroke:#c62828""",
    
    "step03_query_expansion_rewriter": """graph LR
  A([Input: Safe String]) --> B[Rewriter gpt-oss-120b]
  B --> C([Output: JSON Expanded Queries])
  style B fill:#f3e5f5,stroke:#7b1fa2""",
    
    "step04_semantic_intent_triage": """graph LR
  A([Input: JSON Queries]) --> B{Intent Supervisor llama-3.1-8b}
  B --> C([Output: Intent Enum Array])
  style B fill:#e8f5e9,stroke:#388e3c""",

    "step05_dag_path_divergence": """graph LR
  A([Input: Intent Enum Array]) --> B{DAG Controller Node}
  B -->|Greeting| C([Output: Direct JSON Response])
  B -->|Code| D([Output: qwen-32b Route])
  B -->|RAG| E([Output: Proceed to Chunking])
  style B fill:#e8f5e9,stroke:#388e3c""",

    "step06_metadata_filter_extraction": """graph LR
  A([Input: Safe String]) --> B[Metadata Extractor llama-3.1-8b]
  B --> C([Output: JSON Qdrant Filters])
  style B fill:#f3e5f5,stroke:#7b1fa2""",

    "step07_qdrant_similarity_search": """graph LR
  A([Input: Filters + Embeddings]) --> B[(Qdrant Similarity Search)]
  B --> C([Output: 30 Raw Vectors])
  style B fill:#fff3e0,stroke:#f57c00""",

    "step08_cross_encoder_reranker": """graph LR
  A([Input: 30 Raw Vectors]) --> B{Cross-Encoder Reranker Sigmoid}
  B --> C([Output: 5 Validated Chunks])
  style B fill:#e8f5e9,stroke:#388e3c""",

    "step09_complexity_heuristic_analyzer": """graph LR
  A([Input: Chunks + Query]) --> B{Complexity Analyzer}
  B --> C([Output: Target Engine Enum])
  style B fill:#e8f5e9,stroke:#388e3c""",

    "step10_reasoning_synthesis": """graph LR
  A([Input: Target Route Enum]) --> B[Reasoning Synthesis API]
  B --> C([Output: Raw Generated RAG JSON Draft])
  style B fill:#e8eaf6,stroke:#3f51b5""",
  
    "step11_sarvam_fact_verifier": """graph LR
  A([Input: Generated Draft]) --> B{Fact Verifier Sarvam M}
  B --> C([Output: Hallucination Boolean + Array])
  style B fill:#ffebee,stroke:#c62828""",

    "step12_json_api_formatter": """graph LR
  A([Input: Final AgentState]) --> B[Model Formatter]
  B --> C([Output: HTTP 200 API Response JSON])
  style B fill:#f3e5f5,stroke:#7b1fa2"""
}

os.makedirs("assets", exist_ok=True)

for name, content in diagrams.items():
    mmd_path = os.path.join("assets", f"{name}.mmd")
    png_path = os.path.join("assets", f"{name}.png")
    
    with open(mmd_path, "w") as f:
        f.write(content)
    
    print(f"Generating {png_path}...")
    subprocess.run(["npx", "-p", "@mermaid-js/mermaid-cli", "mmdc", "-i", mmd_path, "-o", png_path, "-b", "transparent"], check=True, shell=True)

print("All 12 diagrams successfully generated!")
