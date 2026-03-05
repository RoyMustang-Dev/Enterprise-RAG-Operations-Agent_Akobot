"""
Token budget utilities for trimming retrieved context to fit model limits.
Uses tiktoken when available for consistent token estimation.
"""
from typing import List, Dict, Tuple
import os

try:
    import tiktoken
except Exception:
    tiktoken = None

DEFAULT_CONTEXT_LIMIT = int(os.getenv("MODEL_CONTEXT_LIMIT", "8192"))

MODEL_LIMITS = {
    "llama-3.3-70b-versatile": int(os.getenv("LLAMA_3_3_70B_CTX", "8192")),
    "llama-3.1-8b-instant": int(os.getenv("LLAMA_3_1_8B_CTX", "8192")),
    "openai/gpt-oss-120b": int(os.getenv("GPT_OSS_120B_CTX", "8192")),
}


def _get_limit(model_name: str) -> int:
    if not model_name:
        return DEFAULT_CONTEXT_LIMIT
    return MODEL_LIMITS.get(model_name, DEFAULT_CONTEXT_LIMIT)


def _get_encoder(model_name: str):
    if not tiktoken:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model_name: str = "") -> int:
    if not text:
        return 0
    encoder = _get_encoder(model_name)
    if encoder:
        return len(encoder.encode(text))
    return max(1, len(text) // 4)


def trim_chunks_by_token_budget(
    chunks: List[Dict],
    model_name: str,
    reserve_tokens: int = 1024,
) -> Tuple[List[Dict], int, int]:
    limit = _get_limit(model_name)
    budget = max(512, limit - reserve_tokens)
    used = 0
    kept = []
    for item in chunks:
        text = item.get("text") or item.get("page_content") or ""
        t = estimate_tokens(text, model_name)
        if used + t > budget:
            break
        kept.append(item)
        used += t
    return kept, used, limit
