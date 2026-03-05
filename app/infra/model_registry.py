"""
Model Registry

Centralized mapping of RAG phases to model/provider selections.
"""
from __future__ import annotations

import os
from typing import Dict, Any


PHASE_MODELS: Dict[str, Dict[str, Any]] = {
    # Lightweight routing / classification
    "intent_classifier": {
        "provider": "groq",
        "fallback_provider": "modelslab",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 120,
    },
    "intent_classifier_escalation": {
        "provider": "modelslab",
        "fallback_provider": "gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0.1,
        "max_tokens": 200,
    },
    "source_scope_classifier": {
        "provider": "groq",
        "fallback_provider": "modelslab",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 120,
    },
    "source_scope_classifier_escalation": {
        "provider": "modelslab",
        "fallback_provider": "gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0.1,
        "max_tokens": 200,
    },
    "security_guard": {
        "provider": "groq",
        "fallback_provider": "gemini",
        "model": "meta-llama/llama-guard-4-12b",
        "temperature": 0.0,
        "max_tokens": 20,
    },

    # Query rewrite / extraction / scoring
    "query_rewriter": {
        "provider": "groq",
        "fallback_provider": "modelslab",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 900,
    },
    "metadata_extractor": {
        "provider": "groq",
        "fallback_provider": "modelslab",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.0,
        "max_tokens": 300,
    },
    "complexity_scorer": {
        "provider": "groq",
        "fallback_provider": "modelslab",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.0,
        "max_tokens": 120,
    },

    # Heavyweight reasoning
    "meta_ranker": {
        "provider": "groq",
        "fallback_provider": "modelslab",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.0,
        "max_tokens": 1024,
    },
    "rag_synthesis": {
        "provider": "modelslab",
        "fallback_provider": "gemini",
        "model": "gemini-2.5-flash",
        "model_vision": "gemini-3-pro-image-preview",
        "temperature": 0.1,
        "max_tokens": 1500,
    },
    "coder_agent": {
        "provider": "modelslab",
        "fallback_provider": "gemini",
        "model": "qwen-qwen-2.5-coder-32b-instruct",
        "temperature": 0.1,
        "max_tokens": 1200,
    },
    "reward_scorer": {
        "provider": "groq",
        "fallback_provider": "modelslab",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 200,
    },
    "hallucination_verifier": {
        "provider": "groq",
        "fallback_provider": "gemini",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.0,
        "max_tokens": 600,
    },
    "smalltalk": {
        "provider": "groq",
        "fallback_provider": "modelslab",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.5,
        "max_tokens": 512,
    },
    "bootstrapper": {
        "provider": "modelslab",
        "fallback_provider": "gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
        "max_tokens": 1200,
    },
}


def get_phase_model(phase: str) -> Dict[str, Any]:
    if phase not in PHASE_MODELS:
        raise KeyError(f"Unknown phase model: {phase}")
    cfg = PHASE_MODELS[phase].copy()
    provider = cfg.get("provider")
    fallback = cfg.get("fallback_provider")

    if provider == "modelslab" and not os.getenv("MODELSLAB_API_KEY"):
        if fallback == "gemini" and os.getenv("GEMINI_API_KEY"):
            cfg["provider"] = "gemini"
        else:
            cfg["provider"] = "groq"

    return cfg
