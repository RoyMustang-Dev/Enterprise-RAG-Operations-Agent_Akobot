"""
Provider Router

Selects the best available LLM provider based on env keys and routing rules.
Designed to keep current default behavior unless explicitly enabled.
"""
import os
import logging

logger = logging.getLogger(__name__)


class ProviderRouter:
    def __init__(self):
        self.providers = {
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
        }
        # Auto-routing is decided per-request (requested == "auto").
        # Keep priority order in code to avoid extra .env knobs.
        self.preferred_order = ["groq", "openai", "anthropic", "gemini"]

    def select_provider(self, requested: str | None) -> str:
        req = (requested or "groq").lower()

        # Explicit provider request
        if req != "auto":
            if self.providers.get(req):
                return req
            if req != "groq":
                logger.warning(f"[PROVIDER] {req} requested but API key missing. Falling back to groq.")
            return "groq"

        # Auto routing (per-request)

        for p in self.preferred_order:
            if self.providers.get(p):
                return p

        logger.warning("[PROVIDER] Auto-routing enabled but no provider keys found. Falling back to groq.")
        return "groq"
