"""
Adaptive Planner

Generates a minimal execution plan based on current state and context.
"""
from typing import Dict, Any


class AdaptivePlanner:
    """
    Determines which downstream components to invoke based on context.
    """

    def generate_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intent = state.get("intent") or "rag_question"
        intents = (state.get("optimizations", {}) or {}).get("intents") or [intent]
        complexity = float(state.get("optimizations", {}).get("complexity_score", 0.0))
        query = (state.get("query") or "").lower()
        tenant_id = state.get("tenant_id")
        extra_collections = state.get("extra_collections") or []

        rag_hints = [
            "source", "sources", "document", "documents", "files", "uploaded", "knowledge base",
            "in the docs", "in the file", "in the pdf", "in the csv", "in the report",
        ]
        should_force_rag = bool(tenant_id) or bool(extra_collections) or any(h in query for h in rag_hints)

        # Reasoning effort tiers
        if complexity >= 0.8:
            effort = "high"
        elif complexity >= 0.5:
            effort = "medium"
        else:
            effort = "low"

        # Route selection
        if "out_of_scope" in intents:
            route = "out_of_scope"
            run_rewriter = False
        elif any(i in ["greeting", "smalltalk"] for i in intents):
            route = "smalltalk"
            run_rewriter = False
        elif "crm_request" in intents:
            route = "support_da"
            run_rewriter = True
        elif "email_request" in intents:
            # If there is a primary RAG intent plus email, still run RAG and handle email after.
            if any(i in ["rag_question", "code_request", "analytics_request"] for i in intents):
                route = "rag_agent"
                run_rewriter = True
            else:
                route = "support_da"
                run_rewriter = True
        elif intent in ["code_request", "analytics_request"]:
            if should_force_rag:
                route = "rag_agent"
                run_rewriter = True
            else:
                route = "coder_agent"
                run_rewriter = True
        else:
            route = "rag_agent"
            run_rewriter = True

        # Model override for high complexity
        override_model = "openai/gpt-oss-120b" if complexity >= 0.8 else None

        return {
            "route": route,
            "reasoning_effort": effort,
            "override_model": override_model,
            "run_rewriter": run_rewriter,
        }
