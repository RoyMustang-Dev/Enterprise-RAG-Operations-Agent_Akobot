"""
Support Data Analytics Agent (Skeleton)

This agent is NOT wired into routing yet. It provides a blueprint for a multi-tool
agent that can call smalltalk, prompt guard, prompt rewriter, RAG, BA, and auto emailer.
"""
from __future__ import annotations

from typing import Dict, Any
import logging

from app.prompt_engine.guard import PromptInjectionGuard
from app.prompt_engine.rewriter import PromptRewriter
from app.agents.smalltalk import SmalltalkAgent
from app.agents.rag import RAGAgent
from app.agents.data_analytics.supervisor import DataAnalyticsSupervisor
from app.tools.emailer import send_email_via_composio

logger = logging.getLogger(__name__)


class SupportDataAnalyticsAgent:
    def __init__(self):
        self.guard = PromptInjectionGuard()
        self.rewriter = PromptRewriter()
        self.smalltalk = SmalltalkAgent()
        self.rag = RAGAgent()
        self.ba = DataAnalyticsSupervisor([])

    async def run(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Skeleton router:
        - guard + rewrite
        - choose between smalltalk, RAG, BA, email tool
        """
        rewritten = (await self.rewriter.rewrite(query, "unknown")).get("prompts", {}).get("standard_med", {}).get("prompt", query)
        safety = self.guard.evaluate(rewritten)
        if safety.get("is_malicious"):
            return {"status": "blocked", "reason": "Prompt guard blocked the request."}

        q = (rewritten or "").lower()
        if "email" in q and "send" in q:
            return send_email_via_composio(
                user_id=user_id,
                to=["example@example.com"],
                subject="Placeholder subject",
                body="Placeholder body"
            )
        if "trend" in q or "forecast" in q or "analysis" in q:
            return {"status": "not_executed", "note": "BA requires dataset input; wire dataset here."}
        if "hello" in q or "hi" in q:
            return await self.smalltalk.ainvoke({"query": rewritten, "chat_history": [], "optimizations": {}, "active_persona": None})

        return {"status": "not_executed", "note": "RAG requires full AgentState; wire orchestrator here."}
