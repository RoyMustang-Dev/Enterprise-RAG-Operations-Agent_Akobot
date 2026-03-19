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
from app.core.types import AgentState
from app.tools.emailer import send_email_via_composio
from app.infra.llm_client import set_agent_context
from app.integrations.composio_tool_router import ComposioToolRouterClient
from app.supervisor.intent import IntentClassifier
from app.infra.database import get_session_cache, upsert_session_cache

logger = logging.getLogger(__name__)


class SupportDataAnalyticsAgent:
    def __init__(self):
        self.guard = PromptInjectionGuard()
        self.rewriter = PromptRewriter()
        self.smalltalk = SmalltalkAgent()
        self.rag = RAGAgent()
        self.intent_classifier = IntentClassifier()

    @staticmethod
    def _extract_email_fields(text: str) -> Dict[str, Any]:
        import re
        if not text:
            return {"recipients": [], "subject": "", "body": ""}
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        subject_match = re.search(r"subject\s*:\s*(.+)", text, re.IGNORECASE)
        body_match = re.search(r"body\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        subject = subject_match.group(1).strip() if subject_match else ""
        body = body_match.group(1).strip() if body_match else ""
        return {"recipients": emails, "subject": subject, "body": body}

    @staticmethod
    def _infer_crm_toolkit(text: str) -> str:
        lowered = (text or "").lower()
        for name in ["hubspot", "salesforce", "zendesk", "freshdesk"]:
            if name in lowered:
                return name
        return "hubspot"

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        if not text:
            return {}
        import re
        import json
        # Try fenced json block
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if not match:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
        if not match:
            return {}
        raw = match.group(1)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    @staticmethod
    def _is_done(text: str) -> bool:
        if not text:
            return False
        lowered = text.strip().lower()
        return lowered in {"done", "connected", "completed", "ok", "okay", "yes, done"}

    async def run(
        self,
        query: str,
        user_id: str = "default",
        tenant_id: str = "default",
        session_id: str = "support-da",
        chat_history: list | None = None,
        dataframes: list | None = None,
        sources: list | None = None,
        persona: str | None = None,
    ) -> Dict[str, Any]:
        """
        Skeleton router:
        - guard + rewrite
        - choose between smalltalk, RAG, BA, email tool
        """
        set_agent_context("support_da")
        original_query = query
        rewritten = (await self.rewriter.rewrite(query, "unknown")).get("prompts", {}).get("standard_med", {}).get("prompt", query)
        safety = self.guard.evaluate(rewritten)
        if safety.get("is_malicious"):
            return {"status": "blocked", "reason": "Prompt guard blocked the request."}

        q_raw = (original_query or "").lower()
        q = (rewritten or "").lower()

        # Detect and retry pending email after user confirms "Done"
        if self._is_done(original_query):
            cached = get_session_cache(session_id=session_id, tenant_id=tenant_id) or {}
            pending = (cached.get("cache") or {}).get("pending_email")
            if pending:
                if not pending.get("to"):
                    return {"status": "needs_input", "note": "Please provide the recipient email address to send the results."}
                email_result = send_email_via_composio(
                    user_id=user_id,
                    to=pending.get("to", []),
                    subject=pending.get("subject", "Requested insights"),
                    body=pending.get("body", "")
                )
                upsert_session_cache(session_id, {"pending_email": pending}, tenant_id=tenant_id)
                return {
                    "status": "success" if email_result.get("successful") else "auth_required",
                    "answer": "Email sent." if email_result.get("successful") else "Authentication required to send email.",
                    "email_action": email_result,
                }

        # Intent classification (LLM-based)
        intent_report = await self.intent_classifier.classify(rewritten or query)
        intents = intent_report.get("intents") or [intent_report.get("intent", "rag_question")]
        if self._extract_email_fields(original_query).get("recipients"):
            if "email_request" not in intents:
                intents.append("email_request")
        if "crm" in q_raw or any(name in q_raw for name in ["hubspot", "salesforce", "zendesk", "freshdesk"]):
            if "crm_request" not in intents:
                intents.append("crm_request")

        # CRM flow
        if "crm_request" in intents:
            # Use Composio Tool Router to discover CRM tools and return connect link if needed.
            client = ComposioToolRouterClient()
            if not client.is_configured():
                return {"status": "error", "note": "Composio API key not configured."}
            session = client.create_session(user_id=user_id)
            if not session:
                return {"status": "error", "note": "Failed to create Composio session."}
            if session.get("error"):
                return {"status": "error", "note": f"Composio session error: {session.get('error')}"}
            if not session.get("session_id"):
                return {"status": "error", "note": "Failed to create Composio session."}
            composio_session_id = session["session_id"]
            try:
                search = client.search_tools(composio_session_id, rewritten or query)
                results = search.get("results") or []
                if not results:
                    return {"status": "no_tools", "note": "No CRM tools found for this request."}
                tool_info = results[0]
                connection_status = tool_info.get("toolkit_connection_statuses") or []
                for status in connection_status:
                    toolkit = status.get("toolkit")
                    if not status.get("has_active_connection"):
                        if not toolkit:
                            toolkit = self._infer_crm_toolkit(original_query or query)
                        link = client.create_link(composio_session_id, toolkit) if toolkit else {}
                        resp = {
                            "status": "auth_required",
                            "note": "Please connect your CRM account using this link, then confirm in chat.",
                            "connect_link": link.get("url") or link.get("link") or link,
                            "toolkit": toolkit,
                            "tool_suggestion": tool_info.get("tool_slug"),
                        }
                        if "email_request" in intents:
                            resp["email_action"] = self._prepare_email_action(
                                original_query=original_query,
                                tenant_id=tenant_id,
                                session_id=session_id,
                                fallback_body=resp.get("note", "")
                            )
                        return resp
                # If already connected, allow explicit execution when user provides JSON args.
                toolkit = (connection_status[0].get("toolkit") if connection_status else None)
                if not toolkit:
                    toolkit = self._infer_crm_toolkit(original_query or query)
                link = client.create_link(composio_session_id, toolkit) if toolkit else {}
                payload = self._extract_json_payload(original_query or query)
                tool_slug = payload.get("tool_slug") or tool_info.get("tool_slug")
                arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else None
                if tool_slug and arguments:
                    try:
                        from app.infra.locks import distributed_lock
                        from app.infra.database import record_tool_audit
                        lock_key = f"crm:{tenant_id}:{tool_slug}"
                        with distributed_lock(lock_key, ttl_seconds=60, wait_timeout=0) as acquired:
                            if not acquired:
                                raise RuntimeError("CRM tool execution already in progress.")
                            executed = client.execute_tool(composio_session_id, tool_slug, arguments)
                        record_tool_audit(tenant_id or "default", tool_slug, "success", "")
                        resp = {
                            "status": "executed",
                            "note": "CRM tool executed successfully.",
                            "toolkit": toolkit,
                            "tool_slug": tool_slug,
                            "result": executed,
                        }
                    except Exception as e:
                        try:
                            from app.infra.database import record_tool_audit
                            record_tool_audit(tenant_id or "default", tool_slug, "failed", str(e))
                        except Exception:
                            pass
                        resp = {
                            "status": "error",
                            "note": f"CRM tool execution failed: {e}",
                            "toolkit": toolkit,
                            "tool_slug": tool_slug,
                        }
                else:
                    resp = {
                        "status": "ready",
                        "note": "CRM connection detected. Provide a JSON payload to execute (tool_slug + arguments).",
                        "connect_link": link.get("url") or link.get("link") or link,
                        "toolkit": toolkit,
                        "tool_suggestion": tool_info.get("tool_slug"),
                        "tool_schema": tool_info.get("input_schema") or tool_info.get("input_parameters"),
                        "example_payload": {
                            "tool_slug": tool_info.get("tool_slug"),
                            "arguments": (tool_info.get("input_schema") or tool_info.get("input_parameters") or {})
                        },
                    }
                if "email_request" in intents:
                    resp["email_action"] = self._prepare_email_action(
                        original_query=original_query,
                        tenant_id=tenant_id,
                        session_id=session_id,
                        fallback_body=resp.get("note", "")
                    )
                return resp
            except Exception as e:
                return {"status": "error", "note": f"Composio search failed: {e}"}

        # Smalltalk
        if any(i in ["greeting", "smalltalk"] for i in intents):
            return await self.smalltalk.ainvoke({"query": rewritten, "chat_history": [], "optimizations": {}, "active_persona": None})

        # BA intent (requires dataframes if provided)
        if "analytics_request" in intents and dataframes:
            supervisor = DataAnalyticsSupervisor(dataframes=dataframes, sources=sources)
            payload = await supervisor.run(query=rewritten or query, persona=persona or "", session_id=session_id, tenant_id=tenant_id)
            result = {"status": "success", "agent": "SUPPORT_DATA_ANALYTICS", "data": payload}
        elif "analytics_request" in intents and not dataframes:
            result = {"status": "not_executed", "note": "Business Analytics requires CSV/Excel files. Upload datasets to proceed."}
        else:
            # Default to RAG for knowledge queries
            state: AgentState = {
                "session_id": session_id,
                "tenant_id": tenant_id,
                "query": rewritten,
                "chat_history": chat_history or [],
                "streaming_callback": None,
                "intent": "rag",
                "search_query": None,
                "context_chunks": [],
                "context_text": "",
                "extra_collections": [],
                "retrieval_scope": "kb_only",
                "confidence": 0.0,
                "verifier_verdict": "PENDING",
                "is_hallucinated": False,
                "verification_claims": [],
                "optimizations": {},
                "optimized_prompts": {},
                "reasoning_effort": "low",
                "latency_optimizations": {},
                "active_persona": None,
                "answer": "",
                "sources": [],
            }
            rag_result = await self.rag.ainvoke(state)
            result = {
                "status": "success",
                "agent": "SUPPORT_DATA_ANALYTICS",
                "data": {
                    "answer": rag_result.get("answer"),
                    "sources": rag_result.get("sources", []),
                    "confidence": rag_result.get("confidence", 0.0),
                    "verifier_verdict": rag_result.get("verifier_verdict", "PENDING"),
                    "optimizations": rag_result.get("optimizations", {}),
                    "latency_optimizations": rag_result.get("latency_optimizations", {}),
                },
            }

        # Email intent post-processing
        if "email_request" in intents:
            result.setdefault(
                "email_action",
                self._prepare_email_action(
                    original_query=original_query,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    fallback_body="",
                    result_payload=result,
                ),
            )

        return result

    def _prepare_email_action(
        self,
        original_query: str,
        tenant_id: str,
        session_id: str,
        fallback_body: str,
        result_payload: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        email_fields = self._extract_email_fields(original_query or "")
        recipients = email_fields.get("recipients") or []
        subject = email_fields.get("subject") or "Requested insights"
        body = email_fields.get("body")
        if not body and result_payload and isinstance(result_payload, dict) and "data" in result_payload and isinstance(result_payload["data"], dict):
            body = result_payload["data"].get("summary_paragraph") or result_payload["data"].get("answer") or ""
        if not body:
            body = fallback_body or ""
        if recipients:
            email_result = send_email_via_composio(
                user_id=tenant_id or "default",
                to=recipients,
                subject=subject,
                body=body or ""
            )
            if not email_result.get("successful"):
                upsert_session_cache(session_id, {"pending_email": {"to": recipients, "subject": subject, "body": body or ""}}, tenant_id=tenant_id)
            return email_result
        upsert_session_cache(session_id, {"pending_email": {"to": [], "subject": subject, "body": body or ""}}, tenant_id=tenant_id)
        return {"successful": False, "error": "Missing recipient email. Please provide an email address to send the results."}
