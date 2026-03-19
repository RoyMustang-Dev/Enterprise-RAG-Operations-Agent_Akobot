"""
Composio Tool Router Client (REST)

Implements in-chat auth link generation and tool execution using the Tool Router API.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional, List

import requests

logger = logging.getLogger(__name__)


class ComposioToolRouterClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("COMPOSIO_API_KEY")
        self.base_url = base_url or os.getenv("COMPOSIO_BASE_URL", "https://backend.composio.dev/api/v3")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "x-api-key": self.api_key}

    def create_session(self, user_id: str, toolkits: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        if not self.is_configured():
            return None
        payload: Dict[str, Any] = {"user_id": user_id}
        # Some API versions reject toolkits on session creation; omit for compatibility.
        try:
            resp = requests.post(
                f"{self.base_url}/tool_router/session",
                headers=self._headers(),
                json=payload,
                timeout=20,
            )
            if resp.status_code >= 400:
                logger.error(f"[COMPOSIO] create_session failed status={resp.status_code} body={resp.text[:500]}")
                return {"error": resp.text, "status_code": resp.status_code}
            return resp.json()
        except Exception as e:
            logger.error(f"[COMPOSIO] create_session failed: {e}")
            return None

    def search_tools(self, session_id: str, use_case: str, model: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"queries": [{"use_case": use_case}]}
        if model:
            payload["model"] = model
        resp = requests.post(
            f"{self.base_url}/tool_router/session/{session_id}/search",
            headers=self._headers(),
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    def execute_tool(self, session_id: str, tool_slug: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"tool_slug": tool_slug, "arguments": arguments}
        resp = requests.post(
            f"{self.base_url}/tool_router/session/{session_id}/execute",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def execute_meta(self, session_id: str, slug: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"slug": slug, "arguments": arguments}
        resp = requests.post(
            f"{self.base_url}/tool_router/session/{session_id}/execute_meta",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def create_link(self, session_id: str, toolkit: str, callback_url: Optional[str] = None) -> Dict[str, Any]:
        payload = {"toolkit": toolkit}
        if callback_url:
            payload["callback_url"] = callback_url
        resp = requests.post(
            f"{self.base_url}/tool_router/session/{session_id}/link",
            headers=self._headers(),
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
