"""
Composio MCP Integration (Skeleton)

Provides a minimal wrapper for Composio MCP usage without hard dependency on SDK.
If COMPOSIO_MCP_URL is configured, tools can be invoked via MCP server URL.
"""
from __future__ import annotations

import os
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ComposioMCPClient:
    def __init__(self, api_key: Optional[str] = None, mcp_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("COMPOSIO_API_KEY")
        self.mcp_url = mcp_url or os.getenv("COMPOSIO_MCP_URL")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.mcp_url)

    def get_headers(self) -> Dict[str, str]:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Skeleton MCP call. Requires a valid COMPOSIO_MCP_URL that supports POST /tools/execute or equivalent.
        """
        if not self.is_configured():
            return {"successful": False, "error": "Composio MCP not configured", "data": {}}

        try:
            payload = {"tool": tool_name, "arguments": arguments, "user_id": user_id}
            resp = requests.post(
                self.mcp_url,
                headers=self.get_headers(),
                json=payload,
                timeout=20,
            )
            if resp.status_code >= 400:
                return {"successful": False, "error": resp.text, "data": {}}
            return resp.json()
        except Exception as e:
            logger.error(f"[COMPOSIO MCP] Tool call failed: {e}")
            return {"successful": False, "error": str(e), "data": {}}
