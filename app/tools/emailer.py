"""
Auto Email Tool (Composio MCP Skeleton)

Provides a minimal email tool. Requires COMPOSIO_MCP_URL + COMPOSIO_API_KEY to execute.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
from app.integrations.composio_mcp import ComposioMCPClient


def send_email_via_composio(
    user_id: str,
    to: List[str],
    subject: str,
    body: str,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Sends an email using Composio MCP (tool name: GMAIL_SEND_EMAIL).
    """
    client = ComposioMCPClient()
    if not client.is_configured():
        return {
            "successful": False,
            "error": "Composio MCP not configured. Set COMPOSIO_MCP_URL and COMPOSIO_API_KEY.",
            "data": {},
        }
    args = {
        "to": to,
        "subject": subject,
        "body": body,
        "cc": cc or [],
        "bcc": bcc or [],
    }
    return client.call_tool("GMAIL_SEND_EMAIL", args, user_id=user_id)
