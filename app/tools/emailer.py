"""
Auto Email Tool (Composio Tool Router)

Provides an email tool. Requires COMPOSIO_API_KEY to execute.
If the user has not connected Gmail/Outlook yet, returns a connect link.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
from app.integrations.composio_tool_router import ComposioToolRouterClient
from app.infra.email_policy import check_email_policy, audit_email
from app.infra.locks import distributed_lock


def send_email_via_composio(
    user_id: str,
    to: List[str],
    subject: str,
    body: str,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Sends an email using Composio Tool Router.
    If the user is not connected, returns a connect link.
    """
    tenant_id = user_id or "default"
    policy = check_email_policy(tenant_id, to or [])
    if not policy.get("allowed"):
        audit_email(tenant_id, to or [], subject, "blocked", policy.get("error", "policy_block"))
        return {"successful": False, "error": policy.get("error", "Email policy blocked."), "data": {}}

    lock_key = f"email:{tenant_id}:{','.join(to or [])}:{subject}"
    with distributed_lock(lock_key, ttl_seconds=60, wait_timeout=0) as acquired:
        if not acquired:
            return {"successful": False, "error": "Email dispatch already in progress. Try again shortly.", "data": {}}

    client = ComposioToolRouterClient()
    if not client.is_configured():
        audit_email(tenant_id, to or [], subject, "failed", "Composio not configured. Set COMPOSIO_API_KEY.")
        return {
            "successful": False,
            "error": "Composio not configured. Set COMPOSIO_API_KEY.",
            "data": {},
        }
    session = client.create_session(user_id=user_id, toolkits=["gmail", "outlook"])
    if not session:
        audit_email(tenant_id, to or [], subject, "failed", "Failed to create Composio session.")
        return {"successful": False, "error": "Failed to create Composio session.", "data": {}}
    if session.get("error"):
        audit_email(tenant_id, to or [], subject, "failed", f"Composio session error: {session.get('error')}")
        return {"successful": False, "error": f"Composio session error: {session.get('error')}", "data": {}}
    session_id = session.get("session_id")
    if not session_id:
        audit_email(tenant_id, to or [], subject, "failed", "Composio session_id missing.")
        return {"successful": False, "error": "Composio session_id missing.", "data": {}}

    # Search best tool for sending email
    try:
        search = client.search_tools(session_id, "send an email")
        results = (search.get("results") or [])
        if not results:
            audit_email(tenant_id, to or [], subject, "failed", "No email tool found in Composio.")
            return {"successful": False, "error": "No email tool found in Composio.", "data": {}}
        tool_info = results[0]
        tool_slug = None
        primary = tool_info.get("primary_tool_slugs") or []
        if primary:
            tool_slug = primary[0]
        if not tool_slug:
            tools = tool_info.get("tools") or []
            tool_slug = tools[0].get("slug") if tools else None
        toolkits = tool_info.get("toolkits") or []
        if not tool_slug:
            audit_email(tenant_id, to or [], subject, "failed", "Tool slug missing from Composio search.")
            return {"successful": False, "error": "Tool slug missing from Composio search.", "data": {}}
    except Exception as e:
        audit_email(tenant_id, to or [], subject, "failed", f"Composio tool search failed: {e}")
        return {"successful": False, "error": f"Composio tool search failed: {e}", "data": {}}

    args = {
        "to": to,
        "subject": subject,
        "body": body,
        "cc": cc or [],
        "bcc": bcc or [],
    }
    try:
        result = client.execute_tool(session_id, tool_slug, args)
        audit_email(tenant_id, to or [], subject, "success", "")
        return {"successful": True, "data": result, "error": ""}
    except Exception as e:
        # If execution fails, return connect link for the first toolkit
        toolkit_for_link = toolkits[0] if toolkits else "gmail"
        try:
            link = client.create_link(session_id, toolkit_for_link)
            audit_email(tenant_id, to or [], subject, "auth_required", "Authentication required.")
            return {
                "successful": False,
                "error": "Authentication required.",
                "data": {
                    "connect_link": link.get("redirect_url") or link.get("url") or link.get("link") or link,
                    "toolkit": toolkit_for_link,
                },
            }
        except Exception:
            audit_email(tenant_id, to or [], subject, "failed", str(e))
            return {"successful": False, "error": str(e), "data": {}}
