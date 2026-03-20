"""
Auto Email Tool (Composio Tool Router)

Provides an email tool. Requires COMPOSIO_API_KEY to execute.
If the user has not connected Gmail/Outlook yet, returns a connect link.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
import hashlib
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
    try:
        policy = check_email_policy(tenant_id, to or [])
    except Exception as e:
        # Fail closed but do not crash email execution
        return {"successful": False, "error": f"Email policy check failed: {e}", "data": {}}
    if not policy.get("allowed"):
        try:
            audit_email(tenant_id, to or [], subject, "blocked", policy.get("error", "policy_block"))
        except Exception:
            pass
        return {"successful": False, "error": policy.get("error", "Email policy blocked."), "data": {}}
    # Validate recipients early
    # Normalize recipients: flatten lists and split comma/semicolon strings
    normalized: List[str] = []
    for item in (to or []):
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, str):
                    normalized.extend([s.strip() for s in sub.replace(";", ",").split(",") if s.strip()])
                elif isinstance(sub, dict):
                    for k in ("email", "address", "value"):
                        v = sub.get(k)
                        if isinstance(v, str) and v.strip():
                            normalized.extend([s.strip() for s in v.replace(";", ",").split(",") if s.strip()])
        elif isinstance(item, dict):
            for k in ("email", "address", "value"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    normalized.extend([s.strip() for s in v.replace(";", ",").split(",") if s.strip()])
        elif isinstance(item, str):
            normalized.extend([s.strip() for s in item.replace(";", ",").split(",") if s.strip()])
    cleaned_to = [t for t in normalized if t]
    if not cleaned_to:
        try:
            audit_email(tenant_id, [], subject, "failed", "Invalid or empty recipient list.")
        except Exception:
            pass
        return {"successful": False, "error": "Invalid or empty recipient list.", "data": {}}
    import re as _re
    email_re = _re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
    invalid = [e for e in cleaned_to if not email_re.match(e)]
    if invalid:
        try:
            audit_email(tenant_id, cleaned_to, subject, "failed", f"Invalid email: {', '.join(invalid)}")
        except Exception:
            pass
        return {"successful": False, "error": f"Invalid email: {', '.join(invalid)}", "data": {}}
    to = cleaned_to

    body_sig = hashlib.sha1((body or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
    lock_key = f"email:{tenant_id}:{','.join(to or [])}:{subject}:{body_sig}"
    with distributed_lock(lock_key, ttl_seconds=300, wait_timeout=0) as acquired:
        if not acquired:
            return {"successful": False, "error": "Email dispatch already in progress. Try again shortly.", "data": {}}

    client = ComposioToolRouterClient()
    if not client.is_configured():
        try:
            audit_email(tenant_id, to or [], subject, "failed", "Composio not configured. Set COMPOSIO_API_KEY.")
        except Exception:
            pass
        return {
            "successful": False,
            "error": "Composio not configured. Set COMPOSIO_API_KEY.",
            "data": {},
        }
    session = client.create_session(user_id=user_id, toolkits=["gmail", "outlook"])
    if not session:
        try:
            audit_email(tenant_id, to or [], subject, "failed", "Failed to create Composio session.")
        except Exception:
            pass
        return {"successful": False, "error": "Failed to create Composio session.", "data": {}}
    if session.get("error"):
        try:
            audit_email(tenant_id, to or [], subject, "failed", f"Composio session error: {session.get('error')}")
        except Exception:
            pass
        return {"successful": False, "error": f"Composio session error: {session.get('error')}", "data": {}}
    session_id = session.get("session_id")
    if not session_id:
        try:
            audit_email(tenant_id, to or [], subject, "failed", "Composio session_id missing.")
        except Exception:
            pass
        return {"successful": False, "error": "Composio session_id missing.", "data": {}}

    # Search best tool for sending email
    try:
        search = client.search_tools(session_id, "send an email")
        results = (search.get("results") or [])
        if not results:
            try:
                audit_email(tenant_id, to or [], subject, "failed", "No email tool found in Composio.")
            except Exception:
                pass
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
            try:
                audit_email(tenant_id, to or [], subject, "failed", "Tool slug missing from Composio search.")
            except Exception:
                pass
            return {"successful": False, "error": "Tool slug missing from Composio search.", "data": {}}
    except Exception as e:
        try:
            audit_email(tenant_id, to or [], subject, "failed", f"Composio tool search failed: {e}")
        except Exception:
            pass
        return {"successful": False, "error": f"Composio tool search failed: {e}", "data": {}}

    # Build args based on tool schema to avoid Composio validation errors.
    # Sanitize body to remove placeholders, sample templates, and stray markdown fences
    cleaned_body = (body or "").replace("```", "").strip()
    cleaned_body = cleaned_body.replace("[Your Name]", "").replace("[Your name]", "").strip()
    cleaned_body = cleaned_body.replace("\n\nBest regards,\n", "\n\n").replace("\nBest regards,\n", "\n")
    # Remove common "cannot send email" disclaimers
    for bad in [
        "I cannot send",
        "I can't send",
        "I cant send",
        "do not have the capability to send emails",
        "don't have the capability to send emails",
        "cannot send emails directly",
    ]:
        cleaned_body = cleaned_body.replace(bad, "")
    # Drop any inline "sample email" section
    for marker in ["### Email to", "Since this is a text-based AI model", "**Subject:"]:
        idx = cleaned_body.find(marker)
        if idx != -1:
            cleaned_body = cleaned_body[:idx].strip()
            break

    args = {
        "subject": subject,
        "body": cleaned_body,
        "cc": cc or [],
        "bcc": bcc or [],
    }
    try:
        schema = client.execute_meta(session_id, "COMPOSIO_GET_TOOL_SCHEMAS", {"tool_slugs": [tool_slug]})
        tool_schema = (schema.get("data") or {}).get("tool_schemas", {}).get(tool_slug, {})
        props = ((tool_schema.get("input_schema") or {}).get("properties") or {})
    except Exception:
        props = {}

    # Gmail expects recipient_email (string) + optional extra_recipients array
    if "recipient_email" in props and "to" not in props:
        primary = (to or [""])[0] if isinstance(to, list) else (to or "")
        extras = []
        if isinstance(to, list) and len(to) > 1:
            extras = to[1:]
        args["recipient_email"] = primary
        if "extra_recipients" in props:
            args["extra_recipients"] = extras
    else:
        # Generic OpenAI-style tools expect `to` array
        args["to"] = to or []
    try:
        result = client.execute_tool(session_id, tool_slug, args)
        # Composio may return error payloads with 200 responses
        err = ""
        if isinstance(result, dict):
            err = result.get("error") or result.get("message") or ""
            # Some tool router responses nest error in data
            if not err and isinstance(result.get("data"), dict):
                err = result["data"].get("error") or result["data"].get("message") or ""
        if err:
            try:
                audit_email(tenant_id, to or [], subject, "failed", str(err))
            except Exception:
                pass
            return {"successful": False, "error": str(err), "data": result}
        try:
            audit_email(tenant_id, to or [], subject, "success", "")
        except Exception:
            pass
        return {"successful": True, "data": result, "error": ""}
    except Exception as e:
        # If execution fails, return connect link for the first toolkit
        toolkit_for_link = toolkits[0] if toolkits else "gmail"
        try:
            link = client.create_link(session_id, toolkit_for_link)
            try:
                audit_email(tenant_id, to or [], subject, "auth_required", "Authentication required.")
            except Exception:
                pass
            return {
                "successful": False,
                "error": "Authentication required.",
                "data": {
                    "connect_link": link.get("redirect_url") or link.get("url") or link.get("link") or link,
                    "toolkit": toolkit_for_link,
                },
            }
        except Exception:
            try:
                audit_email(tenant_id, to or [], subject, "failed", str(e))
            except Exception:
                pass
            return {"successful": False, "error": str(e), "data": {}}
