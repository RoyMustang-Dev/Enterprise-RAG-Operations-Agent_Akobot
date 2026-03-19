"""
Composio Tools — v3 REST API Implementation

Provides Gmail OAuth connect URL generation and email action execution
via the Composio v3 REST API. Used by ModularOrchestrator V2.

Auth Flow (first time):
1. Agent calls get_gmail_connect_url(entity_id) → returns OAuth URL
2. User opens URL in browser → completes Gmail OAuth
3. On subsequent calls, is_gmail_connected(entity_id) returns True
4. Agent calls send_email(entity_id, ...) → executes action

Env vars required:
  COMPOSIO_API_KEY — from https://app.composio.dev/settings (ak_...)
"""
from __future__ import annotations

import os
import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# v3 API base (v1 is deprecated and returns 410)
_COMPOSIO_BASE = "https://backend.composio.dev/api/v3"
_COMPOSIO_BASE_V1 = "https://backend.composio.dev/api/v1"  # kept for action execution only


def is_configured() -> bool:
    return bool(os.getenv("COMPOSIO_API_KEY"))


def _headers() -> Dict[str, str]:
    return {
        "x-api-key": os.getenv("COMPOSIO_API_KEY", ""),
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Gmail connected account check (v1 — still works for reads)
# ---------------------------------------------------------------------------
def is_gmail_connected(entity_id: str) -> bool:
    """
    Checks if the entity has an active Gmail connected account.
    """
    if not is_configured():
        return False
    try:
        url = f"{_COMPOSIO_BASE_V1}/connectedAccounts"
        resp = requests.get(
            url,
            headers=_headers(),
            params={"entityId": entity_id, "showActiveOnly": "true"},
            timeout=10,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        accounts = data.get("items", data) if isinstance(data, dict) else data
        if isinstance(accounts, list):
            for acc in accounts:
                integration = (acc.get("integrationId") or "").lower()
                app_name = (acc.get("appName") or "").lower()
                status = (acc.get("status") or "").lower()
                if ("gmail" in integration or "gmail" in app_name) and status == "active":
                    return True
        return False
    except Exception as e:
        logger.warning(f"[COMPOSIO] is_gmail_connected check failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Get Gmail integration ID (v3 API)
# ---------------------------------------------------------------------------
def _get_gmail_integration_id() -> Optional[str]:
    """
    Fetches the UUID for the Gmail integration from Composio v3 API.
    Required because v3 changed integrationId to a UUID from a plain string.
    """
    try:
        url = f"{_COMPOSIO_BASE}/integrations"
        resp = requests.get(
            url,
            headers=_headers(),
            params={"appName": "gmail"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", data) if isinstance(data, dict) else data
            if isinstance(items, list) and items:
                integration_id = items[0].get("id") or items[0].get("integrationId")
                if integration_id:
                    logger.info(f"[COMPOSIO] Resolved Gmail integration ID: {integration_id}")
                    return str(integration_id)
        logger.warning(f"[COMPOSIO] Could not resolve Gmail integration ID: HTTP {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"[COMPOSIO] _get_gmail_integration_id failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Gmail OAuth connect URL (v3 API)
# ---------------------------------------------------------------------------
def get_gmail_connect_url(entity_id: str) -> str:
    """
    Initiates a Gmail OAuth connection and returns the URL the user must open.
    The URL is valid for ~10 minutes.
    Uses Composio v3 API.
    """
    if not is_configured():
        return "https://app.composio.dev/settings (COMPOSIO_API_KEY not set)"

    # Step 1: Resolve Gmail integration UUID
    integration_id = _get_gmail_integration_id()
    if not integration_id:
        return "https://app.composio.dev/ (Could not resolve Gmail integration ID — check Composio dashboard)"

    try:
        # v3 endpoint for initiating a connection
        url = f"{_COMPOSIO_BASE}/connectedAccounts"
        payload = {
            "integrationId": integration_id,
            "entityId": entity_id,
            "data": {},
            "redirectUri": os.getenv("COMPOSIO_REDIRECT_URI", "https://app.composio.dev/"),
        }
        resp = requests.post(url, headers=_headers(), json=payload, timeout=15)
        if resp.status_code in (200, 201):
            data = resp.json()
            connect_url = (
                data.get("redirectUrl")
                or data.get("connectionData", {}).get("redirectUrl")
                or data.get("url", "")
            )
            if connect_url:
                logger.info(f"[COMPOSIO] Generated Gmail connect URL for entity_id={entity_id}")
                return connect_url
            logger.warning(f"[COMPOSIO] No redirectUrl in response: {data}")
            return "https://app.composio.dev/ (no redirect URL returned)"
        logger.warning(f"[COMPOSIO] Connect URL request returned {resp.status_code}: {resp.text[:300]}")
        return f"https://app.composio.dev/ (error {resp.status_code} — check Composio dashboard)"
    except Exception as e:
        logger.error(f"[COMPOSIO] get_gmail_connect_url failed: {e}")
        return f"https://app.composio.dev/ (error: {e})"


# ---------------------------------------------------------------------------
# Email execution (v1 action execution still works)
# ---------------------------------------------------------------------------
def send_email(entity_id: str, to: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Sends an email via the connected Gmail account using Composio.
    Returns action result or error dict.
    """
    if not is_configured():
        return {"status": "error", "message": "Composio not configured."}
    if not to:
        return {"status": "error", "message": "Recipient email address (to) is required."}

    try:
        url = f"{_COMPOSIO_BASE_V1}/actions/execute"
        payload = {
            "actionName": "GMAIL_SEND_EMAIL",
            "entityId": entity_id,
            "requestBody": {
                "recipient_email": to,
                "subject": subject or "(no subject)",
                "body": body or "",
            },
        }
        resp = requests.post(url, headers=_headers(), json=payload, timeout=20)
        if resp.status_code in (200, 201):
            result = resp.json()
            logger.info(f"[COMPOSIO] Email sent to {to} via entity {entity_id}")
            return {"status": "success", "result": result}
        logger.warning(f"[COMPOSIO] send_email returned {resp.status_code}: {resp.text[:300]}")
        return {"status": "error", "message": f"Composio returned {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        logger.error(f"[COMPOSIO] send_email failed: {e}")
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Legacy stubs (kept for backward compatibility)
# ---------------------------------------------------------------------------
def get_composio_session(user_id: str) -> Optional[Dict[str, Any]]:
    if not is_configured():
        return None
    return {"user_id": user_id, "api_key": os.getenv("COMPOSIO_API_KEY")}


def discover_tools(user_id: str, toolkits: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not is_configured():
        return []
    try:
        url = f"{_COMPOSIO_BASE_V1}/actions/list/all"
        resp = requests.get(url, headers=_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("items", [])[:20]
    except Exception as e:
        logger.warning(f"[COMPOSIO] discover_tools failed: {e}")
    return []


def augment_context_with_composio(data: Dict[str, Any]) -> Dict[str, Any]:
    return data
