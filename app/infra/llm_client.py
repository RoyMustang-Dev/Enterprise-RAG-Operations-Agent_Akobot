"""
LLM Client

Unified helper for chat-completions across providers.
Currently supports: modelslab, gemini, groq.
"""
from __future__ import annotations

import os
import asyncio
import json
import time
import logging
import contextvars
import threading
from typing import Any, Dict, List, Optional

import aiohttp
import requests

from app.infra.circuit_breaker import get_breaker

logger = logging.getLogger(__name__)

_AGENT_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar("llm_agent_context", default="global")


def set_agent_context(agent: str) -> None:
    if agent:
        _AGENT_CONTEXT.set(agent)


def get_agent_context() -> str:
    return _AGENT_CONTEXT.get()

_MODELSLAB_TEMP_FIXED = {
    # ModelsLab constraint: gpt-5-mini only supports temperature=1
    "gpt-5-mini": 1,
}

_MODELSLAB_RESPONSE_FORMAT_BLOCKLIST = {
    # Observed ModelsLab 200+error when response_format is used with these models
    "qwen-qwen3.5-122b-a10b",
    "qwen-qwen3.5-35b-a3b",
}

_PROVIDER_STREAM_SUPPORT = {
    "modelslab": True,
    "groq": True,
    "gemini": True,
    "openai": True,
    "anthropic": True,
}

_provider_sync_semaphores: Dict[str, threading.Semaphore] = {}
_provider_async_semaphores: Dict[str, asyncio.Semaphore] = {}


def provider_supports_stream(provider: str) -> bool:
    return _PROVIDER_STREAM_SUPPORT.get((provider or "").lower(), False)


def _get_sync_semaphore(provider: str) -> threading.Semaphore:
    provider = (provider or "").lower()
    if provider not in _provider_sync_semaphores:
        limit = int(os.getenv(f"{provider.upper()}_BULKHEAD_LIMIT", "8"))
        _provider_sync_semaphores[provider] = threading.Semaphore(limit)
    return _provider_sync_semaphores[provider]


def _get_async_semaphore(provider: str) -> asyncio.Semaphore:
    provider = (provider or "").lower()
    if provider not in _provider_async_semaphores:
        limit = int(os.getenv(f"{provider.upper()}_BULKHEAD_LIMIT", "8"))
        _provider_async_semaphores[provider] = asyncio.Semaphore(limit)
    return _provider_async_semaphores[provider]


def _ensure_json_keyword(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ensure at least one message contains the word 'json' (lowercase) for json_object requests."""
    for msg in messages:
        if "json" in (msg.get("content") or "").lower():
            return messages

    # Prefer to append to an existing system message for minimal disruption
    patched = [dict(m) for m in messages]
    for msg in patched:
        if msg.get("role") == "system":
            msg["content"] = (msg.get("content") or "") + "\nReturn valid json."
            return patched

    # No system message found; prepend a small system instruction
    return [{"role": "system", "content": "Return valid json."}] + patched


def _apply_modelslab_constraints(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    response_format: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Apply ModelsLab model-specific constraints:
    - Some models only support temperature=1.
    - Some models reject response_format and return 200+error.
    - response_format requires 'json' keyword in messages.
    """
    adjusted = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
    }

    if model in _MODELSLAB_TEMP_FIXED:
        adjusted["temperature"] = _MODELSLAB_TEMP_FIXED[model]

    if response_format and model in _MODELSLAB_RESPONSE_FORMAT_BLOCKLIST:
        adjusted["response_format"] = None
        adjusted["messages"] = _ensure_json_keyword(messages)
        return adjusted

    if response_format:
        adjusted["messages"] = _ensure_json_keyword(messages)

    return adjusted


def _provider_key(provider: str) -> Optional[str]:
    provider = (provider or "").lower()
    if provider == "modelslab":
        return os.getenv("MODELSLAB_API_KEY")
    if provider == "groq":
        return os.getenv("GROQ_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY")
    return None


def _endpoint_and_payload(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    response_format: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    provider = (provider or "").lower()

    if provider == "modelslab":
        constrained = _apply_modelslab_constraints(model, messages, temperature, response_format)
        model = constrained["model"]
        messages = constrained["messages"]
        temperature = constrained["temperature"]
        response_format = constrained["response_format"]
        payload: Dict[str, Any] = {
            "key": _provider_key("modelslab"),
            "model_id": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format
        return {
            "url": "https://modelslab.com/api/v7/llm/chat/completions",
            "headers": {"Content-Type": "application/json"},
            "payload": payload,
            "adapter": "modelslab",
        }

    if provider == "gemini":
        # Map chat messages into a single prompt for Gemini generateContent.
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        prompt = "\n".join(parts)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        return {
            "url": f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={_provider_key('gemini')}",
            "headers": {"Content-Type": "application/json"},
            "payload": payload,
            "adapter": "gemini",
        }

    # Default: Groq OpenAI-compatible API
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    if response_format:
        payload["response_format"] = response_format

    return {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "headers": {"Authorization": f"Bearer {_provider_key('groq')}", "Content-Type": "application/json"},
        "payload": payload,
    }


async def achat_completion(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    key = _provider_key(provider)
    if not key:
        raise RuntimeError(f"API key missing for provider: {provider}")

    breaker = get_breaker(
        provider,
        int(os.getenv("LLM_CB_FAILS", "3")),
        int(os.getenv("LLM_CB_RECOVERY_SEC", "30")),
        agent=get_agent_context(),
    )
    if not breaker.allow():
        raise RuntimeError(f"Circuit breaker open for provider: {provider}")

    req = _endpoint_and_payload(provider, model, messages, temperature, max_tokens, response_format)
    groq_retries = int(os.getenv("GROQ_RETRY_MAX", "3")) if provider.lower() == "groq" else 1
    backoff = float(os.getenv("GROQ_RETRY_BACKOFF_SEC", "1.5"))
    try:
        async with aiohttp.ClientSession() as session:
            for attempt in range(groq_retries):
                async with _get_async_semaphore(provider):
                    async with session.post(req["url"], headers=req["headers"], json=req["payload"], timeout=timeout) as resp:
                        if provider.lower() == "groq" and resp.status in (429, 500, 502, 503, 504):
                            if attempt < groq_retries - 1:
                                await asyncio.sleep(backoff * (2 ** attempt))
                                continue
                        resp.raise_for_status()
                        data = await resp.json()
                    if req.get("adapter") == "modelslab" and isinstance(data, dict) and data.get("status") == "error":
                        msg = data.get("message", "")
                        if "temperature" in msg and "default (1)" in msg:
                            retry_payload = dict(req["payload"])
                            retry_payload["temperature"] = 1
                            async with session.post(req["url"], headers=req["headers"], json=retry_payload, timeout=timeout) as retry_resp:
                                retry_resp.raise_for_status()
                                data = await retry_resp.json()
                        else:
                            if "out of credits" in msg.lower() and os.getenv("GEMINI_API_KEY"):
                                logger.warning("[LLM] ModelsLab out of credits; falling back to Gemini.")
                                return await achat_completion(
                                    provider="gemini",
                                    model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash"),
                                    messages=messages,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    response_format=response_format,
                                    timeout=timeout,
                                )
                            raise RuntimeError(f"ModelsLab error: {msg}")
                    if req.get("adapter") == "gemini":
                        # Normalize Gemini response to OpenAI-style dict
                        text = ""
                        try:
                            text = data["candidates"][0]["content"]["parts"][0]["text"]
                        except Exception:
                            text = ""
                        breaker.record_success()
                        return {
                            "choices": [{"message": {"content": text}}],
                            "raw": data,
                        }
                    breaker.record_success()
                    return data
    except Exception as e:
        logger.error(f"[LLM] Async request failed provider={provider} model={model}: {e}")
        breaker.record_failure()
        raise


async def achat_completion_stream(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
):
    """
    Provider-native streaming completion generator.
    Yields token strings as they arrive and returns full text on completion.
    """
    key = _provider_key(provider)
    if not key:
        raise RuntimeError(f"API key missing for provider: {provider}")

    breaker = get_breaker(
        provider,
        int(os.getenv("LLM_CB_FAILS", "3")),
        int(os.getenv("LLM_CB_RECOVERY_SEC", "30")),
        agent=get_agent_context(),
    )
    if not breaker.allow():
        raise RuntimeError(f"Circuit breaker open for provider: {provider}")

    req = _endpoint_and_payload(provider, model, messages, temperature, max_tokens, response_format)
    payload = dict(req["payload"])
    payload["stream"] = True

    # Gemini streaming uses streamGenerateContent with SSE.
    if req.get("adapter") == "gemini":
        req = dict(req)
        req["url"] = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={_provider_key('gemini')}"
        # Gemini streaming does not use OpenAI-style `stream` param
        payload.pop("stream", None)

    async def _iter_sse_lines(resp):
        buffer = ""
        async for chunk in resp.content.iter_chunked(1024):
            buffer += chunk.decode(errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                yield line

    full_text = ""
    async with aiohttp.ClientSession() as session:
        async with _get_async_semaphore(provider):
            async with session.post(req["url"], headers=req["headers"], json=payload, timeout=timeout) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").lower()
                # If server doesn't stream, fall back to JSON response.
                if "text/event-stream" not in content_type and "stream" not in content_type:
                    data = await resp.json()
                    if req.get("adapter") == "gemini":
                        text = ""
                        try:
                            text = data["candidates"][0]["content"]["parts"][0]["text"]
                        except Exception:
                            text = ""
                        yield text
                        breaker.record_success()
                        return
                    if req.get("adapter") == "modelslab" and isinstance(data, dict) and data.get("status") == "error":
                        msg = data.get("message", "")
                        raise RuntimeError(f"ModelsLab error: {msg}")
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    yield text
                    breaker.record_success()
                    return

                # Stream parsing
                async for line in _iter_sse_lines(resp):
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    # Gemini streamGenerateContent returns JSON objects per line
                    if req.get("adapter") == "gemini":
                        try:
                            obj = json.loads(line)
                            delta = obj["candidates"][0]["content"]["parts"][0]["text"]
                        except Exception:
                            delta = ""
                    else:
                        # OpenAI-compatible streaming
                        try:
                            obj = json.loads(line)
                            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        except Exception:
                            delta = ""
                    if delta:
                        full_text += delta
                        yield delta

    if not full_text:
        yield ""
    breaker.record_success()


def chat_completion(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    key = _provider_key(provider)
    if not key:
        raise RuntimeError(f"API key missing for provider: {provider}")

    breaker = get_breaker(
        provider,
        int(os.getenv("LLM_CB_FAILS", "3")),
        int(os.getenv("LLM_CB_RECOVERY_SEC", "30")),
        agent=get_agent_context(),
    )
    if not breaker.allow():
        raise RuntimeError(f"Circuit breaker open for provider: {provider}")

    req = _endpoint_and_payload(provider, model, messages, temperature, max_tokens, response_format)
    groq_retries = int(os.getenv("GROQ_RETRY_MAX", "3")) if provider.lower() == "groq" else 1
    backoff = float(os.getenv("GROQ_RETRY_BACKOFF_SEC", "1.5"))
    try:
        data = None
        for attempt in range(groq_retries):
            with _get_sync_semaphore(provider):
                resp = requests.post(req["url"], headers=req["headers"], json=req["payload"], timeout=timeout)
            if provider.lower() == "groq" and resp.status_code in (429, 500, 502, 503, 504):
                if attempt < groq_retries - 1:
                    time.sleep(backoff * (2 ** attempt))
                    continue
            resp.raise_for_status()
            data = resp.json()
            break
        if req.get("adapter") == "modelslab" and isinstance(data, dict) and data.get("status") == "error":
            msg = data.get("message", "")
            if "temperature" in msg and "default (1)" in msg:
                retry_payload = dict(req["payload"])
                retry_payload["temperature"] = 1
                resp = requests.post(req["url"], headers=req["headers"], json=retry_payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
            else:
                if "out of credits" in msg.lower() and os.getenv("GEMINI_API_KEY"):
                    logger.warning("[LLM] ModelsLab out of credits; falling back to Gemini.")
                    return chat_completion(
                        provider="gemini",
                        model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash"),
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        timeout=timeout,
                    )
                raise RuntimeError(f"ModelsLab error: {msg}")
        if req.get("adapter") == "gemini":
            text = ""
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                text = ""
            breaker.record_success()
            return {
                "choices": [{"message": {"content": text}}],
                "raw": data,
            }
        breaker.record_success()
        return data
    except requests.exceptions.Timeout as e:
        # ModelsLab retry with longer timeout
        if provider.lower() == "modelslab":
            retry_timeout = int(os.getenv("MODELSLAB_TIMEOUT_RETRY_SEC", "45"))
            try:
                resp = requests.post(req["url"], headers=req["headers"], json=req["payload"], timeout=retry_timeout)
                resp.raise_for_status()
                data = resp.json()
                breaker.record_success()
                return data
            except Exception:
                pass
        logger.error(f"[LLM] Sync request failed provider={provider} model={model}: {e}")
        breaker.record_failure()
        raise
    except Exception as e:
        logger.error(f"[LLM] Sync request failed provider={provider} model={model}: {e}")
        breaker.record_failure()
        raise


def run_chat_completion(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Convenience wrapper for sync chat completions.
    """
    return chat_completion(
        provider=provider,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        timeout=timeout,
    )
