"""
LLM Client

Unified helper for chat-completions across providers.
Currently supports: modelslab, gemini, groq.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp
import requests

logger = logging.getLogger(__name__)

_MODELSLAB_TEMP_FIXED = {
    # ModelsLab constraint: gpt-5-mini only supports temperature=1
    "gpt-5-mini": 1,
}

_MODELSLAB_RESPONSE_FORMAT_BLOCKLIST = {
    # Observed ModelsLab 200+error when response_format is used with these models
    "qwen-qwen3.5-122b-a10b",
    "qwen-qwen3.5-35b-a3b",
}


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

    req = _endpoint_and_payload(provider, model, messages, temperature, max_tokens, response_format)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(req["url"], headers=req["headers"], json=req["payload"], timeout=timeout) as resp:
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
                        raise RuntimeError(f"ModelsLab error: {msg}")
                if req.get("adapter") == "gemini":
                    # Normalize Gemini response to OpenAI-style dict
                    text = ""
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                    except Exception:
                        text = ""
                    return {
                        "choices": [{"message": {"content": text}}],
                        "raw": data,
                    }
                return data
    except Exception as e:
        logger.error(f"[LLM] Async request failed provider={provider} model={model}: {e}")
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
                    return
                if req.get("adapter") == "modelslab" and isinstance(data, dict) and data.get("status") == "error":
                    msg = data.get("message", "")
                    raise RuntimeError(f"ModelsLab error: {msg}")
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                yield text
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

    req = _endpoint_and_payload(provider, model, messages, temperature, max_tokens, response_format)
    try:
        resp = requests.post(req["url"], headers=req["headers"], json=req["payload"], timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if req.get("adapter") == "modelslab" and isinstance(data, dict) and data.get("status") == "error":
            msg = data.get("message", "")
            if "temperature" in msg and "default (1)" in msg:
                retry_payload = dict(req["payload"])
                retry_payload["temperature"] = 1
                resp = requests.post(req["url"], headers=req["headers"], json=retry_payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
            else:
                raise RuntimeError(f"ModelsLab error: {msg}")
        if req.get("adapter") == "gemini":
            text = ""
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                text = ""
            return {
                "choices": [{"message": {"content": text}}],
                "raw": data,
            }
        return data
    except Exception as e:
        logger.error(f"[LLM] Sync request failed provider={provider} model={model}: {e}")
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
