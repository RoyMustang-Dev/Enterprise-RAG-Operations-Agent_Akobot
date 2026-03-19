"""
Redis Client Utilities

Provides a singleton Redis client for rate limiting and distributed locks.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)

_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    global _client
    if _client is not None:
        return _client
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        _client = redis.Redis.from_url(redis_url, decode_responses=True)
        # Validate connectivity quickly
        _client.ping()
        logger.info("[REDIS] Connected to Redis for shared state.")
        return _client
    except Exception as e:
        logger.warning(f"[REDIS] Failed to connect: {e}")
        _client = None
        return None
