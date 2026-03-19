"""
Token Bucket Rate Limiter Middleware

Enterprise applications must prevent abusive users from rapidly draining the API limits.
This module checks incoming requests against an in-memory or Redis-backed bucket.
"""
import time
import logging
from typing import Dict, Tuple, Optional
from fastapi import HTTPException

from app.infra.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class TokenBucketRateLimiter:
    """
    Standard implementation of the Token Bucket algorithm.
    Each unique user/IP is mapped to a bucket filled with N tokens.
    Generating a response costs 1 token. Tokens refill linearly over time.
    """
    
    def __init__(self, capacity: int = 10, refill_rate_per_second: float = 0.5):
        """
        Args:
            capacity (int): Maximum burst amount of requests per client.
            refill_rate_per_second (float): How many tokens generate per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate_per_second

        # State: { "client_id": (tokens_remaining, last_refill_timestamp) }
        # If Redis is configured, we use shared buckets across workers.
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._redis = get_redis_client()
        self._redis_script = None
        if self._redis:
            self._redis_script = self._redis.register_script(
                """
                local key = KEYS[1]
                local capacity = tonumber(ARGV[1])
                local refill_rate = tonumber(ARGV[2])
                local now = tonumber(ARGV[3])
                local cost = tonumber(ARGV[4])

                local data = redis.call("HMGET", key, "tokens", "ts")
                local tokens = tonumber(data[1])
                local ts = tonumber(data[2])

                if tokens == nil or ts == nil then
                    tokens = capacity
                    ts = now
                end

                local elapsed = math.max(0, now - ts)
                local refill = math.floor(elapsed * refill_rate)
                if refill > 0 then
                    tokens = math.min(capacity, tokens + refill)
                    ts = now
                end

                if tokens >= cost then
                    tokens = tokens - cost
                    redis.call("HMSET", key, "tokens", tokens, "ts", ts)
                    redis.call("EXPIRE", key, 3600)
                    return 1
                else
                    redis.call("HMSET", key, "tokens", tokens, "ts", ts)
                    redis.call("EXPIRE", key, 3600)
                    return 0
                end
                """
            )
        
    def _refill(self, client_id: str):
        """Calculates time elapsed and adds tokens mathematically."""
        if self._redis:
            return
        now = time.time()
        
        # New client initialization
        if client_id not in self._buckets:
            self._buckets[client_id] = (self.capacity, now)
            return
            
        tokens, last_refill = self._buckets[client_id]
        elapsed = now - last_refill
        
        # Generate new tokens, capping at the physical `capacity` limit
        new_tokens = int(elapsed * self.refill_rate)
        if new_tokens > 0:
            tokens = min(self.capacity, tokens + new_tokens)
            self._buckets[client_id] = (tokens, now)
            
    def consume(self, client_id: str, cost: int = 1) -> bool:
        """
        Attempts to subtract a specific amount of tokens from the bucket.
        
        Args:
            client_id (str): The identifier (e.g. Session ID or IP address).
            cost (int): How much weight to pull from the bucket.
            
        Raises:
            HTTPException (429): If the client is depleted.
            
        Returns:
            bool: True if authorized to proceed.
        """
        # 1. Redis-backed limiter if available
        if self._redis and self._redis_script:
            try:
                allowed = int(self._redis_script(
                    keys=[f"rl:{client_id}"],
                    args=[self.capacity, self.refill_rate, time.time(), cost]
                ))
                if allowed == 1:
                    return True
            except Exception as e:
                logger.warning(f"[RATE LIMIT] Redis limiter failed; falling back to local. Error={e}")

        # 2. Update the bucket math natively (local fallback)
        self._refill(client_id)

        # 3. Extract state
        tokens, last_refill = self._buckets[client_id]

        # 4. Validation Logic
        if tokens >= cost:
            self._buckets[client_id] = (tokens - cost, last_refill)
            return True
            
        logger.warning(f"[RATE LIMIT] Client {client_id} exhausted bucket. Returning 429.")
        raise HTTPException(
            status_code=429, 
            detail="You're moving too fast. Generating enterprise responses requires heavy compute. Try again shortly."
        )
