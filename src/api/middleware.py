"""
Security Middleware & Utilities
===============================
Provides Rate Limiting and Security Headers for the API.
"""

import time
import logging
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.async_storage import AsyncRedisStorage
from src.core.config import get_config

logger = logging.getLogger("haim.security")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to every response.
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # HSTS is recommended for HTTPS, but might break local dev if not careful.
        # Uncomment for production with SSL.
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class RateLimiter:
    """
    Dependency for rate limiting based on IP address using Redis.
    Can be used as a dependency on specific routes or globally.
    """
    def __init__(self, requests: Optional[int] = None, window: Optional[int] = None):
        self.requests = requests
        self.window = window

    async def __call__(self, request: Request):
        config = get_config()
        if not config.security.rate_limit_enabled:
            return

        # Determine limits (default to config if not specified in init)
        limit = self.requests or config.security.rate_limit_requests
        window = self.window or config.security.rate_limit_window

        # Identify client
        client_ip = request.client.host if request.client else "unknown"

        # Redis Key Strategy: rate_limit:IP:WINDOW_INDEX
        # This is a fixed window counter.
        current_time = int(time.time())
        window_index = current_time // window
        key = f"rate_limit:{client_ip}:{window_index}"

        try:
            storage = AsyncRedisStorage.get_instance()
            # We access the underlying redis client directly for atomic operations
            if not hasattr(storage, 'redis_client'):
                 # Fallback or initialization check
                 logger.warning("Redis client not initialized in storage, skipping rate limit.")
                 return

            redis = storage.redis_client

            # Pipeline: INCR, EXPIRE
            async with redis.pipeline() as pipe:
                pipe.incr(key)
                pipe.expire(key, window + 10) # Set expiry slightly longer than window
                results = await pipe.execute()

            count = results[0]

            if count > limit:
                logger.warning(f"Rate limit exceeded for {client_ip}: {count}/{limit}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )

        except HTTPException:
            raise
        except Exception as e:
            # If Redis fails, we generally fail open to maintain availability
            # unless strict security is required.
            logger.error(f"Rate limiter error (failing open): {e}")
            pass
