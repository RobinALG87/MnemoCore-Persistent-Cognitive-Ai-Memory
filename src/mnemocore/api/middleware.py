"""
Security Middleware & Utilities
===============================
Provides Rate Limiting and Security Headers for the API.
Supports differentiated rate limits per endpoint category.
"""

import os
import secrets
import time
from typing import Optional, Callable
from fastapi import Request, HTTPException, status, Response, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from mnemocore.core.config import get_config


# --- API Key Security Dependency ---
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key_dependency(api_key: str = Security(_api_key_header)):
    """
    FastAPI dependency for API key validation.

    Validates the X-API-Key header against the configured API key.
    Can be used as a route dependency for per-route auth checks.
    """
    config = get_config()
    security = config.security if config else None
    expected_key = (security.api_key if security else None) or os.getenv("HAIM_API_KEY", "")

    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Misconfiguration: API Key not set"
        )

    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key"
        )
    return api_key


# Rate limit configurations per endpoint category
RATE_LIMIT_CONFIGS = {
    "store": {
        "requests": 100,
        "window": 60,  # 100/minute
        "description": "Memory storage operations"
    },
    "query": {
        "requests": 500,
        "window": 60,  # 500/minute
        "description": "Query operations"
    },
    "concept": {
        "requests": 100,
        "window": 60,  # 100/minute
        "description": "Concept operations"
    },
    "analogy": {
        "requests": 100,
        "window": 60,  # 100/minute
        "description": "Analogy operations"
    },
    "default": {
        "requests": 100,
        "window": 60,  # 100/minute
        "description": "Default rate limit"
    }
}


def get_endpoint_category(path: str) -> str:
    """Determine the rate limit category based on the endpoint path."""
    if "/store" in path:
        return "store"
    elif "/query" in path:
        return "query"
    elif "/concept" in path:
        return "concept"
    elif "/analogy" in path:
        return "analogy"
    return "default"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to every response.
    HSTS is configurable via security.hsts_enabled in config.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS - configurable via security.hsts_enabled
        # Default: enabled in production (when not in debug/dev mode)
        config = get_config()
        hsts_enabled = getattr(config.security, 'hsts_enabled', True) if config and hasattr(config, 'security') else True

        if hsts_enabled:
            # Only add HSTS if the request is over HTTPS or if explicitly enabled
            # HSTS header tells browsers to only use HTTPS for future requests
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class RateLimiter:
    """
    Dependency for rate limiting based on IP address using Redis.
    Supports differentiated rate limits per endpoint category.

    Usage:
        # Default rate limit (from config)
        @app.post("/endpoint", dependencies=[Depends(RateLimiter())])

        # Custom rate limit
        @app.post("/endpoint", dependencies=[Depends(RateLimiter(requests=100, window=60))])

        # Category-based rate limit
        @app.post("/store", dependencies=[Depends(RateLimiter(category="store"))])
    """
    def __init__(
        self,
        requests: Optional[int] = None,
        window: Optional[int] = None,
        category: Optional[str] = None
    ):
        self.requests = requests
        self.window = window
        self.category = category

    async def __call__(self, request: Request) -> None:
        config = get_config()
        if not config.security.rate_limit_enabled:
            return

        # Get container from app state
        container = getattr(request.app.state, 'container', None)
        if not container or not container.redis_storage:
            logger.warning("Redis storage not available, skipping rate limit.")
            return

        # Determine limits based on category or explicit parameters
        if self.category and self.category in RATE_LIMIT_CONFIGS:
            category_config = RATE_LIMIT_CONFIGS[self.category]
            limit = self.requests or category_config["requests"]
            window = self.window or category_config["window"]
        elif self.category:
            # Auto-detect from path if category specified but not found
            category = get_endpoint_category(request.url.path)
            category_config = RATE_LIMIT_CONFIGS[category]
            limit = self.requests or category_config["requests"]
            window = self.window or category_config["window"]
        else:
            # Use explicit params or fall back to config defaults
            limit = self.requests or config.security.rate_limit_requests
            window = self.window or config.security.rate_limit_window

        # SECURITY: Identify client IP safely
        # Only trust X-Forwarded-For header if the request comes from a trusted proxy
        client_ip = self._get_client_ip(request, config)

        # Determine category for key
        effective_category = self.category or get_endpoint_category(request.url.path)

        # Redis Key Strategy: rate_limit:CATEGORY:IP:WINDOW_INDEX
        # This is a fixed window counter.
        current_time = int(time.time())
        window_index = current_time // window
        key = f"rate_limit:{effective_category}:{client_ip}:{window_index}"

        try:
            redis = container.redis_storage.redis_client

            # Pipeline: INCR, EXPIRE
            # Ensure we await the creation of the pipeline for redis-py async compatibility
            async with (await redis.pipeline()) as pipe:
                pipe.incr(key)
                pipe.expire(key, window + 10)  # Set expiry slightly longer than window
                results = await pipe.execute()

            count = results[0]

            if count > limit:
                # Calculate retry-after time
                window_end = (window_index + 1) * window
                retry_after = window_end - current_time

                logger.warning(f"Rate limit exceeded for {client_ip} on {effective_category}: {count}/{limit}")

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)}
                )

        except HTTPException:
            raise
        except Exception as e:
            # If Redis fails, we generally fail open to maintain availability
            # unless strict security is required.
            logger.error(f"Rate limiter error (failing open): {e}")
            pass

    def _get_client_ip(self, request: Request, config) -> str:
        """
        SECURITY: Safely determine client IP address.

        Only trusts X-Forwarded-For header if the immediate client is in the
        trusted_proxies list. This prevents IP spoofing attacks where an
        attacker sets X-Forwarded-For to bypass rate limits.

        Args:
            request: The FastAPI request object
            config: The HAIM configuration

        Returns:
            The client IP address to use for rate limiting
        """
        # Get the immediate client IP
        immediate_client = request.client.host if request.client else "unknown"

        # Get trusted proxies from config
        trusted_proxies = getattr(config.security, 'trusted_proxies', [])

        # If no trusted proxies configured, never trust X-Forwarded-For
        if not trusted_proxies:
            return immediate_client

        # Check if immediate client is a trusted proxy
        if immediate_client in trusted_proxies:
            # Only then do we trust X-Forwarded-For
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                # Take the first IP in the chain (original client)
                # X-Forwarded-For: client, proxy1, proxy2
                client_ip = forwarded.split(",")[0].strip()
                # Validate it looks like an IP address
                if self._is_valid_ip(client_ip):
                    return client_ip

        # Default to immediate client
        return immediate_client

    def _is_valid_ip(self, ip: str) -> bool:
        """Validate that a string looks like a valid IP address."""
        import re
        # IPv4 pattern
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        # IPv6 pattern (simplified)
        ipv6_pattern = r'^[0-9a-fA-F:]+$'

        if re.match(ipv4_pattern, ip):
            # Validate IPv4 octets
            try:
                parts = [int(p) for p in ip.split('.')]
                return all(0 <= p <= 255 for p in parts)
            except ValueError:
                return False
        elif re.match(ipv6_pattern, ip) and len(ip) <= 45:
            return True
        return False


class StoreRateLimiter(RateLimiter):
    """Rate limiter for store endpoints: 100/minute."""
    def __init__(self):
        super().__init__(category="store")


class QueryRateLimiter(RateLimiter):
    """Rate limiter for query endpoints: 500/minute."""
    def __init__(self):
        super().__init__(category="query")


class ConceptRateLimiter(RateLimiter):
    """Rate limiter for concept endpoints: 100/minute."""
    def __init__(self):
        super().__init__(category="concept")


class AnalogyRateLimiter(RateLimiter):
    """Rate limiter for analogy endpoints: 100/minute."""
    def __init__(self):
        super().__init__(category="analogy")


async def rate_limit_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Custom exception handler for rate limit errors.
    Ensures HTTP 429 with Retry-After header.
    """
    retry_after = exc.headers.get("Retry-After", "60") if exc.headers else "60"

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": exc.detail,
            "error_type": "rate_limit_exceeded",
            "retry_after": int(retry_after)
        },
        headers={"Retry-After": retry_after}
    )
