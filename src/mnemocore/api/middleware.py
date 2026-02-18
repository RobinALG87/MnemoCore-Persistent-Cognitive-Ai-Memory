"""
Security Middleware & Utilities
===============================
Provides Rate Limiting and Security Headers for the API.
Supports differentiated rate limits per endpoint category.
"""

import time
from typing import Optional, Callable
from fastapi import Request, HTTPException, status, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from mnemocore.core.config import get_config


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

        # Identify client (prefer X-Forwarded-For if behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

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
            async with redis.pipeline() as pipe:
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
