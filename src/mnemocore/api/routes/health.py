"""
Health Routes
=============
Health check and system statistics endpoints.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Request

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.container import Container
from mnemocore.core.reliability import storage_circuit_breaker, vector_circuit_breaker
from mnemocore.api.models import HealthResponse, RootResponse
from mnemocore.api.middleware import RATE_LIMIT_CONFIGS
from mnemocore.api.version import get_version

router = APIRouter(tags=["Health & Stats"])


def get_engine(request: Request) -> HAIMEngine:
    return request.app.state.engine


def get_container(request: Request) -> Container:
    return request.app.state.container


@router.get("/", response_model=RootResponse)
async def root():
    return {
        "status": "ok",
        "service": "MnemoCore",
        "version": get_version(),
        "phase": "Async I/O",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/health", response_model=HealthResponse)
async def health(container: Container = Depends(get_container), engine: HAIMEngine = Depends(get_engine)):
    # Check Redis connectivity
    redis_connected = False
    if container.redis_storage:
        redis_connected = await container.redis_storage.check_health()

    # Check Circuit Breaker States (native implementation uses string state)
    storage_cb_state = storage_circuit_breaker.state
    vector_cb_state = vector_circuit_breaker.state

    is_healthy = redis_connected and storage_cb_state == "closed" and vector_cb_state == "closed"

    return {
        "status": "healthy" if is_healthy else "degraded",
        "redis_connected": redis_connected,
        "storage_circuit_breaker": storage_cb_state,
        "qdrant_circuit_breaker": vector_cb_state,
        "engine_ready": engine is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/stats")
async def get_stats(engine: HAIMEngine = Depends(get_engine)):
    """Get aggregate engine stats."""
    return await engine.get_stats()


@router.get("/rate-limits")
async def get_rate_limits():
    """Get current rate limit configuration."""
    return {
        "limits": {
            category: {
                "requests": cfg["requests"],
                "window_seconds": cfg["window"],
                "requests_per_minute": cfg["requests"],
                "description": cfg["description"]
            }
            for category, cfg in RATE_LIMIT_CONFIGS.items()
        }
    }
