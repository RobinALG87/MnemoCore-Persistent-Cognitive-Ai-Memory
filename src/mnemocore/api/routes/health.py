"""
Health Routes
=============
Health check and system statistics endpoints.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Request, Response, status

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.container import Container
from mnemocore.core.reliability import storage_circuit_breaker, vector_circuit_breaker
from mnemocore.api.models import HealthResponse, ReadinessResponse, RootResponse
from mnemocore.api.middleware import RATE_LIMIT_CONFIGS
from mnemocore.api.version import get_version

router = APIRouter(tags=["Health & Stats"])


def get_engine(request: Request) -> HAIMEngine:
    return request.app.state.engine


def get_container(request: Request) -> Container:
    return request.app.state.container


async def _redis_is_connected(container: Container | None) -> bool:
    if container is None or not getattr(container, "redis_storage", None):
        return False
    try:
        return bool(await container.redis_storage.check_health())
    except Exception:
        return False


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
async def health(request: Request):
    """Liveness endpoint; downstream degradation never makes the API non-live."""
    container = getattr(request.app.state, "container", None)
    engine = getattr(request.app.state, "engine", None)
    redis_connected = await _redis_is_connected(container)

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


@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request, response: Response):
    """Report whether the initialized local runtime can accept traffic."""
    container = getattr(request.app.state, "container", None)
    engine = getattr(request.app.state, "engine", None)
    engine_ready = engine is not None
    local_runtime_ready = engine_ready and container is not None
    redis_connected = await _redis_is_connected(container)

    if not local_runtime_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if local_runtime_ready else "not_ready",
        "engine_ready": engine_ready,
        "local_runtime_ready": local_runtime_ready,
        "redis_connected": redis_connected,
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
