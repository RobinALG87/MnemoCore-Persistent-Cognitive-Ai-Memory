"""
MnemoCore REST API
==================
FastAPI server for MnemoCore (Phase 3.5.1+).
Fully Async I/O with Redis backing.
"""

from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import sys
import os
import asyncio
import secrets
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator
from loguru import logger

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.engine import HAIMEngine
from src.core.config import get_config
from src.core.container import build_container, Container
from src.api.middleware import (
    SecurityHeadersMiddleware,
    RateLimiter,
    StoreRateLimiter,
    QueryRateLimiter,
    ConceptRateLimiter,
    AnalogyRateLimiter,
    rate_limit_exception_handler,
    RATE_LIMIT_CONFIGS
)
from src.api.models import (
    StoreRequest,
    QueryRequest,
    ConceptRequest,
    AnalogyRequest,
    StoreResponse,
    QueryResponse,
    QueryResult,
    DeleteResponse,
    ConceptResponse,
    AnalogyResponse,
    AnalogyResult,
    HealthResponse,
    RootResponse,
    ErrorResponse
)
from src.core.logging_config import configure_logging
from src.core.exceptions import (
    MnemoCoreError,
    RecoverableError,
    IrrecoverableError,
    ValidationError,
    NotFoundError,
    MemoryNotFoundError,
    is_debug_mode,
)

# Configure logging
configure_logging()

# --- Observability ---
from prometheus_client import make_asgi_app
from src.core.metrics import (
    API_REQUEST_COUNT,
    API_REQUEST_LATENCY,
    track_async_latency,
    STORAGE_OPERATION_COUNT,
    extract_trace_context,
    get_trace_id,
    init_opentelemetry,
    update_memory_count,
    update_queue_length,
    OTEL_AVAILABLE
)

# Initialize OpenTelemetry (optional, gracefully degrades if not installed)
if OTEL_AVAILABLE:
    init_opentelemetry(service_name="mnemocore", exporter="console")
    logger.info("OpenTelemetry tracing initialized")

metrics_app = make_asgi_app()


# --- Trace Context Middleware ---
class TraceContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract and propagate trace context via X-Trace-ID header.
    Integrates with OpenTelemetry for distributed tracing.
    """

    async def dispatch(self, request: Request, call_next):
        # Extract trace context from incoming headers
        headers = dict(request.headers)
        trace_id = headers.get("x-trace-id")

        if trace_id:
            # Set trace ID in context for downstream operations
            from src.core.metrics import set_trace_id
            set_trace_id(trace_id)
        else:
            # Try to extract from W3C Trace Context format
            extracted_id = extract_trace_context(headers)
            if extracted_id:
                trace_id = extracted_id

        # Process request
        response = await call_next(request)

        # Add trace ID to response headers for debugging
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id

        return response


# --- Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Security Check
    config = get_config()
    security = config.security if config else None
    _api_key = (security.api_key if security else None) or os.getenv("HAIM_API_KEY", "")
    if not _api_key:
        logger.critical("No API Key configured! Set HAIM_API_KEY env var or security.api_key in config.")
        sys.exit(1)

    # Startup: Build dependency container
    logger.info("Building dependency container...")
    container = build_container(config)
    app.state.container = container

    # Check Redis health
    logger.info("Checking Redis connection...")
    if container.redis_storage:
        if not await container.redis_storage.check_health():
            logger.warning("Redis connection failed. Running in degraded mode (local only).")
    else:
        logger.warning("Redis storage not available.")

    # Initialize implementation of engine with injected dependencies
    logger.info("Initializing HAIMEngine...")
    from src.core.tier_manager import TierManager
    tier_manager = TierManager(config=config, qdrant_store=container.qdrant_store)
    engine = HAIMEngine(
        persist_path="./data/memory.jsonl",
        config=config,
        tier_manager=tier_manager,
    )
    await engine.initialize()
    app.state.engine = engine

    yield

    # Shutdown: Clean up
    logger.info("Closing HAIMEngine...")
    await app.state.engine.close()

    logger.info("Closing Redis...")
    if container.redis_storage:
        await container.redis_storage.close()

app = FastAPI(
    title="MnemoCore API",
    description="MnemoCore - Infrastructure for Persistent Cognitive Memory - REST API (Async)",
    version="3.5.2",
    lifespan=lifespan
)

from src.core.reliability import (
    CircuitBreakerError,
    storage_circuit_breaker,
    vector_circuit_breaker
)


@app.exception_handler(CircuitBreakerError)
async def circuit_breaker_exception_handler(request: Request, exc: CircuitBreakerError):
    logger.error(f"Service Unavailable (Circuit Open): {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "Service Unavailable: Storage backend is down or overloaded.", "error": str(exc)},
    )


@app.exception_handler(MnemoCoreError)
async def mnemocore_exception_handler(request: Request, exc: MnemoCoreError):
    """
    Centralized exception handler for all MnemoCore errors.
    Returns JSON with error details and stacktrace only in DEBUG mode.
    """
    # Log the error with appropriate level
    if exc.recoverable:
        logger.warning(f"Recoverable error: {exc}")
    else:
        logger.error(f"Irrecoverable error: {exc}")

    # Determine HTTP status code based on error type
    if isinstance(exc, NotFoundError):
        status_code = 404
    elif isinstance(exc, ValidationError):
        status_code = 400
    elif isinstance(exc, RecoverableError):
        status_code = 503  # Service Unavailable
    else:
        status_code = 500

    # Build response
    response_data = exc.to_dict(include_traceback=is_debug_mode())

    return JSONResponse(
        status_code=status_code,
        content=response_data,
    )


# Security Headers
app.add_middleware(SecurityHeadersMiddleware)

# Trace Context Middleware (for OpenTelemetry distributed tracing)
app.add_middleware(TraceContextMiddleware)

# CORS
config = get_config()
cors_origins = config.security.cors_origins if hasattr(config, "security") else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics
app.mount("/metrics", metrics_app)

# --- Security ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    config = get_config()
    # Phase 3.5.1 Security - Prioritize config.security.api_key, fallback to env var
    security = config.security if config else None
    expected_key = (security.api_key if security else None) or os.getenv("HAIM_API_KEY", "")

    if not expected_key:
        # Should be caught by startup check, but double check
        logger.error("API Key not configured during request processing.")
        raise HTTPException(
            status_code=500,
            detail="Server Misconfiguration: API Key not set"
        )

    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API Key"
        )
    return api_key


def get_engine(request: Request) -> HAIMEngine:
    return request.app.state.engine


def get_container(request: Request) -> Container:
    return request.app.state.container


# --- Endpoints ---

@app.get("/", response_model=RootResponse)
async def root():
    return {
        "status": "ok",
        "service": "MnemoCore",
        "version": "3.5.1",
        "phase": "Async I/O",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health", response_model=HealthResponse)
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


@app.post(
    "/store",
    response_model=StoreResponse,
    dependencies=[Depends(get_api_key), Depends(StoreRateLimiter())]
)
@track_async_latency(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/store"})
async def store_memory(
    req: StoreRequest,
    engine: HAIMEngine = Depends(get_engine),
    container: Container = Depends(get_container)
):
    """Store a new memory (Async + Dual Write). Rate limit: 100/minute."""
    API_REQUEST_COUNT.labels(method="POST", endpoint="/store", status="200").inc()

    metadata = req.metadata or {}
    if req.agent_id:
        metadata["agent_id"] = req.agent_id

    # 1. Run Core Engine (now Async)
    mem_id = await engine.store(req.content, metadata=metadata)

    # 2. Async Write to Redis (Metadata & LTP Index)
    # Get the node details we just created
    node = await engine.get_memory(mem_id)
    if node and container.redis_storage:
        redis_data = {
            "id": node.id,
            "content": node.content,
            "metadata": node.metadata,
            "ltp_strength": node.ltp_strength,
            "created_at": node.created_at.isoformat()
        }
        try:
            await container.redis_storage.store_memory(mem_id, redis_data, ttl=req.ttl)

            # PubSub Event
            await container.redis_storage.publish_event("memory.created", {"id": mem_id})
        except Exception as e:
            logger.exception(f"Failed async write for {mem_id}")
            # Non-blocking failure for Redis write

    return {
        "ok": True,
        "memory_id": mem_id,
        "message": f"Stored memory: {mem_id}"
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(get_api_key), Depends(QueryRateLimiter())]
)
@track_async_latency(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/query"})
async def query_memory(
    req: QueryRequest,
    engine: HAIMEngine = Depends(get_engine)
):
    """Query memories by semantic similarity (Async Wrapper). Rate limit: 500/minute."""
    API_REQUEST_COUNT.labels(method="POST", endpoint="/query", status="200").inc()

    # CPU heavy vector search (offloaded inside engine)
    results = await engine.query(req.query, top_k=req.top_k)

    formatted = []
    for mem_id, score in results:
        # Check Redis first? For now rely on Engine's TierManager (which uses RAM/File)
        # because Engine has the object cache + Hashing logic.
        node = await engine.get_memory(mem_id)
        if node:
            formatted.append({
                "id": mem_id,
                "content": node.content,
                "score": float(score),
                "metadata": node.metadata,
                "tier": getattr(node, "tier", "unknown")
            })

    return {
        "ok": True,
        "query": req.query,
        "results": formatted
    }


@app.get("/memory/{memory_id}", dependencies=[Depends(get_api_key)])
async def get_memory(
    memory_id: str,
    engine: HAIMEngine = Depends(get_engine),
    container: Container = Depends(get_container)
):
    """Get a specific memory by ID."""
    # Validate memory_id format
    if not memory_id or len(memory_id) > 256:
        raise ValidationError(
            field="memory_id",
            reason="Memory ID must be between 1 and 256 characters",
            value=memory_id
        )

    # Try Redis first (L2 cache)
    cached = None
    if container.redis_storage:
        cached = await container.redis_storage.retrieve_memory(memory_id)

    if cached:
        return {
            "source": "redis",
            **cached
        }

    # Fallback to Engine (TierManager)
    node = await engine.get_memory(memory_id)
    if not node:
        raise MemoryNotFoundError(memory_id)

    return {
        "source": "engine",
        "id": node.id,
        "content": node.content,
        "metadata": node.metadata,
        "created_at": node.created_at.isoformat(),
        "epistemic_value": getattr(node, "epistemic_value", 0.0),
        "ltp_strength": getattr(node, "ltp_strength", 0.0),
        "tier": getattr(node, "tier", "unknown")
    }


@app.delete(
    "/memory/{memory_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(get_api_key)]
)
async def delete_memory(
    memory_id: str,
    engine: HAIMEngine = Depends(get_engine),
    container: Container = Depends(get_container)
):
    """Delete a memory via Engine."""
    # Validate memory_id format
    if not memory_id or len(memory_id) > 256:
        raise ValidationError(
            field="memory_id",
            reason="Memory ID must be between 1 and 256 characters",
            value=memory_id
        )

    # Check if exists first for 404
    node = await engine.get_memory(memory_id)
    if not node:
        raise MemoryNotFoundError(memory_id)

    # Engine delete (handles HOT/WARM)
    await engine.delete_memory(memory_id)

    # Also Redis
    if container.redis_storage:
        await container.redis_storage.delete_memory(memory_id)

    return {"ok": True, "deleted": memory_id}

# --- Conceptual Endpoints ---

@app.post(
    "/concept",
    response_model=ConceptResponse,
    dependencies=[Depends(get_api_key), Depends(ConceptRateLimiter())]
)
async def define_concept(req: ConceptRequest, engine: HAIMEngine = Depends(get_engine)):
    """Define a concept with attributes. Rate limit: 100/minute."""
    await engine.define_concept(req.name, req.attributes)
    return {"ok": True, "concept": req.name}


@app.post(
    "/analogy",
    response_model=AnalogyResponse,
    dependencies=[Depends(get_api_key), Depends(AnalogyRateLimiter())]
)
async def solve_analogy(req: AnalogyRequest, engine: HAIMEngine = Depends(get_engine)):
    """Solve an analogy. Rate limit: 100/minute."""
    results = await engine.reason_by_analogy(
        req.source_concept,
        req.source_value,
        req.target_concept
    )
    return {
        "ok": True,
        "analogy": f"{req.source_concept}:{req.source_value} :: {req.target_concept}:?",
        "results": [{"value": v, "score": float(s)} for v, s in results[:10]]
    }


@app.get("/stats", dependencies=[Depends(get_api_key)])
async def get_stats(engine: HAIMEngine = Depends(get_engine)):
    """Get aggregate engine stats."""
    return await engine.get_stats()


# Rate limit info endpoint
@app.get("/rate-limits")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
