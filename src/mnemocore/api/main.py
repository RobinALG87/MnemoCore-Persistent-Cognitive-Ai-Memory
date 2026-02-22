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

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.config import get_config
from mnemocore.core.container import build_container, Container
from mnemocore.api.middleware import (
    SecurityHeadersMiddleware,
    RateLimiter,
    StoreRateLimiter,
    QueryRateLimiter,
    ConceptRateLimiter,
    AnalogyRateLimiter,
    rate_limit_exception_handler,
    RATE_LIMIT_CONFIGS
)
from mnemocore.api.models import (
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
from mnemocore.core.logging_config import configure_logging
from mnemocore.core.exceptions import (
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
from mnemocore.core.metrics import (
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
            from mnemocore.core.metrics import set_trace_id
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
    from mnemocore.core.tier_manager import TierManager
    tier_manager = TierManager(config=config, qdrant_store=container.qdrant_store)
    engine = HAIMEngine(
        persist_path=config.paths.memory_file,
        config=config,
        tier_manager=tier_manager,
        working_memory=container.working_memory,
        episodic_store=container.episodic_store,
        semantic_store=container.semantic_store,
    )
    await engine.initialize()
    app.state.engine = engine
    # Also expose the cognitive client to app state for agentic frameworks
    from mnemocore.agent_interface import CognitiveMemoryClient
    app.state.cognitive_client = CognitiveMemoryClient(
        engine=engine,
        wm=container.working_memory,
        episodic=container.episodic_store,
        semantic=container.semantic_store,
        procedural=container.procedural_store,
        meta=container.meta_memory,
    )

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

from mnemocore.core.reliability import (
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
    metadata_filter = {"agent_id": req.agent_id} if req.agent_id else None
    results = await engine.query(req.query, top_k=req.top_k, metadata_filter=metadata_filter)

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

# --- Phase 5: Cognitive Client Endpoints ---

class ObserveRequest(BaseModel):
    agent_id: str
    content: str
    kind: str = "observation"
    importance: float = 0.5
    tags: Optional[List[str]] = None

@app.post("/wm/observe", dependencies=[Depends(get_api_key)])
async def observe_context(req: ObserveRequest, request: Request):
    """Push an observation explicitly into Working Memory."""
    client = request.app.state.cognitive_client
    if not client:
        raise HTTPException(status_code=503, detail="Cognitive Client unavailable")
    item_id = client.observe(
        agent_id=req.agent_id,
        content=req.content,
        kind=req.kind,
        importance=req.importance,
        tags=req.tags
    )
    return {"ok": True, "item_id": item_id}

@app.get("/wm/context/{agent_id}", dependencies=[Depends(get_api_key)])
async def get_working_context(agent_id: str, limit: int = 16, request: Request = None):
    """Read active Working Memory context."""
    client = request.app.state.cognitive_client
    items = client.get_working_context(agent_id, limit=limit)
    return {"ok": True, "items": [
        {"id": i.id, "content": i.content, "kind": i.kind, "importance": i.importance} 
        for i in items
    ]}

class EpisodeStartRequest(BaseModel):
    agent_id: str
    goal: str
    context: Optional[str] = None

@app.post("/episodes/start", dependencies=[Depends(get_api_key)])
async def start_episode(req: EpisodeStartRequest, request: Request):
    """Start a new episode chain."""
    client = request.app.state.cognitive_client
    ep_id = client.start_episode(req.agent_id, goal=req.goal, context=req.context)
    return {"ok": True, "episode_id": ep_id}

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


# ─────────────────────────────────────────────────────────────────────────────
# Maintenance Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/maintenance/cleanup", dependencies=[Depends(get_api_key)])
async def cleanup_maintenance(threshold: float = 0.1, engine: HAIMEngine = Depends(get_engine)):
    """Remove decayed synapses and stale index nodes."""
    await engine.cleanup_decay(threshold=threshold)
    return {"ok": True, "message": f"Synapse cleanup triggered with threshold {threshold}"}


@app.post("/maintenance/consolidate", dependencies=[Depends(get_api_key)])
async def consolidate_maintenance(engine: HAIMEngine = Depends(get_engine)):
    """Trigger manual semantic consolidation pulse."""
    if not engine._semantic_worker:
        raise HTTPException(status_code=503, detail="Consolidation worker not initialized")
    
    stats = await engine._semantic_worker.run_once()
    return {"ok": True, "stats": stats}


@app.post("/maintenance/sweep", dependencies=[Depends(get_api_key)])
async def sweep_maintenance(engine: HAIMEngine = Depends(get_engine)):
    """Trigger manual immunology sweep."""
    if not engine._immunology:
        raise HTTPException(status_code=503, detail="Immunology loop not initialized")
    
    stats = await engine._immunology.sweep()
    return {"ok": True, "stats": stats}


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



# ─────────────────────────────────────────────────────────────────────────────
# Phase 4.5: Recursive Synthesis Engine Endpoint
# ─────────────────────────────────────────────────────────────────────────────

class RLMQueryRequest(BaseModel):
    """Request model for Phase 4.5 recursive memory query."""
    query: str = Field(..., min_length=1, max_length=4096, description="The query to synthesize (can be complex/multi-topic)")
    context_text: Optional[str] = Field(None, max_length=500000, description="Optional large external text (Ripple environment)")
    project_id: Optional[str] = Field(None, max_length=128, description="Optional project scope for isolation masking")
    max_depth: Optional[int] = Field(None, ge=0, le=5, description="Max recursion depth (0-5, default 3)")
    max_sub_queries: Optional[int] = Field(None, ge=1, le=10, description="Max sub-queries to decompose into (1-10, default 5)")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Final results to return (default 10)")


class RLMQueryResponse(BaseModel):
    """Response model for Phase 4.5 recursive memory query."""
    ok: bool
    query: str
    sub_queries: List[str]
    results: List[Dict[str, Any]]
    synthesis: str
    max_depth_hit: int
    elapsed_ms: float
    ripple_snippets: List[str]
    stats: Dict[str, Any]


@app.post(
    "/rlm/query",
    response_model=RLMQueryResponse,
    dependencies=[Depends(get_api_key), Depends(QueryRateLimiter())],
    tags=["Phase 4.5"],
    summary="Recursive Synthesis Query",
    description=(
        "Phase 4.5: Recursive Language Model (RLM) query. "
        "Decomposes complex queries into sub-questions, searches MnemoCore in parallel, "
        "recursively analyzes low-confidence clusters, and synthesizes a final answer. "
        "Implements the MIT CSAIL RLM paradigm to eliminate Context Rot."
    ),
)
@track_async_latency(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/rlm/query"})
async def rlm_query(
    req: RLMQueryRequest,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Phase 4.5 Recursive Synthesis Engine.

    Instead of a single flat search, this endpoint:
    1. Decomposes your query into focused sub-questions
    2. Searches MnemoCore in PARALLEL for each sub-question
    3. Recursively drills into low-confidence clusters
    4. Synthesizes all results into a coherent answer

    Rate limit: 500/minute (shared with /query).
    """
    API_REQUEST_COUNT.labels(method="POST", endpoint="/rlm/query", status="200").inc()

    from mnemocore.core.recursive_synthesizer import RecursiveSynthesizer, SynthesizerConfig
    from mnemocore.core.ripple_context import RippleContext

    # Build config from request overrides
    synth_config = SynthesizerConfig(
        max_depth=req.max_depth if req.max_depth is not None else 3,
        max_sub_queries=req.max_sub_queries if req.max_sub_queries is not None else 5,
        final_top_k=req.top_k if req.top_k is not None else 10,
    )

    # Build RippleContext if external text provided
    ripple_ctx = None
    if req.context_text and req.context_text.strip():
        ripple_ctx = RippleContext(text=req.context_text, source_label="api_context")

    # Run recursive synthesis (no LLM wired at API level — use heuristic mode)
    # To enable LLM synthesis, configure via RLMIntegrator in your application code
    synthesizer = RecursiveSynthesizer(engine=engine, config=synth_config)
    result = await synthesizer.synthesize(
        query=req.query,
        ripple_context=ripple_ctx,
        project_id=req.project_id,
    )

    return {
        "ok": True,
        "query": result.query,
        "sub_queries": result.sub_queries,
        "results": result.results,
        "synthesis": result.synthesis,
        "max_depth_hit": result.max_depth_hit,
        "elapsed_ms": result.total_elapsed_ms,
        "ripple_snippets": result.ripple_snippets,
        "stats": result.stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5.0 — Agent 1: Trust & Provenance Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/memory/{memory_id}/lineage",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Trust"],
    summary="Get full provenance lineage for a memory",
)
async def get_memory_lineage(
    memory_id: str,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Return the complete provenance lineage of a memory:
    origin (who created it, how, when) and all transformation events
    (consolidated, verified, contradicted, archived, …).
    """
    node = await engine.get_memory(memory_id)
    if not node:
        raise MemoryNotFoundError(memory_id)

    prov = getattr(node, "provenance", None)
    if prov is None:
        return {
            "ok": True,
            "memory_id": memory_id,
            "provenance": None,
            "message": "No provenance record attached to this memory.",
        }

    return {
        "ok": True,
        "memory_id": memory_id,
        "provenance": prov.to_dict(),
    }


@app.get(
    "/memory/{memory_id}/confidence",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Trust"],
    summary="Get confidence envelope for a memory",
)
async def get_memory_confidence(
    memory_id: str,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Return a structured confidence envelope for a memory, combining:
    - Bayesian reliability (BayesianLTP posterior mean)
    - access_count (evidence strength)
    - staleness (days since last verification)
    - source_type trust weight
    - contradiction flag

    Level: high | medium | low | contradicted | stale
    """
    from mnemocore.core.confidence import build_confidence_envelope

    node = await engine.get_memory(memory_id)
    if not node:
        raise MemoryNotFoundError(memory_id)

    prov = getattr(node, "provenance", None)
    envelope = build_confidence_envelope(node, prov)

    return {
        "ok": True,
        "memory_id": memory_id,
        "confidence": envelope,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5.0 — Agent 3 stub: Proactive Recall
# (Full implementation added by Agent 3 workstream)
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/proactive",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Autonomy"],
    summary="Retrieve contextually relevant memories without explicit query",
)
async def get_proactive_memories(
    agent_id: Optional[str] = None,
    limit: int = 10,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Proactive recall stub (Phase 5.0 / Agent 3).
    Returns the most recently active high-LTP memories as a stand-in
    until the full ProactiveRecallDaemon is implemented.
    """
    nodes = await engine.tier_manager.get_hot_snapshot() if hasattr(engine, "tier_manager") else []
    sorted_nodes = sorted(nodes, key=lambda n: n.ltp_strength, reverse=True)[:limit]

    from mnemocore.core.confidence import build_confidence_envelope
    results = []
    for n in sorted_nodes:
        prov = getattr(n, "provenance", None)
        results.append({
            "id": n.id,
            "content": n.content,
            "ltp_strength": round(n.ltp_strength, 4),
            "confidence": build_confidence_envelope(n, prov),
            "tier": getattr(n, "tier", "hot"),
        })

    return {"ok": True, "proactive_results": results, "count": len(results)}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5.0 — Agent 2: Memory Lifecycle Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/contradictions",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Lifecycle"],
    summary="List active contradiction groups requiring resolution",
)
async def list_contradictions(
    unresolved_only: bool = True,
):
    """
    Returns all detected contradiction groups from the ContradictionRegistry.
    By default only unresolved contradictions are returned.
    """
    from mnemocore.core.contradiction import get_contradiction_detector
    detector = get_contradiction_detector()
    records = detector.registry.list_all(unresolved_only=unresolved_only)
    return {
        "ok": True,
        "count": len(records),
        "contradictions": [r.to_dict() for r in records],
    }


class ResolveContradictionRequest(BaseModel):
    note: Optional[str] = None


@app.post(
    "/contradictions/{group_id}/resolve",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Lifecycle"],
    summary="Mark a contradiction group as resolved",
)
async def resolve_contradiction(group_id: str, req: ResolveContradictionRequest):
    """Manually resolve a detected contradiction."""
    from mnemocore.core.contradiction import get_contradiction_detector
    detector = get_contradiction_detector()
    success = detector.registry.resolve(group_id, note=req.note)
    if not success:
        raise HTTPException(status_code=404, detail=f"Contradiction group {group_id!r} not found.")
    return {"ok": True, "resolved_group_id": group_id}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5.0 — Agent 3: Autonomous Cognition Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/memory/{memory_id}/emotional-tag",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Autonomy"],
    summary="Get emotional (valence/arousal) tag for a memory",
)
async def get_emotional_tag_ep(
    memory_id: str,
    engine: HAIMEngine = Depends(get_engine),
):
    """Return the valence/arousal emotional metadata for a memory."""
    from mnemocore.core.emotional_tag import get_emotional_tag
    node = await engine.get_memory(memory_id)
    if not node:
        raise MemoryNotFoundError(memory_id)
    tag = get_emotional_tag(node)
    return {
        "ok": True,
        "memory_id": memory_id,
        "emotional_tag": {
            "valence": tag.valence,
            "arousal": tag.arousal,
            "salience": round(tag.salience(), 4),
        },
    }


class EmotionalTagPatchRequest(BaseModel):
    valence: float
    arousal: float


@app.patch(
    "/memory/{memory_id}/emotional-tag",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Autonomy"],
    summary="Attach or update emotional tag on a memory",
)
async def patch_emotional_tag(
    memory_id: str,
    req: EmotionalTagPatchRequest,
    engine: HAIMEngine = Depends(get_engine),
):
    from mnemocore.core.emotional_tag import EmotionalTag, attach_emotional_tag
    node = await engine.get_memory(memory_id)
    if not node:
        raise MemoryNotFoundError(memory_id)
    tag = EmotionalTag(valence=req.valence, arousal=req.arousal)
    attach_emotional_tag(node, tag)
    return {"ok": True, "memory_id": memory_id, "emotional_tag": tag.to_metadata_dict()}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5.0 — Agent 4: Prediction Endpoints
# ─────────────────────────────────────────────────────────────────────────────

_prediction_store_instance = None


def _get_prediction_store(engine: HAIMEngine = Depends(get_engine)):
    from mnemocore.core.prediction_store import PredictionStore
    global _prediction_store_instance
    if _prediction_store_instance is None:
        _prediction_store_instance = PredictionStore(engine=engine)
    return _prediction_store_instance


class CreatePredictionRequest(BaseModel):
    content: str
    confidence: float = 0.5
    deadline_days: Optional[float] = None
    related_memory_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class VerifyPredictionRequest(BaseModel):
    success: bool
    notes: Optional[str] = None


@app.post(
    "/predictions",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Prediction"],
    summary="Store a new forward-looking prediction",
)
async def create_prediction(req: CreatePredictionRequest):
    from mnemocore.core.prediction_store import PredictionStore
    global _prediction_store_instance
    if _prediction_store_instance is None:
        _prediction_store_instance = PredictionStore()
    pred_id = _prediction_store_instance.create(
        content=req.content,
        confidence=req.confidence,
        deadline_days=req.deadline_days,
        related_memory_ids=req.related_memory_ids,
        tags=req.tags,
    )
    pred = _prediction_store_instance.get(pred_id)
    return {"ok": True, "prediction": pred.to_dict()}


@app.get(
    "/predictions",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Prediction"],
    summary="List all predictions",
)
async def list_predictions(status: Optional[str] = None):
    from mnemocore.core.prediction_store import PredictionStore
    global _prediction_store_instance
    if _prediction_store_instance is None:
        _prediction_store_instance = PredictionStore()
    return {
        "ok": True,
        "predictions": [
            p.to_dict()
            for p in _prediction_store_instance.list_all(status=status)
        ],
    }


@app.post(
    "/predictions/{pred_id}/verify",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Prediction"],
    summary="Verify or falsify a prediction",
)
async def verify_prediction(pred_id: str, req: VerifyPredictionRequest):
    from mnemocore.core.prediction_store import PredictionStore
    global _prediction_store_instance
    if _prediction_store_instance is None:
        _prediction_store_instance = PredictionStore()
    pred = await _prediction_store_instance.verify(pred_id, success=req.success, notes=req.notes)
    if pred is None:
        raise HTTPException(status_code=404, detail=f"Prediction {pred_id!r} not found.")
    return {"ok": True, "prediction": pred.to_dict()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
