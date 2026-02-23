"""
MnemoCore REST API
==================
FastAPI server for MnemoCore (Phase 3.5.1+).
Fully Async I/O with Redis backing.
"""

from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime, timezone
import sys
import os
import asyncio
import secrets

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
    ErrorResponse,
    # Phase 6.0: Association models
    AssociationsQueryRequest,
    AssociationsQueryResponse,
    AssociationsPathRequest,
    AssociationsPathResponse,
    GraphMetricsResponse,
    ReinforceAssociationRequest,
    ReinforceAssociationResponse,
    AssociatedMemoryModel,
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

@app.get("/agents/{agent_id}/subtle-thoughts", dependencies=[Depends(get_api_key)])
async def get_subtle_thoughts(agent_id: str, limit: int = 5, engine: HAIMEngine = Depends(get_engine)):
    """Retrieve subtle thoughts (associations) generated from active working memory and episodic history."""
    if not hasattr(engine, "generate_subtle_thoughts"):
        return {"ok": True, "associations": []}
        
    associations = await engine.generate_subtle_thoughts(agent_id, limit=limit)
    return {"ok": True, "associations": associations}

@app.get("/procedures/search", dependencies=[Depends(get_api_key)])
async def search_procedures(query: str, agent_id: Optional[str] = None, top_k: int = 5, request: Request = None):
    """Search for applicable procedural skills or workflows."""
    client = request.app.state.cognitive_client
    if not client:
        raise HTTPException(status_code=503, detail="Cognitive Client unavailable")
        
    procedures = client.suggest_procedures(agent_id=agent_id, query=query, top_k=top_k)
    import dataclasses
    
    return {"ok": True, "procedures": [dataclasses.asdict(p) for p in procedures]}

class ProcedureFeedbackRequest(BaseModel):
    success: bool

@app.post("/procedures/{proc_id}/feedback", dependencies=[Depends(get_api_key)])
async def procedure_feedback(proc_id: str, req: ProcedureFeedbackRequest, request: Request = None):
    """Provide success/failure feedback on a procedure to update its reliability."""
    client = request.app.state.cognitive_client
    if not client:
        raise HTTPException(status_code=503, detail="Cognitive Client unavailable")
        
    client.record_procedure_outcome(proc_id, req.success)
    return {"ok": True, "procedure_id": proc_id, "success_recorded": req.success}

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


@app.get("/meta/proposals", dependencies=[Depends(get_api_key)], tags=["Phase 5.0 — Autonomy"])
async def get_meta_proposals(status: Optional[str] = None, request: Request = None):
    """List all self-improvement proposals generated by the AGI loop."""
    client = request.app.state.cognitive_client
    if not client:
        raise HTTPException(status_code=503, detail="Cognitive Client unavailable")
        
    proposals = client.get_self_improvement_proposals()
    if status:
        proposals = [p for p in proposals if p.status == status]
        
    import dataclasses
    return {"ok": True, "count": len(proposals), "proposals": [dataclasses.asdict(p) for p in proposals]}

class ProposalStatusUpdate(BaseModel):
    status: Literal["accepted", "rejected", "implemented"]

@app.post("/meta/proposals/{proposal_id}/status", dependencies=[Depends(get_api_key)], tags=["Phase 5.0 — Autonomy"])
async def update_proposal_status(proposal_id: str, req: ProposalStatusUpdate, request: Request = None):
    """Mark a generated self-improvement proposal as accepted, rejected, or implemented."""
    client = request.app.state.cognitive_client
    if not client:
        raise HTTPException(status_code=503, detail="Cognitive Client unavailable")
        
    client.meta.update_proposal_status(proposal_id, req.status)
    return {"ok": True, "proposal_id": proposal_id, "status": req.status}


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


@app.get("/gaps", dependencies=[Depends(get_api_key)], tags=["Phase 5.0 — Autonomy"])
async def get_knowledge_gaps(engine: HAIMEngine = Depends(get_engine)):
    """Retrieve detected knowledge gaps from the GapDetector."""
    if not hasattr(engine, "gap_detector"):
        return {"ok": True, "gaps": [], "count": 0}

    gaps = engine.gap_detector.list_gaps()
    return {
        "ok": True,
        "gaps": [g.to_dict() if hasattr(g, "to_dict") else g for g in gaps],
        "count": len(gaps)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5.0 — Advanced Synthesis, Dream & Export Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class DreamRequest(BaseModel):
    """Request model for triggering a dream session."""
    max_cycles: int = Field(default=1, ge=1, le=10, description="Number of dream cycles to run")
    force_insight: bool = Field(default=False, description="Force generation of a meta-insight")


class DreamResponse(BaseModel):
    """Response model for dream session."""
    ok: bool
    cycles_completed: int
    insights_generated: int
    concepts_extracted: int
    parallels_found: int
    memories_processed: int
    message: str


@app.post(
    "/dream",
    response_model=DreamResponse,
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Dream Loop"],
    summary="Trigger a dream session",
)
async def trigger_dream(
    req: DreamRequest,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Manually trigger a dream session (SubconsciousDaemon cycle).

    The dream loop performs:
    - Concept extraction from recent memories
    - Parallel drawing (finding unexpected connections)
    - Memory re-evaluation and valuation
    - Meta-insight generation

    This endpoint runs the daemon synchronously for the requested number of cycles
    and returns the results immediately.
    """
    try:
        # Import here to avoid circular dependency
        from mnemocore.subconscious.daemon import SubconsciousDaemon
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Dream loop not available: {e}"
        )

    # Get all hot memories for processing
    memories = list(engine.tier_manager.hot.values())

    if not memories:
        return {
            "ok": True,
            "cycles_completed": 0,
            "insights_generated": 0,
            "concepts_extracted": 0,
            "parallels_found": 0,
            "memories_processed": 0,
            "message": "No memories to process",
        }

    # Create a temporary daemon instance for this session
    daemon = SubconsciousDaemon(config=get_config())
    daemon.engine = engine

    # Run the requested number of cycles
    total_insights = 0
    total_concepts = 0
    total_parallels = 0

    for cycle in range(req.max_cycles):
        daemon.cycle_count += 1

        # 1. Extract concepts (every 5 cycles or forced)
        if cycle % 5 == 0 or req.force_insight:
            concepts = await daemon.extract_concepts(memories)
            for concept in concepts:
                if "name" in concept:
                    attrs = {k: str(v) for k, v in concept.items() if k != "name"}
                    engine.define_concept(concept["name"], attrs)
                    total_concepts += 1

        # 2. Draw parallels (every 3 cycles)
        if cycle % 3 == 0:
            parallels = await daemon.draw_parallels(memories)
            for p in parallels:
                # Store parallel as new memory
                await asyncio.to_thread(
                    engine.store,
                    f"[PARALLEL] {p}",
                    metadata={"type": "insight", "source": "dream_loop"}
                )
                total_insights += 1
                total_parallels += 1

        # 3. Generate meta-insight (every 7 cycles or forced)
        if cycle % 7 == 0 or req.force_insight:
            insight = await daemon.generate_insight(memories)
            if insight:
                await asyncio.to_thread(
                    engine.store,
                    f"[META-INSIGHT] {insight}",
                    metadata={"type": "meta", "source": "dream_loop"}
                )
                total_insights += 1

    return {
        "ok": True,
        "cycles_completed": req.max_cycles,
        "insights_generated": total_insights,
        "concepts_extracted": total_concepts,
        "parallels_found": total_parallels,
        "memories_processed": len(memories),
        "message": f"Completed {req.max_cycles} dream cycles",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6.0 — Association Network Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/associations/{node_id}",
    response_model=AssociationsQueryResponse,
    dependencies=[Depends(get_api_key)],
    tags=["Phase 6.0 — Associations"],
    summary="Get associations for a memory node",
)
async def get_associations(
    node_id: str,
    max_results: int = 10,
    min_strength: float = 0.1,
    include_content: bool = True,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Retrieve memories associated with the given node.

    Associations are formed through co-retrieval and represent
    Hebbian learning patterns in the memory network.
    """
    results = await engine.get_associated_memories(
        node_id=node_id,
        max_results=max_results,
        min_strength=min_strength,
        include_content=include_content,
    )

    return AssociationsQueryResponse(
        ok=True,
        node_id=node_id,
        associations=[
            AssociatedMemoryModel(
                id=r["id"],
                content=r.get("content", ""),
                strength=r["strength"],
                association_type=r["association_type"],
                fire_count=r["fire_count"],
                metadata=r.get("metadata"),
            )
            for r in results
        ],
    )


@app.post(
    "/associations/path",
    response_model=AssociationsPathResponse,
    dependencies=[Depends(get_api_key)],
    tags=["Phase 6.0 — Associations"],
    summary="Find association paths between memories",
)
async def find_association_path(
    req: AssociationsPathRequest,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Find paths connecting two memory nodes through the association network.

    Useful for discovering indirect relationships between memories.
    """
    if not hasattr(engine, "associations"):
        raise HTTPException(
            status_code=503,
            detail="Association network not available"
        )

    paths = engine.associations.find_associations_path(
        source_id=req.from_id,
        target_id=req.to_id,
        max_hops=req.max_hops,
        min_strength=req.min_strength,
    )

    return AssociationsPathResponse(
        ok=True,
        from_id=req.from_id,
        to_id=req.to_id,
        paths=paths,
    )


@app.get(
    "/associations/{node_id}/clusters",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 6.0 — Associations"],
    summary="Get clusters containing a memory node",
)
async def get_node_clusters(
    node_id: str,
    min_cluster_size: int = 3,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Find clusters (groups of strongly associated memories) that contain this node.
    """
    if not hasattr(engine, "associations"):
        raise HTTPException(
            status_code=503,
            detail="Association network not available"
        )

    clusters = engine.associations.find_clusters(
        min_cluster_size=min_cluster_size,
        min_strength=0.2,
    )

    # Filter clusters that contain the node
    node_clusters = [c for c in clusters if node_id in c]

    return {
        "ok": True,
        "node_id": node_id,
        "clusters": [
            {"size": len(c), "nodes": list(c)}
            for c in node_clusters
        ],
        "count": len(node_clusters),
    }


@app.get(
    "/associations/metrics",
    response_model=GraphMetricsResponse,
    dependencies=[Depends(get_api_key)],
    tags=["Phase 6.0 — Associations"],
    summary="Get association network metrics",
)
async def get_association_metrics(
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Get structural metrics about the association network.

    Includes node count, edge count, density, clustering coefficient,
    and other graph-theoretic measures.
    """
    if not hasattr(engine, "associations"):
        raise HTTPException(
            status_code=503,
            detail="Association network not available"
        )

    metrics = engine.associations.compute_metrics()

    from mnemocore.api.models import GraphMetricsModel
    return GraphMetricsResponse(
        ok=True,
        metrics=GraphMetricsModel(
            node_count=metrics.node_count,
            edge_count=metrics.edge_count,
            avg_degree=metrics.avg_degree,
            density=metrics.density,
            avg_clustering=metrics.avg_clustering,
            connected_components=metrics.connected_components,
            largest_component_size=metrics.largest_component_size,
        ),
    )


@app.post(
    "/associations/reinforce",
    response_model=ReinforceAssociationResponse,
    dependencies=[Depends(get_api_key)],
    tags=["Phase 6.0 — Associations"],
    summary="Manually reinforce an association",
)
async def reinforce_association(
    req: ReinforceAssociationRequest,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Manually reinforce the association between two memory nodes.

    This can be used to explicitly encode domain knowledge about
    relationships between memories.
    """
    if not hasattr(engine, "associations"):
        raise HTTPException(
            status_code=503,
            detail="Association network not available"
        )

    from mnemocore.cognitive.associations import AssociationType

    edge = engine.associations.reinforce(
        node_a=req.node_a,
        node_b=req.node_b,
        association_type=AssociationType(req.association_type),
    )

    if edge:
        from mnemocore.api.models import AssociationEdgeModel
        return ReinforceAssociationResponse(
            ok=True,
            edge=AssociationEdgeModel(
                source_id=edge.source_id,
                target_id=edge.target_id,
                strength=edge.strength,
                association_type=edge.association_type.value,
                created_at=edge.created_at.isoformat(),
                last_strengthened=edge.last_strengthened.isoformat(),
                fire_count=edge.fire_count,
            ),
            message=f"Reinforced association between {req.node_a[:8]} and {req.node_b[:8]}",
        )
    else:
        return ReinforceAssociationResponse(
            ok=False,
            edge=None,
            message="Failed to reinforce association (nodes may not exist)",
        )


@app.get(
    "/associations/visualize",
    dependencies=[Depends(get_api_key)],
    tags=["Phase 6.0 — Associations"],
    summary="Get association network visualization",
)
async def visualize_associations(
    max_nodes: int = 100,
    min_strength: float = 0.1,
    layout: str = "spring",
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Generate an HTML visualization of the association network.

    Returns HTML that can be rendered in a browser.
    """
    if not hasattr(engine, "associations"):
        raise HTTPException(
            status_code=503,
            detail="Association network not available"
        )

    html = engine.associations.visualize(
        max_nodes=max_nodes,
        min_strength=min_strength,
        layout=layout,
    )

    if html is None:
        raise HTTPException(
            status_code=503,
            detail="Visualization not available (Plotly not installed or no data)"
        )

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


class ExportResponse(BaseModel):
    """Response model for memory export."""
    ok: bool
    count: int
    format: str
    memories: List[Dict[str, Any]]


@app.get(
    "/export",
    response_model=ExportResponse,
    dependencies=[Depends(get_api_key)],
    tags=["Phase 5.0 — Export"],
    summary="Export memories as JSON",
)
async def export_memories(
    agent_id: Optional[str] = None,
    tier: Optional[str] = None,
    limit: int = 100,
    include_metadata: bool = True,
    format: str = "json",
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Export memories for backup, analysis, or migration.

    Returns memories in the requested format with optional filtering by
    agent_id or tier.
    """
    # Validate tier
    valid_tiers = {"hot", "warm", "cold", "soul"}
    if tier and tier not in valid_tiers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier. Must be one of: {', '.join(valid_tiers)}"
        )

    # Validate format
    if format not in ("json", "jsonl"):
        raise HTTPException(
            status_code=400,
            detail="Invalid format. Must be 'json' or 'jsonl'"
        )

    # Collect memories based on tier filter
    memories_to_export = []

    if tier == "hot" or tier is None:
        for node in engine.tier_manager.hot.values():
            if agent_id is None or node.metadata.get("agent_id") == agent_id:
                memories_to_export.append(node)

    if tier == "warm" or tier is None:
        # For warm tier, we need to fetch from Qdrant
        if hasattr(engine.tier_manager, "qdrant_store") and engine.tier_manager.qdrant_store:
            from mnemocore.core.qdrant_store import QdrantStore
            qdrant: QdrantStore = engine.tier_manager.qdrant_store
            # This would need a proper list method in QdrantStore
            # For now, we skip warm tier export or add the method
            pass

    # Apply limit
    memories_to_export = memories_to_export[:limit]

    # Format output
    exported = []
    for node in memories_to_export:
        mem_dict = {
            "id": node.id,
            "content": node.content,
            "created_at": node.created_at.isoformat(),
            "ltp_strength": node.ltp_strength,
            "tier": getattr(node, "tier", "hot"),
        }
        if include_metadata:
            mem_dict["metadata"] = node.metadata
        exported.append(mem_dict)

    return {
        "ok": True,
        "count": len(exported),
        "format": format,
        "memories": exported,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
