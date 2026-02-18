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
import logging
import secrets
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.engine import HAIMEngine
from src.core.config import get_config
from src.api.middleware import SecurityHeadersMiddleware, RateLimiter

# Configure logging
from loguru import logger

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# --- Observability ---
from prometheus_client import make_asgi_app
from src.core.metrics import (
    API_REQUEST_COUNT, 
    API_REQUEST_LATENCY, 
    track_async_latency,
    STORAGE_OPERATION_COUNT
)

metrics_app = make_asgi_app()


# --- Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Security Check
    config = get_config()
    if not config.security.api_key:
        logger.critical("No API Key configured! Set HAIM_API_KEY env var or security.api_key in config.")
        sys.exit(1)

    # Startup: Initialize Async Resources
    logger.info("Initializing Async Redis...")
    async_redis = AsyncRedisStorage.get_instance()
    if not await async_redis.check_health():
        logger.warning("Redis connection failed. Running in degraded mode (local only).")
    
    # Initialize implementation of engine
    logger.info("Initializing HAIMEngine...")
    engine = HAIMEngine(persist_path="./data/memory.jsonl")
    await engine.initialize()
    app.state.engine = engine
    
    yield
    
    # Shutdown: Clean up
    logger.info("Closing HAIMEngine...")
    await app.state.engine.close()
    
    logger.info("Closing Async Redis...")
    await async_redis.close()

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

# Security Headers
app.add_middleware(SecurityHeadersMiddleware)

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
    # Phase 3.5.1 Security - Prioritize config.security.api_key
    expected_key = config.security.api_key
    
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


# --- Request Models ---

class StoreRequest(BaseModel):
    content: str = Field(..., max_length=100_000)
    metadata: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    # Phase 3.5: TTL
    ttl: Optional[int] = None

    @field_validator('metadata')
    @classmethod
    def check_metadata_size(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return v
        if len(v) > 50:
            raise ValueError('Too many metadata keys')
        for key, value in v.items():
            if len(key) > 64:
                raise ValueError(f'Metadata key {key} too long')
            # Metadata values can be Any, but let's limit strings
            if isinstance(value, str) and len(value) > 1000:
                raise ValueError(f'Metadata value for {key} too long')
        return v

class QueryRequest(BaseModel):
    query: str = Field(..., max_length=10000)
    top_k: int = 5
    agent_id: Optional[str] = None

class ConceptRequest(BaseModel):
    name: str = Field(..., max_length=256)
    attributes: Dict[str, str]

    @field_validator('attributes')
    @classmethod
    def check_attributes_size(cls, v: Dict[str, str]) -> Dict[str, str]:
        if len(v) > 50:
            raise ValueError('Too many attributes')
        for key, value in v.items():
            if len(key) > 64:
                raise ValueError(f'Attribute key {key} too long')
            if len(value) > 1000:
                raise ValueError(f'Attribute value for {key} too long')
        return v

class AnalogyRequest(BaseModel):
    source_concept: str = Field(..., max_length=256)
    source_value: str = Field(..., max_length=1000)
    target_concept: str = Field(..., max_length=256)

# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "service": "MnemoCore", 
        "version": "3.5.1", 
        "phase": "Async I/O",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health")
async def health(engine: HAIMEngine = Depends(get_engine)):
    # Check Redis connectivity
    redis_connected = await AsyncRedisStorage.get_instance().check_health()

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


@app.post("/store", dependencies=[Depends(get_api_key), Depends(RateLimiter())])
@track_async_latency(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/store"})
async def store_memory(req: StoreRequest, engine: HAIMEngine = Depends(get_engine)):
    API_REQUEST_COUNT.labels(method="POST", endpoint="/store", status="200").inc()

    """Store a new memory (Async + Dual Write)."""
    metadata = req.metadata or {}
    if req.agent_id:
        metadata["agent_id"] = req.agent_id
    
    # 1. Run Core Engine (now Async)
    mem_id = await engine.store(req.content, metadata=metadata)
    
    # 2. Async Write to Redis (Metadata & LTP Index)
    # Get the node details we just created
    node = await engine.get_memory(mem_id)
    if node:
        redis_data = {
            "id": node.id,
            "content": node.content,
            "metadata": node.metadata,
            "ltp_strength": node.ltp_strength,
            "created_at": node.created_at.isoformat()
        }
        try:
            storage = AsyncRedisStorage.get_instance()
            await storage.store_memory(mem_id, redis_data, ttl=req.ttl)
            
            # PubSub Event
            await storage.publish_event("memory.created", {"id": mem_id})
        except Exception as e:
            logger.exception(f"Failed async write for {mem_id}")
            # Non-blocking failure for Redis write
    
    return {
        "ok": True,
        "memory_id": mem_id,
        "message": f"Stored memory: {mem_id}"
    }


@app.post("/query", dependencies=[Depends(get_api_key), Depends(RateLimiter())])
@track_async_latency(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/query"})
async def query_memory(req: QueryRequest, engine: HAIMEngine = Depends(get_engine)):
    API_REQUEST_COUNT.labels(method="POST", endpoint="/query", status="200").inc()

    """Query memories by semantic similarity (Async Wrapper)."""
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
async def get_memory(memory_id: str, engine: HAIMEngine = Depends(get_engine)):
    """Get a specific memory by ID."""
    # Try Redis first (L2 cache)
    storage = AsyncRedisStorage.get_instance()
    cached = await storage.retrieve_memory(memory_id)
    
    if cached:
        return {
            "source": "redis",
            **cached
        }
    
    # Fallback to Engine (TierManager)
    node = await engine.get_memory(memory_id)
    if not node:
        raise HTTPException(status_code=404, detail="Memory not found")
    
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


@app.delete("/memory/{memory_id}", dependencies=[Depends(get_api_key)])
async def delete_memory(memory_id: str, engine: HAIMEngine = Depends(get_engine)):
    """Delete a memory via Engine."""
    # Check if exists first for 404
    node = await engine.get_memory(memory_id)
    if not node:
         raise HTTPException(status_code=404, detail="Memory not found")
    
    # Engine delete (handles HOT/WARM)
    await engine.delete_memory(memory_id)
        
    # Also Redis
    storage = AsyncRedisStorage.get_instance()
    await storage.delete_memory(memory_id)
    
    return {"ok": True, "deleted": memory_id}

# --- Conceptual Endpoints ---

@app.post("/concept", dependencies=[Depends(get_api_key), Depends(RateLimiter())])
async def define_concept(req: ConceptRequest, engine: HAIMEngine = Depends(get_engine)):
    await engine.define_concept(req.name, req.attributes)
    return {"ok": True, "concept": req.name}


@app.post("/analogy", dependencies=[Depends(get_api_key), Depends(RateLimiter())])
async def solve_analogy(req: AnalogyRequest, engine: HAIMEngine = Depends(get_engine)):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
