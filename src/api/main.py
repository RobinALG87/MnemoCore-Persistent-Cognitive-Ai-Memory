"""
MnemoCore REST API
==================
FastAPI server for MnemoCore (Phase 3.5.1+).
Fully Async I/O with Redis backing.
"""

from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import sys
import os
import asyncio
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.engine import HAIMEngine
from src.core.async_storage import AsyncRedisStorage
from src.core.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("haim.api")

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
    # Startup: Initialize Async Resources
    logger.info("Initializing Async Redis...")
    async_redis = AsyncRedisStorage.get_instance()
    if not await async_redis.check_health():
        logger.warning("Redis connection failed. Running in degraded mode (local only).")
    
    # Initialize implementation of engine
    # Ideally engine should be initialized here too
    yield
    
    # Shutdown: Clean up
    logger.info("Closing Async Redis...")
    await async_redis.close()

app = FastAPI(
    title="MnemoCore API",
    description="MnemoCore - Infrastructure for Persistent Cognitive Memory - REST API (Async)",
    version="3.5.1",
    lifespan=lifespan
)

# Mount Prometheus metrics
app.mount("/metrics", metrics_app)


# Initialize Sync Engine (Global Singleton-ish for now)
# We keep persistent path as fallback
engine = HAIMEngine(persist_path="./data/memory.jsonl")


# --- Helpers ---

async def run_in_thread(func, *args, **kwargs):
    """Run blocking function in thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# --- Request Models ---

class StoreRequest(BaseModel):
    content: str = Field(..., max_length=100000)
    metadata: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    # Phase 3.5: TTL
    ttl: Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    agent_id: Optional[str] = None

class ConceptRequest(BaseModel):
    name: str
    attributes: Dict[str, str]

class AnalogyRequest(BaseModel):
    source_concept: str
    source_value: str
    target_concept: str

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "ok", "service": "MnemoCore", "version": "3.5.1", "phase": "Async I/O"}


@app.get("/health")
async def health():
    stats = engine.get_stats()
    redis_ok = await AsyncRedisStorage.get_instance().check_health()
    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis_connected": redis_ok,
        "stats": stats
    }


@app.post("/store")
@track_async_latency(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/store"})
async def store_memory(req: StoreRequest):
    API_REQUEST_COUNT.labels(method="POST", endpoint="/store", status="200").inc()

    """Store a new memory (Async + Dual Write)."""
    metadata = req.metadata or {}
    if req.agent_id:
        metadata["agent_id"] = req.agent_id
    
    # 1. Run Core Engine (CPU heavy / File I/O) in thread pool
    mem_id = await run_in_thread(engine.store, req.content, metadata=metadata)
    
    # 2. Async Write to Redis (Metadata & LTP Index)
    # Get the node details we just created
    node = engine.get_memory(mem_id)
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
            logger.error(f"Failed async write for {mem_id}: {e}")
            # Non-blocking failure for Redis write
    
    return {
        "ok": True,
        "memory_id": mem_id,
        "message": f"Stored memory: {mem_id}"
    }


@app.post("/query")
@track_async_latency(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/query"})
async def query_memory(req: QueryRequest):
    API_REQUEST_COUNT.labels(method="POST", endpoint="/query", status="200").inc()

    """Query memories by semantic similarity (Async Wrapper)."""
    # CPU heavy vector search
    results = await run_in_thread(engine.query, req.query, top_k=req.top_k)
    
    formatted = []
    for mem_id, score in results:
        # Check Redis first? For now rely on Engine's TierManager (which uses RAM/File)
        # because Engine has the object cache + Hashing logic.
        node = engine.get_memory(mem_id)
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


@app.get("/memory/{memory_id}")
async def get_memory(memory_id: str):
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
    node = engine.get_memory(memory_id)
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


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory."""
    # Engine delete (sync) - wait, engine doesn't expose delete explicitly except internal dictionary manip
    if memory_id not in engine.tier_manager.hot and not engine.tier_manager.get_memory(memory_id):
          raise HTTPException(status_code=404, detail="Memory not found")
    
    # Primitives to delete from tier manager?
    # Engine delete was: del engine.memory_nodes[memory_id]
    # We should expose delete in engine properly.
    # For now, replicate logic:
    if memory_id in engine.tier_manager.hot:
        del engine.tier_manager.hot[memory_id]
        
    # Also Redis
    storage = AsyncRedisStorage.get_instance()
    await storage.delete_memory(memory_id)
    
    return {"ok": True, "deleted": memory_id}

# --- Conceptual Endpoints (Async Wrappers) ---

@app.post("/concept")
async def define_concept(req: ConceptRequest):
    await run_in_thread(engine.define_concept, req.name, req.attributes)
    return {"ok": True, "concept": req.name}


@app.post("/analogy")
async def solve_analogy(req: AnalogyRequest):
    results = await run_in_thread(
        engine.reason_by_analogy,
        req.source_concept,
        req.source_value,
        req.target_concept
    )
    return {
        "ok": True,
        "analogy": f"{req.source_concept}:{req.source_value} :: {req.target_concept}:?",
        "results": [{"value": v, "score": float(s)} for v, s in results[:10]]
    }


@app.get("/stats")
async def get_stats():
    return await run_in_thread(engine.get_stats)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
