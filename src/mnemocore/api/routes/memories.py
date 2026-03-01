"""
Memory Routes
=============
Core memory CRUD operations: store, query, get, delete.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.container import Container
from mnemocore.core.exceptions import ValidationError, MemoryNotFoundError
from mnemocore.core.metrics import API_REQUEST_LATENCY, timer, API_REQUEST_COUNT
from mnemocore.api.middleware import StoreRateLimiter, QueryRateLimiter
from mnemocore.api.models import (
    StoreRequest,
    QueryRequest,
    StoreResponse,
    QueryResponse,
    DeleteResponse,
)

router = APIRouter(prefix="", tags=["Memory Operations"])


def get_engine(request: Request) -> HAIMEngine:
    return request.app.state.engine


def get_container(request: Request) -> Container:
    return request.app.state.container


@router.post(
    "/store",
    response_model=StoreResponse,
    dependencies=[Depends(StoreRateLimiter())]
)
@timer(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/store"})
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


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(QueryRateLimiter())]
)
@timer(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/query"})
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


@router.get("/memory/{memory_id}")
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


@router.delete(
    "/memory/{memory_id}",
    response_model=DeleteResponse,
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
