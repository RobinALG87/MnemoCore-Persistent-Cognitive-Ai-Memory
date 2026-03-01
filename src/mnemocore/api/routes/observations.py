"""
Observation Routes
==================
Working memory observation endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request

from mnemocore.api.models import ObserveRequest

router = APIRouter(prefix="/wm", tags=["Working Memory"])


@router.post("/observe")
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


@router.get("/context/{agent_id}")
async def get_working_context(agent_id: str, limit: int = 16, request: Request = None):
    """Read active Working Memory context."""
    client = request.app.state.cognitive_client
    items = client.get_working_context(agent_id, limit=limit)
    return {"ok": True, "items": [
        {"id": i.id, "content": i.content, "kind": i.kind, "importance": i.importance}
        for i in items
    ]}
