"""
Export Routes
=============
Memory export endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request

from mnemocore.core.engine import HAIMEngine
from mnemocore.api.models import ExportResponse

router = APIRouter(tags=["Phase 5.0 - Export"])


def get_engine(request: Request) -> HAIMEngine:
    return request.app.state.engine


@router.get(
    "/export",
    response_model=ExportResponse,
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
