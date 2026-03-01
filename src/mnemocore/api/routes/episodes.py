"""
Episode Routes
==============
Episodic memory management endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request

from mnemocore.core.engine import HAIMEngine
from mnemocore.api.models import EpisodeStartRequest

router = APIRouter(prefix="/episodes", tags=["Episodic Memory"])


def get_engine(request: Request) -> HAIMEngine:
    return request.app.state.engine


@router.post("/start")
async def start_episode(req: EpisodeStartRequest, request: Request):
    """Start a new episode chain."""
    client = request.app.state.cognitive_client
    if not client:
        raise HTTPException(status_code=503, detail="Cognitive Client unavailable")
    ep_id = client.start_episode(req.agent_id, goal=req.goal, context=req.context)
    return {"ok": True, "episode_id": ep_id}
