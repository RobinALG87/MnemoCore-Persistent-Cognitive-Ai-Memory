"""
Procedure Routes
================
Procedural memory and skill management endpoints.
"""

import dataclasses
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request

from mnemocore.core.engine import HAIMEngine
from mnemocore.api.models import ProcedureFeedbackRequest

router = APIRouter(prefix="/procedures", tags=["Procedural Memory"])


def get_engine(request: Request) -> HAIMEngine:
    return request.app.state.engine


@router.get("/search")
async def search_procedures(query: str, agent_id: Optional[str] = None, top_k: int = 5, request: Request = None):
    """Search for applicable procedural skills or workflows."""
    client = request.app.state.cognitive_client
    if not client:
        raise HTTPException(status_code=503, detail="Cognitive Client unavailable")

    procedures = client.suggest_procedures(agent_id=agent_id, query=query, top_k=top_k)

    return {"ok": True, "procedures": [dataclasses.asdict(p) for p in procedures]}


@router.post("/{proc_id}/feedback")
async def procedure_feedback(proc_id: str, req: ProcedureFeedbackRequest, request: Request = None):
    """Provide success/failure feedback on a procedure to update its reliability."""
    client = request.app.state.cognitive_client
    if not client:
        raise HTTPException(status_code=503, detail="Cognitive Client unavailable")

    client.record_procedure_outcome(proc_id, req.success)
    return {"ok": True, "procedure_id": proc_id, "success_recorded": req.success}
