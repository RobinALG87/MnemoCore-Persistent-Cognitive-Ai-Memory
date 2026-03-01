"""
Prediction Routes
=================
Prediction management endpoints.
"""

from typing import Optional
from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException, Request

from mnemocore.core.engine import HAIMEngine
from mnemocore.api.models import CreatePredictionRequest, VerifyPredictionRequest

router = APIRouter(prefix="/predictions", tags=["Phase 5.0 - Prediction"])


def get_engine(request: Request) -> HAIMEngine:
    return request.app.state.engine


@lru_cache(maxsize=1)
def _get_prediction_store():
    """
    Get the prediction store singleton using lru_cache for thread-safe initialization.
    This fixes the race condition in the original global variable approach.
    """
    from mnemocore.core.prediction_store import PredictionStore
    return PredictionStore()


@router.post(
    "",
    summary="Store a new forward-looking prediction",
)
async def create_prediction(req: CreatePredictionRequest):
    store = _get_prediction_store()
    pred_id = store.create(
        content=req.content,
        confidence=req.confidence,
        deadline_days=req.deadline_days,
        related_memory_ids=req.related_memory_ids,
        tags=req.tags,
    )
    pred = store.get(pred_id)
    return {"ok": True, "prediction": pred.to_dict()}


@router.get(
    "",
    summary="List all predictions",
)
async def list_predictions(status: Optional[str] = None):
    store = _get_prediction_store()
    return {
        "ok": True,
        "predictions": [
            p.to_dict()
            for p in store.list_all(status=status)
        ],
    }


@router.post(
    "/{pred_id}/verify",
    summary="Verify or falsify a prediction",
)
async def verify_prediction(pred_id: str, req: VerifyPredictionRequest):
    store = _get_prediction_store()
    pred = await store.verify(pred_id, success=req.success, notes=req.notes)
    if pred is None:
        raise HTTPException(status_code=404, detail=f"Prediction {pred_id!r} not found.")
    return {"ok": True, "prediction": pred.to_dict()}
