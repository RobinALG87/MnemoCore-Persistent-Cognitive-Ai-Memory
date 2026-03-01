"""
Dream Routes
============
Dream loop trigger endpoints.
"""

import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.config import get_config
from mnemocore.api.models import DreamRequest, DreamResponse
from mnemocore.api.middleware import RateLimiter

router = APIRouter(tags=["Phase 5.0 - Dream Loop"])


def get_engine(request: Request) -> HAIMEngine:
    return request.app.state.engine


# Rate limit for dream endpoint: 5 calls per minute per user
class DreamRateLimiter(RateLimiter):
    """Rate limiter for dream endpoint: 5/minute."""
    def __init__(self):
        super().__init__(requests=5, window=60)


@router.post(
    "/dream",
    response_model=DreamResponse,
    dependencies=[Depends(DreamRateLimiter())],
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

    Rate limit: 5/minute (expensive LLM operations).
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
