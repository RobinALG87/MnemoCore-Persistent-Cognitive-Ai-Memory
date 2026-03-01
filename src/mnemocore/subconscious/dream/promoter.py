"""
Semantic Promoter – Stage 5 of Dream Pipeline
==============================================
Promotes important memories from HOT to WARM tier.

Promotion criteria:
- High LTP strength (consolidated memory)
- High access count (frequently retrieved)
- Part of important episodic clusters
- Tagged as important by user or system

Ensures important memories are properly consolidated and archived.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from ...core.engine import HAIMEngine
    from ...core.node import MemoryNode


@dataclass
class PromotionResult:
    """Result from promoting memories."""
    candidates_count: int
    promoted_count: int
    promoted_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates_count": self.candidates_count,
            "promoted_count": self.promoted_count,
            "promoted_ids": self.promoted_ids,
        }


class SemanticPromoter:
    """
    Promotes important memories from HOT to WARM tier.

    Promotion criteria:
    - High LTP strength (consolidated memory)
    - High access count (frequently retrieved)
    - Part of important episodic clusters
    - Tagged as important by user or system

    Ensures important memories are properly consolidated and archived.
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        ltp_threshold: float = 0.7,
        access_threshold: int = 5,
        auto_promote: bool = True,
    ):
        self.engine = engine
        self.ltp_threshold = ltp_threshold
        self.access_threshold = access_threshold
        self.auto_promote = auto_promote

    async def promote(
        self,
        memories: List["MemoryNode"],
        clusters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Identify and promote memories to WARM tier.

        Args:
            memories: List of memory nodes to evaluate.
            clusters: Optional episodic clusters for context-aware promotion.

        Returns:
            Dict with promotion results.
        """
        promoted = []
        candidates = []

        # Find candidates
        for memory in memories:
            if memory.tier != "hot":
                continue

            # Check promotion criteria
            if self._should_promote(memory):
                candidates.append(memory)

        # Promote if auto-promote is enabled
        if self.auto_promote and candidates:
            for memory in candidates:
                try:
                    await self._promote_to_warm(memory)
                    promoted.append(memory.id)
                except Exception as e:
                    logger.debug(f"[SemanticPromoter] Failed to promote {memory.id}: {e}")

        logger.info(
            f"[SemanticPromoter] Promoted {len(promoted)} memories "
            f"from {len(candidates)} candidates"
        )

        return {
            "candidates_count": len(candidates),
            "promoted_count": len(promoted),
            "promoted_ids": promoted,
        }

    def _should_promote(self, memory: "MemoryNode") -> bool:
        """Check if a memory should be promoted."""
        # High LTP strength
        if memory.ltp_strength >= self.ltp_threshold:
            return True

        # High access count
        if memory.access_count >= self.access_threshold:
            return True

        # Manually tagged as important
        if memory.metadata.get("important"):
            return True

        return False

    async def _promote_to_warm(self, memory: "MemoryNode") -> None:
        """Promote a memory to WARM tier."""
        # Update tier
        memory.tier = "warm"

        # Use tier manager's promotion logic
        await asyncio.to_thread(
            self.engine.tier_manager.promote_to_warm,
            memory.id,
        )


__all__ = ["SemanticPromoter", "PromotionResult"]
