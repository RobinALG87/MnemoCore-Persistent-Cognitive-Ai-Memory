"""
Contradiction Resolver – Stage 4 of Dream Pipeline
===================================================
Detects and resolves contradictions using the existing ContradictionDetector.

During dream sessions, we:
1. Scan memories for contradictions
2. Attempt auto-resolution for simple cases
3. Flag complex cases for manual review
4. Track resolution status across sessions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from ...core.engine import HAIMEngine
    from ...core.node import MemoryNode
    from ...core.contradiction import ContradictionDetector


@dataclass
class ContradictionScanResult:
    """Result from scanning for contradictions."""
    contradictions_found: int
    contradictions_resolved: int
    resolved_ids: List[str]
    unresolved_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradictions_found": self.contradictions_found,
            "contradictions_resolved": self.contradictions_resolved,
            "resolved_ids": self.resolved_ids,
            "unresolved_ids": self.unresolved_ids,
        }


class ContradictionResolver:
    """
    Detects and resolves contradictions using the existing ContradictionDetector.

    During dream sessions, we:
    1. Scan memories for contradictions
    2. Attempt auto-resolution for simple cases
    3. Flag complex cases for manual review
    4. Track resolution status across sessions
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        similarity_threshold: float = 0.80,
        auto_resolve: bool = False,
    ):
        self.engine = engine
        self.similarity_threshold = similarity_threshold
        self.auto_resolve = auto_resolve

        # Lazy load detector
        self._detector: Optional["ContradictionDetector"] = None

    def _get_detector(self) -> "ContradictionDetector":
        """Get or create the contradiction detector."""
        if self._detector is None:
            from ...core.contradiction import get_contradiction_detector
            self._detector = get_contradiction_detector(self.engine)
        return self._detector

    async def scan_and_resolve(
        self,
        memories: List["MemoryNode"],
    ) -> Dict[str, Any]:
        """
        Scan memories for contradictions and attempt resolution.

        Args:
            memories: List of memory nodes to scan.

        Returns:
            Dict with scan results and resolution info.
        """
        if not memories:
            return {
                "contradictions_found": 0,
                "contradictions_resolved": 0,
                "resolved_ids": [],
                "unresolved_ids": [],
            }

        detector = self._get_detector()

        # Run background scan
        found = await detector.scan(memories)

        contradictions = []
        resolved = []

        for record in found:
            if not record.resolved:
                contradictions.append(record)

                # Attempt auto-resolution if enabled
                if self.auto_resolve and self._is_simple_contradiction(record):
                    resolution = await self._auto_resolve(record)
                    if resolution:
                        resolved.append(resolution)

        logger.info(
            f"[ContradictionResolver] Found {len(contradictions)} contradictions, "
            f"resolved {len(resolved)}"
        )

        return {
            "contradictions_found": len(contradictions),
            "contradictions_resolved": len(resolved),
            "resolved_ids": [r.group_id for r in resolved],
            "unresolved_ids": [r.group_id for r in contradictions if r not in resolved],
        }

    def _is_simple_contradiction(self, record: Any) -> bool:
        """Check if a contradiction is simple enough to auto-resolve."""
        # Simple contradictions have high similarity and clear resolution
        return (
            record.similarity_score > 0.95 and
            not record.llm_confirmed  # Only auto-resolve non-LLM confirmed
        )

    async def _auto_resolve(self, record: Any) -> Optional[Any]:
        """Attempt automatic resolution of a contradiction."""
        try:
            # Mark as resolved with a note
            self._detector.registry.resolve(
                record.group_id,
                note="Auto-resolved by DreamSession - high similarity"
            )
            return record
        except Exception as e:
            logger.debug(f"[ContradictionResolver] Auto-resolve failed: {e}")
            return None


__all__ = ["ContradictionResolver", "ContradictionScanResult"]
