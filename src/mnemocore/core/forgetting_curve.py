"""
Forgetting Curve Manager (Phase 5.0)
=====================================
Implements Ebbinghaus-based spaced repetition scheduling for MnemoCore.

The ForgettingCurveManager layers on top of AdaptiveDecayEngine to:
  1. Schedule "review" events at optimal intervals (spaced repetition)
  2. Decide whether low-retention memories should be consolidated vs. deleted
  3. Work collaboratively with the ConsolidationWorker

Key idea: at each review interval, the system re-evaluates a memory's EIG.
  - High EIG + low retention → CONSOLIDATE (absorb into a stronger anchor)
  - Low EIG + low retention  → ARCHIVE / EVICT

The review scheduling uses the SuperMemo-inspired interval:
    next_review_days = S_i * ln(1 / TARGET_RETENTION)^-1

where TARGET_RETENTION = 0.70 (retain 70% at next review point).

Public API:
    manager = ForgettingCurveManager(engine)
    await manager.run_once(nodes)   # scan HOT/WARM nodes, schedule reviews
    schedule = manager.get_schedule()  # sorted list of upcoming reviews
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional

from loguru import logger

from .temporal_decay import AdaptiveDecayEngine, get_adaptive_decay_engine

if TYPE_CHECKING:
    from .node import MemoryNode


# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

TARGET_RETENTION: float = 0.70  # Retention level at which we schedule the next review
MIN_EIG_TO_CONSOLIDATE: float = 0.3  # Minimum epistemic value to consolidate instead of evict


# ------------------------------------------------------------------ #
#  Review Schedule Entry                                             #
# ------------------------------------------------------------------ #

@dataclass
class ReviewEntry:
    """A scheduled review for a single memory."""
    memory_id: str
    due_at: datetime          # When to review
    current_retention: float  # Retention at scheduling time
    stability: float          # S_i at scheduling time
    action: str = "review"    # "review" | "consolidate" | "evict"

    def to_dict(self) -> Dict:
        return {
            "memory_id": self.memory_id,
            "due_at": self.due_at.isoformat(),
            "current_retention": round(self.current_retention, 4),
            "stability": round(self.stability, 4),
            "action": self.action,
        }


# ------------------------------------------------------------------ #
#  Forgetting Curve Manager                                          #
# ------------------------------------------------------------------ #

class ForgettingCurveManager:
    """
    Schedules spaced-repetition review events for MemoryNodes.

    Attach to a running HAIMEngine to enable automatic review scheduling.
    Works in concert with AdaptiveDecayEngine and ConsolidationWorker.
    """

    def __init__(
        self,
        engine=None,  # HAIMEngine – typed as Any to avoid circular import
        decay_engine: Optional[AdaptiveDecayEngine] = None,
        target_retention: float = TARGET_RETENTION,
        min_eig_to_consolidate: float = MIN_EIG_TO_CONSOLIDATE,
    ) -> None:
        self.engine = engine
        self.decay = decay_engine or get_adaptive_decay_engine()
        self.target_retention = target_retention
        self.min_eig_to_consolidate = min_eig_to_consolidate
        self._schedule: List[ReviewEntry] = []

    # ---- Interval calculation ------------------------------------ #

    def next_review_days(self, node: "MemoryNode") -> float:
        """
        Days until the next review should be scheduled.

        Derived from: TARGET_RETENTION = e^(-next_days / S_i)
        → next_days = -S_i * ln(TARGET_RETENTION)

        Example: S_i=5, target=0.70 → next_days = 5 × 0.357 ≈ 1.78 days
        """
        s_i = self.decay.stability(node)
        # Protect against math domain errors
        target = max(1e-6, min(self.target_retention, 0.999))
        return -s_i * math.log(target)

    def _determine_action(self, node: "MemoryNode", retention: float) -> str:
        """
        Decide what to do with a low-retention memory:
        - consolidate: has historical importance (epistemic_value > threshold)
        - evict: low value, low retention
        - review: needs attention but not critical yet
        """
        if self.decay.should_evict(node):
            eig = getattr(node, "epistemic_value", 0.0)
            if eig >= self.min_eig_to_consolidate:
                return "consolidate"
            return "evict"
        return "review"

    # ---- Scan and schedule --------------------------------------- #

    def schedule_reviews(self, nodes: "List[MemoryNode]") -> List[ReviewEntry]:
        """
        Scan the provided nodes and build a schedule of upcoming reviews.
        Nodes with retention ≤ REVIEW_THRESHOLD are immediately flagged.

        Returns the new ReviewEntry objects added to the schedule.
        """
        now = datetime.now(timezone.utc)
        new_entries: List[ReviewEntry] = []

        for node in nodes:
            retention = self.decay.retention(node)
            s_i = self.decay.stability(node)

            # Always update review_candidate flag on the node itself
            self.decay.update_review_candidate(node)

            # Schedule next review based on spaced repetition interval
            days_until = self.next_review_days(node)
            due_at = now + timedelta(days=days_until)
            action = self._determine_action(node, retention)

            entry = ReviewEntry(
                memory_id=node.id,
                due_at=due_at,
                current_retention=retention,
                stability=s_i,
                action=action,
            )
            new_entries.append(entry)

        # Merge into the schedule (replace existing entries for same memory_id)
        existing_ids = {e.memory_id for e in self._schedule}
        self._schedule = [
            e for e in self._schedule if e.memory_id not in {n.id for n in nodes}
        ]
        self._schedule.extend(new_entries)
        self._schedule.sort(key=lambda e: e.due_at)

        logger.info(
            f"ForgettingCurveManager: scheduled {len(new_entries)} reviews for {len(nodes)} nodes. "
            f"Total scheduled: {len(self._schedule)}"
        )
        return new_entries

    def get_schedule(self) -> List[ReviewEntry]:
        """Return the current review schedule sorted by due_at."""
        return sorted(self._schedule, key=lambda e: e.due_at)

    def get_due_reviews(self) -> List[ReviewEntry]:
        """Return entries that are due now (due_at <= now)."""
        now = datetime.now(timezone.utc)
        return [e for e in self._schedule if e.due_at <= now]

    def get_actions_by_type(self, action: str) -> List[ReviewEntry]:
        """Filter schedule by action type: 'review', 'consolidate', or 'evict'."""
        return [e for e in self._schedule if e.action == action]

    def remove_entry(self, memory_id: str) -> None:
        """Remove a memory from the review schedule (e.g., it was evicted)."""
        self._schedule = [e for e in self._schedule if e.memory_id != memory_id]

    # ---- Engine integration ------------------------------------- #

    async def run_once(self) -> Dict:
        """
        Run a full scan over HOT + WARM nodes and update the review schedule.

        Returns a stats dict with counts per action.
        """
        if self.engine is None:
            logger.warning("ForgettingCurveManager: no engine attached, cannot scan tiers.")
            return {}

        nodes: List["MemoryNode"] = []
        try:
            hot = await self.engine.tier_manager.get_hot_snapshot()
            nodes.extend(hot)
        except Exception as e:
            logger.warning(f"ForgettingCurveManager: could not fetch HOT nodes: {e}")

        try:
            warm = await self.engine.tier_manager.list_warm(max_results=1000)
            nodes.extend(warm)
        except (AttributeError, Exception) as e:
            logger.debug(f"ForgettingCurveManager: WARM fetch skipped: {e}")

        entries = self.schedule_reviews(nodes)

        # Count actions
        from collections import Counter
        action_counts = dict(Counter(e.action for e in entries))

        logger.info(f"ForgettingCurveManager scan: {action_counts}")
        return {
            "nodes_scanned": len(nodes),
            "entries_scheduled": len(entries),
            "action_counts": action_counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Convenience import alias
from typing import Dict  # noqa: E402 (already imported above, just ensuring type hint works)
