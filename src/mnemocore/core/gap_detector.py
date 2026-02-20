"""
Knowledge Gap Detection — Proactive Curiosity Engine (Phase 4.0)
================================================================
Detects "gaps" in the memory system: topics the engine has been queried
about but lacks sufficient high-quality information to answer confidently.

Gap detection signals:
  1. Low retrieval confidence: top-k query returns results with average
     similarity below a threshold (engine didn't know much about it).
  2. Sparse coverage: fewer than min_results candidates were retrieved.
  3. High EIG residual: the XOR attention mask still has high Hamming
     entropy after retrieval (the query dimension is underrepresented).
  4. Explicit negative feedback: caller marks a retrieval as unhelpful.

Each detected gap is stored as a GapRecord in the gap registry.
The registry is periodically inspected by the gap-filling LLM component.

Public API:
    detector = GapDetector(engine)

    # After a query:
    gaps = await detector.assess_query(query_text, results, attention_mask)

    # Register explicit negative feedback:
    await detector.register_negative_feedback(query_text)

    # Retrieve open gaps (for LLM filling):
    open_gaps = detector.get_open_gaps(top_n=10)

    # Mark gap as filled:
    detector.mark_filled(gap_id)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .binary_hdv import BinaryHDV

# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #


@dataclass
class GapDetectorConfig:
    """Tuning knobs for gap detection sensitivity."""

    min_confidence_threshold: float = 0.45  # avg top-k similarity below this → gap
    min_results_required: int = 2  # fewer results than this → sparse gap
    mask_entropy_threshold: float = 0.46  # XOR mask entropy above this → coverage gap
    negative_feedback_weight: float = 2.0  # multiplier for explicit negative signals
    gap_ttl_seconds: float = 86400 * 7  # auto-expire gaps after 7 days
    max_gap_registry_size: int = 500  # cap registry to prevent unbounded growth
    enabled: bool = True


# ------------------------------------------------------------------ #
#  Gap record                                                         #
# ------------------------------------------------------------------ #


@dataclass
class GapRecord:
    """A single detected knowledge gap."""

    gap_id: str
    query_text: str
    detected_at: datetime
    last_seen: datetime
    signal: str  # "low_confidence" | "sparse" | "coverage" | "negative"
    confidence: float  # retrieval confidence at detection time
    seen_count: int = 1
    filled: bool = False
    filled_at: Optional[datetime] = None
    priority_score: float = 0.0  # higher = fill sooner

    def update_priority(self) -> None:
        """Recompute priority from recency + frequency."""
        age_hours = (
            datetime.now(timezone.utc) - self.detected_at
        ).total_seconds() / 3600.0
        recency = 1.0 / (1.0 + age_hours / 24.0)  # decays over days
        frequency = min(1.0, self.seen_count / 10.0)
        self.priority_score = (
            0.5 * recency + 0.3 * frequency + 0.2 * (1.0 - self.confidence)
        )


def _query_id(query_text: str) -> str:
    """Stable short ID for a query string."""
    return hashlib.shake_256(query_text.lower().strip().encode()).hexdigest(8)


def _bit_entropy_packed(hdv: BinaryHDV) -> float:
    """Shannon entropy of bit distribution in [0, 1] (1 = perfectly balanced)."""
    import numpy as np

    bits = __import__("numpy").unpackbits(hdv.data)
    p = float(bits.sum()) / len(bits)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * __import__("math").log2(p) + (1 - p) * __import__("math").log2(1 - p))


# ------------------------------------------------------------------ #
#  Detector                                                           #
# ------------------------------------------------------------------ #


class GapDetector:
    """
    Detects and tracks knowledge gaps from query telemetry.
    """

    def __init__(self, config: Optional[GapDetectorConfig] = None):
        self.cfg = config or GapDetectorConfig()
        self._registry: Dict[str, GapRecord] = {}  # gap_id → GapRecord

    # ---- Main assessment ----------------------------------------- #

    async def assess_query(
        self,
        query_text: str,
        results: List[Tuple[str, float]],
        attention_mask: Optional[BinaryHDV] = None,
    ) -> List[GapRecord]:
        """
        Assess a completed query for knowledge gaps.

        Args:
            query_text: The original query string.
            results: List of (node_id, similarity_score) from the engine.
            attention_mask: Optional XOR attention mask for coverage analysis.

        Returns:
            List of newly created or updated GapRecord instances.
        """
        if not self.cfg.enabled:
            return []

        detected: List[GapRecord] = []

        # Signal 1: low confidence
        if results:
            avg_score = sum(s for _, s in results) / len(results)
        else:
            avg_score = 0.0

        if avg_score < self.cfg.min_confidence_threshold:
            rec = self._upsert_gap(
                query_text,
                signal="low_confidence",
                confidence=avg_score,
            )
            detected.append(rec)

        # Signal 2: sparse retrieval
        if len(results) < self.cfg.min_results_required:
            rec = self._upsert_gap(
                query_text,
                signal="sparse",
                confidence=avg_score,
            )
            if rec not in detected:
                detected.append(rec)

        # Signal 3: XOR attention mask entropy (coverage gap)
        if attention_mask is not None:
            entropy = _bit_entropy_packed(attention_mask)
            if entropy > self.cfg.mask_entropy_threshold:
                rec = self._upsert_gap(
                    query_text,
                    signal="coverage",
                    confidence=avg_score,
                )
                if rec not in detected:
                    detected.append(rec)

        for rec in detected:
            rec.update_priority()
            logger.debug(
                f"Gap detected [{rec.signal}]: '{query_text[:60]}' "
                f"(conf={avg_score:.3f} priority={rec.priority_score:.3f})"
            )

        self._evict_stale()
        return detected

    async def register_negative_feedback(self, query_text: str) -> GapRecord:
        """
        Explicitly register that a retrieval was unhelpful (user feedback).
        Results in a HIGH-priority gap record.
        """
        gap_id = _query_id(query_text)
        if gap_id in self._registry:
            rec = self._registry[gap_id]
            rec.seen_count += int(self.cfg.negative_feedback_weight)
        else:
            rec = GapRecord(
                gap_id=gap_id,
                query_text=query_text,
                detected_at=datetime.now(timezone.utc),
                last_seen=datetime.now(timezone.utc),
                signal="negative",
                confidence=0.0,
                seen_count=int(self.cfg.negative_feedback_weight),
            )
            self._registry[gap_id] = rec

        rec.update_priority()
        rec.priority_score = min(
            1.0, rec.priority_score * self.cfg.negative_feedback_weight
        )
        logger.info(f"Negative feedback registered for gap '{query_text[:60]}'")
        return rec

    # ---- Registry management ------------------------------------- #

    def get_open_gaps(self, top_n: int = 10) -> List[GapRecord]:
        """Return the top-N highest-priority unfilled gaps."""
        open_gaps = [g for g in self._registry.values() if not g.filled]
        open_gaps.sort(key=lambda g: g.priority_score, reverse=True)
        return open_gaps[:top_n]

    def get_all_gaps(self) -> List[GapRecord]:
        """Return all gap records (including filled)."""
        return list(self._registry.values())

    def mark_filled(self, gap_id: str) -> bool:
        """Mark a gap as filled (e.g., after LLM has stored an answer)."""
        if gap_id in self._registry:
            self._registry[gap_id].filled = True
            self._registry[gap_id].filled_at = datetime.now(timezone.utc)
            logger.info(f"Gap {gap_id} marked as filled.")
            return True
        return False

    @property
    def stats(self) -> Dict:
        total = len(self._registry)
        filled = sum(1 for g in self._registry.values() if g.filled)
        return {
            "total_gaps": total,
            "open_gaps": total - filled,
            "filled_gaps": filled,
        }

    # ---- Internal helpers ---------------------------------------- #

    def _upsert_gap(self, query_text: str, signal: str, confidence: float) -> GapRecord:
        gap_id = _query_id(query_text)
        now = datetime.now(timezone.utc)

        if gap_id in self._registry:
            rec = self._registry[gap_id]
            rec.seen_count += 1
            rec.last_seen = now
            rec.confidence = min(rec.confidence, confidence)
            # Re-open if previously (incorrectly) filled
            if rec.filled and confidence < self.cfg.min_confidence_threshold:
                rec.filled = False
        else:
            rec = GapRecord(
                gap_id=gap_id,
                query_text=query_text,
                detected_at=now,
                last_seen=now,
                signal=signal,
                confidence=confidence,
            )
            self._registry[gap_id] = rec

        return rec

    def _evict_stale(self) -> None:
        """Remove expired gap records to keep registry bounded."""
        now_ts = time.time()
        stale = [
            gid
            for gid, rec in self._registry.items()
            if (
                rec.filled
                and (now_ts - rec.last_seen.timestamp()) > self.cfg.gap_ttl_seconds
            )
            or (now_ts - rec.last_seen.timestamp()) > self.cfg.gap_ttl_seconds * 2
        ]
        for gid in stale:
            del self._registry[gid]

        # Hard cap
        if len(self._registry) > self.cfg.max_gap_registry_size:
            sorted_gaps = sorted(
                self._registry.items(), key=lambda x: x[1].priority_score
            )
            for gid, _ in sorted_gaps[
                : len(sorted_gaps) - self.cfg.max_gap_registry_size
            ]:
                del self._registry[gid]
