"""
Confidence Calibration Module (Phase 5.0)
==========================================
Generates structured confidence envelopes for retrieved memories,
combining all available trust signals into a single queryable object.

Signals used:
  - BayesianLTP reliability (mean of Beta posterior)
  - access_count       (low count → less evidence)
  - staleness          (days since last verification)
  - source type        (external ≤ user_correction vs observation)
  - contradiction flag (from ProvenanceRecord)

Output: a ConfidenceEnvelope dict appended to every query response,
enabling consuming agents to make trust-aware decisions.

Public API:
    env = ConfidenceEnvelopeGenerator.build(node, provenance)
    level = env["level"]   # "high" | "medium" | "low" | "contradicted" | "stale"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .node import MemoryNode
    from .provenance import ProvenanceRecord


# ------------------------------------------------------------------ #
#  Confidence levels (ordered by trust)                              #
# ------------------------------------------------------------------ #

LEVEL_HIGH = "high"
LEVEL_MEDIUM = "medium"
LEVEL_LOW = "low"
LEVEL_CONTRADICTED = "contradicted"
LEVEL_STALE = "stale"

# Thresholds
RELIABILITY_HIGH_THRESHOLD = 0.80
RELIABILITY_MEDIUM_THRESHOLD = 0.50
ACCESS_COUNT_MIN_EVIDENCE = 2       # Less than this → low confidence
ACCESS_COUNT_HIGH_EVIDENCE = 5      # At least this → supports high confidence
STALENESS_STALE_DAYS = 30           # Days without verification → stale


# ------------------------------------------------------------------ #
#  Source-type trust weights                                         #
# ------------------------------------------------------------------ #

SOURCE_TRUST: Dict[str, float] = {
    "observation": 1.0,
    "inference": 0.8,
    "external_sync": 0.75,
    "dream": 0.6,
    "consolidation": 0.85,
    "prediction": 0.5,
    "user_correction": 1.0,
    "unknown": 0.5,
}


# ------------------------------------------------------------------ #
#  Confidence Envelope Generator                                     #
# ------------------------------------------------------------------ #

class ConfidenceEnvelopeGenerator:
    """
    Builds a confidence_envelope dict for a MemoryNode.

    Does NOT mutate the node — only reads fields.
    Thread-safe; no shared state.
    """

    @staticmethod
    def _reliability(node: "MemoryNode") -> float:
        """
        Extract reliability float from the node.
        Falls back to ltp_strength if no Bayesian state is attached.
        """
        bayes = getattr(node, "_bayes", None)
        if bayes is not None:
            return float(bayes.mean)
        return float(getattr(node, "ltp_strength", 0.5))

    @staticmethod
    def _staleness_days(node: "MemoryNode", provenance: Optional["ProvenanceRecord"]) -> float:
        """Days since last verification, or days since last access."""
        if provenance:
            # Find the most recent 'verified' event
            for evt in reversed(provenance.lineage):
                if evt.event == "verified" and evt.outcome is True:
                    try:
                        ts = datetime.fromisoformat(evt.timestamp)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        delta = datetime.now(timezone.utc) - ts
                        return delta.total_seconds() / 86400.0
                    except (ValueError, TypeError):
                        pass

        # Fall back to last_accessed on the node
        last = getattr(node, "last_accessed", None)
        if last is not None:
            if getattr(last, "tzinfo", None) is None:
                last = last.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last
            return delta.total_seconds() / 86400.0

        return 0.0

    @classmethod
    def build(
        cls,
        node: "MemoryNode",
        provenance: Optional["ProvenanceRecord"] = None,
    ) -> Dict[str, Any]:
        """
        Build a full confidence_envelope dict for the given node.

        Returns a dict suitable for direct JSON serialization.
        """
        reliability = cls._reliability(node)
        access_count: int = getattr(node, "access_count", 1)
        staleness: float = cls._staleness_days(node, provenance)

        # Determine source type for trust weighting
        source_type = "unknown"
        if provenance:
            source_type = provenance.origin.type
        source_trust = SOURCE_TRUST.get(source_type, 0.5)

        # Contradiction check
        is_contradicted = provenance.is_contradicted() if provenance else False

        # Last verified date (human-readable)
        last_verified: Optional[str] = None
        if provenance:
            for evt in reversed(provenance.lineage):
                if evt.event == "verified" and evt.outcome is True:
                    last_verified = evt.timestamp
                    break

        # ---- Determine level ------------------------------------ #
        if is_contradicted:
            level = LEVEL_CONTRADICTED
        elif staleness > STALENESS_STALE_DAYS:
            level = LEVEL_STALE
        elif (
            reliability >= RELIABILITY_HIGH_THRESHOLD
            and access_count >= ACCESS_COUNT_HIGH_EVIDENCE
            and source_trust >= 0.75
        ):
            level = LEVEL_HIGH
        elif reliability >= RELIABILITY_MEDIUM_THRESHOLD and access_count >= ACCESS_COUNT_MIN_EVIDENCE:
            level = LEVEL_MEDIUM
        else:
            level = LEVEL_LOW

        envelope: Dict[str, Any] = {
            "level": level,
            "reliability": round(reliability, 4),
            "access_count": access_count,
            "staleness_days": round(staleness, 1),
            "source_type": source_type,
            "source_trust": round(source_trust, 2),
            "is_contradicted": is_contradicted,
        }
        if last_verified:
            envelope["last_verified"] = last_verified

        return envelope


# ------------------------------------------------------------------ #
#  Convenience function                                              #
# ------------------------------------------------------------------ #

def build_confidence_envelope(
    node: "MemoryNode",
    provenance: Optional["ProvenanceRecord"] = None,
) -> Dict[str, Any]:
    """
    Module-level shortcut for ConfidenceEnvelopeGenerator.build().

    Args:
        node:        MemoryNode to evaluate.
        provenance:  Optional ProvenanceRecord for the node.

    Returns:
        confidence_envelope dict with level, reliability, staleness, etc.
    """
    return ConfidenceEnvelopeGenerator.build(node, provenance)
