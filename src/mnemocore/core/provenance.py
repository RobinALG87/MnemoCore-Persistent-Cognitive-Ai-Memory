"""
Provenance Tracking Module (Phase 5.0)
=======================================
W3C PROV-inspired source tracking for MnemoCore memories.

Tracks the full lifecycle of every MemoryNode:
  - origin: where/how the memory was created
  - lineage: ordered list of transformation events
  - version: incremented on each significant mutation

This is the foundation for:
  - Trust & audit trails (AI Governance)
  - Contradiction resolution
  - Memory-as-a-Service lineage API
  - Source reliability scoring

Public API:
    record = ProvenanceRecord.new(origin_type="observation", agent_id="agent-001")
    record.add_event("consolidated", source_memories=["mem_a", "mem_b"])
    serialized = record.to_dict()
    restored = ProvenanceRecord.from_dict(serialized)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ #
#  Origin types                                                       #
# ------------------------------------------------------------------ #

ORIGIN_TYPES = {
    "observation",      # Direct input from agent or user
    "inference",        # Derived/reasoned by LLM or engine
    "dream",            # Produced by SubconsciousAI dream cycle
    "consolidation",    # Result of SemanticConsolidation merge
    "external_sync",    # Fetched from external source (RSS, API, etc.)
    "user_correction",  # Explicit user override
    "prediction",       # Stored as a future prediction
}


# ------------------------------------------------------------------ #
#  Lineage event                                                      #
# ------------------------------------------------------------------ #

@dataclass
class LineageEvent:
    """
    A single step in a memory's transformation history.

    Examples:
        created       – initial storage
        accessed      – retrieved by a query
        consolidated  – merged into or from a proto-memory cluster
        verified      – reliability confirmed externally
        contradicted  – flagged as contradicting another memory
        updated       – content or metadata modified
        archived      – moved to COLD tier
        expired       – TTL reached or evicted
    """
    event: str
    timestamp: str  # ISO 8601
    actor: Optional[str] = None         # agent_id, "system", "user", etc.
    source_memories: List[str] = field(default_factory=list)  # for consolidation
    outcome: Optional[bool] = None      # for verification events
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "event": self.event,
            "timestamp": self.timestamp,
        }
        if self.actor is not None:
            d["actor"] = self.actor
        if self.source_memories:
            d["source_memories"] = self.source_memories
        if self.outcome is not None:
            d["outcome"] = self.outcome
        if self.notes:
            d["notes"] = self.notes
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LineageEvent":
        return cls(
            event=d["event"],
            timestamp=d["timestamp"],
            actor=d.get("actor"),
            source_memories=d.get("source_memories", []),
            outcome=d.get("outcome"),
            notes=d.get("notes"),
            extra=d.get("extra", {}),
        )


# ------------------------------------------------------------------ #
#  Origin                                                             #
# ------------------------------------------------------------------ #

@dataclass
class ProvenanceOrigin:
    """Where/how a memory was first created."""

    type: str                           # One of ORIGIN_TYPES
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    source_url: Optional[str] = None   # For external_sync
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type,
            "timestamp": self.timestamp,
        }
        if self.agent_id:
            d["agent_id"] = self.agent_id
        if self.session_id:
            d["session_id"] = self.session_id
        if self.source_url:
            d["source_url"] = self.source_url
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceOrigin":
        return cls(
            type=d.get("type", "observation"),
            agent_id=d.get("agent_id"),
            session_id=d.get("session_id"),
            source_url=d.get("source_url"),
            timestamp=d.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


# ------------------------------------------------------------------ #
#  ProvenanceRecord — the full provenance object on a MemoryNode     #
# ------------------------------------------------------------------ #

@dataclass
class ProvenanceRecord:
    """
    Full provenance object attached to a MemoryNode.

    Designed to be serialized into node.metadata["provenance"] for
    backward compatibility with existing storage layers.
    """

    origin: ProvenanceOrigin
    lineage: List[LineageEvent] = field(default_factory=list)
    version: int = 1
    confidence_source: str = "bayesian_ltp"  # How the confidence score is derived

    # ---- Factory methods ------------------------------------------ #

    @classmethod
    def new(
        cls,
        origin_type: str = "observation",
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source_url: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> "ProvenanceRecord":
        """Create a fresh ProvenanceRecord and log the 'created' event."""
        now = datetime.now(timezone.utc).isoformat()
        origin = ProvenanceOrigin(
            type=origin_type if origin_type in ORIGIN_TYPES else "observation",
            agent_id=agent_id,
            session_id=session_id,
            source_url=source_url,
            timestamp=now,
        )
        record = cls(origin=origin)
        record.add_event(
            event="created",
            actor=actor or agent_id or "system",
        )
        return record

    # ---- Mutation ------------------------------------------------- #

    def add_event(
        self,
        event: str,
        actor: Optional[str] = None,
        source_memories: Optional[List[str]] = None,
        outcome: Optional[bool] = None,
        notes: Optional[str] = None,
        **extra: Any,
    ) -> "ProvenanceRecord":
        """Append a new lineage event and bump the version counter."""
        evt = LineageEvent(
            event=event,
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            source_memories=source_memories or [],
            outcome=outcome,
            notes=notes,
            extra=extra,
        )
        self.lineage.append(evt)
        self.version += 1
        return self

    def mark_consolidated(
        self,
        source_memory_ids: List[str],
        actor: str = "consolidation_worker",
    ) -> "ProvenanceRecord":
        """Convenience wrapper for consolidation events."""
        return self.add_event(
            event="consolidated",
            actor=actor,
            source_memories=source_memory_ids,
        )

    def mark_verified(
        self,
        success: bool,
        actor: str = "system",
        notes: Optional[str] = None,
    ) -> "ProvenanceRecord":
        """Record a verification outcome."""
        return self.add_event(
            event="verified",
            actor=actor,
            outcome=success,
            notes=notes,
        )

    def mark_contradicted(
        self,
        contradiction_group_id: str,
        actor: str = "contradiction_detector",
    ) -> "ProvenanceRecord":
        """Flag this memory as contradicted."""
        return self.add_event(
            event="contradicted",
            actor=actor,
            contradiction_group_id=contradiction_group_id,
        )

    # ---- Serialization -------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin.to_dict(),
            "lineage": [e.to_dict() for e in self.lineage],
            "version": self.version,
            "confidence_source": self.confidence_source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceRecord":
        return cls(
            origin=ProvenanceOrigin.from_dict(d.get("origin", {"type": "observation"})),
            lineage=[LineageEvent.from_dict(e) for e in d.get("lineage", [])],
            version=d.get("version", 1),
            confidence_source=d.get("confidence_source", "bayesian_ltp"),
        )

    # ---- Helpers -------------------------------------------------- #

    @property
    def created_at(self) -> str:
        """ISO timestamp of the creation event."""
        for event in self.lineage:
            if event.event == "created":
                return event.timestamp
        return self.origin.timestamp

    @property
    def last_event(self) -> Optional[LineageEvent]:
        """Most recent lineage event."""
        return self.lineage[-1] if self.lineage else None

    def is_contradicted(self) -> bool:
        return any(e.event == "contradicted" for e in self.lineage)

    def is_verified(self) -> bool:
        return any(
            e.event == "verified" and e.outcome is True for e in self.lineage
        )

    def __repr__(self) -> str:
        return (
            f"ProvenanceRecord(origin_type={self.origin.type!r}, "
            f"version={self.version}, events={len(self.lineage)})"
        )
