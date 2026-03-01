"""
Prediction Memory Store (Phase 5.0 — Agent 4)
==============================================
Stores explicitly made predictions about future events and tracks their outcomes.

A prediction has a lifecycle:
    pending → verified (correct) OR falsified (wrong) OR expired (deadline passed)

Key behaviors:
  - Verified predictions STRENGTHEN related strategic memories via synaptic binding
  - Falsified predictions REDUCE confidence on related memories + generate a
    "lesson learned" via SubconsciousAI
  - Expired predictions are flagged for manual review

Backed by a lightweight in-memory + provenance-attached store.
For persistence, predictions are serialized to node.metadata["prediction"] and
stored as regular MemoryNodes in the HOT tier with a special tag.

Public API:
    store = PredictionStore()
    pred_id = store.create(content="...", confidence=0.7, deadline_days=90)
    store.verify(pred_id, success=True, notes="EU AI Act enforced")
    due = store.get_due()   # predictions past their deadline
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger


# ------------------------------------------------------------------ #
#  Prediction status constants                                       #
# ------------------------------------------------------------------ #

STATUS_PENDING = "pending"
STATUS_VERIFIED = "verified"
STATUS_FALSIFIED = "falsified"
STATUS_EXPIRED = "expired"


# ------------------------------------------------------------------ #
#  PredictionRecord                                                  #
# ------------------------------------------------------------------ #

@dataclass
class PredictionRecord:
    """A single forward-looking prediction stored in MnemoCore."""

    id: str = field(default_factory=lambda: f"pred_{uuid.uuid4().hex[:16]}")
    content: str = ""
    predicted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    verification_deadline: Optional[str] = None  # ISO datetime string
    confidence_at_creation: float = 0.5
    status: str = STATUS_PENDING
    outcome: Optional[bool] = None          # True=verified, False=falsified
    verification_notes: Optional[str] = None
    verified_at: Optional[str] = None
    related_memory_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """True if the deadline has passed and status is still pending."""
        if self.status != STATUS_PENDING or self.verification_deadline is None:
            return False
        deadline = datetime.fromisoformat(self.verification_deadline)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > deadline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "predicted_at": self.predicted_at,
            "verification_deadline": self.verification_deadline,
            "confidence_at_creation": round(self.confidence_at_creation, 4),
            "status": self.status,
            "outcome": self.outcome,
            "verification_notes": self.verification_notes,
            "verified_at": self.verified_at,
            "related_memory_ids": self.related_memory_ids,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionRecord":
        return cls(
            id=d.get("id", f"pred_{uuid.uuid4().hex[:16]}"),
            content=d.get("content", ""),
            predicted_at=d.get("predicted_at", datetime.now(timezone.utc).isoformat()),
            verification_deadline=d.get("verification_deadline"),
            confidence_at_creation=d.get("confidence_at_creation", 0.5),
            status=d.get("status", STATUS_PENDING),
            outcome=d.get("outcome"),
            verification_notes=d.get("verification_notes"),
            verified_at=d.get("verified_at"),
            related_memory_ids=d.get("related_memory_ids", []),
            tags=d.get("tags", []),
        )


# ------------------------------------------------------------------ #
#  PredictionStore                                                   #
# ------------------------------------------------------------------ #

class PredictionStore:
    """
    In-memory store for PredictionRecords with lifecycle management.

    For production use, wire to an engine so verified/falsified predictions
    can update related MemoryNode synapses and generate LLM insights.
    """

    def __init__(self, engine=None) -> None:
        self.engine = engine
        self._records: Dict[str, PredictionRecord] = {}

    # ---- CRUD ---------------------------------------------------- #

    def create(
        self,
        content: str,
        confidence: float = 0.5,
        deadline_days: Optional[float] = None,
        deadline: Optional[datetime] = None,
        related_memory_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store a new prediction.

        Args:
            content:           The prediction statement.
            confidence:        Confidence at creation time [0, 1].
            deadline_days:     Days from now until deadline (alternative to deadline).
            deadline:          Explicit deadline datetime (overrides deadline_days).
            related_memory_ids: IDs of memories this prediction relates to.
            tags:              Optional classification tags.

        Returns:
            The prediction ID.
        """
        deadline_iso: Optional[str] = None
        if deadline is not None:
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            deadline_iso = deadline.isoformat()
        elif deadline_days is not None:
            deadline_iso = (
                datetime.now(timezone.utc) + timedelta(days=deadline_days)
            ).isoformat()

        rec = PredictionRecord(
            content=content,
            confidence_at_creation=max(0.0, min(1.0, confidence)),
            verification_deadline=deadline_iso,
            related_memory_ids=related_memory_ids or [],
            tags=tags or [],
        )
        self._records[rec.id] = rec
        logger.info(
            f"Prediction created: {rec.id} | confidence={confidence:.2f} | "
            f"deadline={deadline_iso or 'none'}"
        )
        return rec.id

    def get(self, pred_id: str) -> Optional[PredictionRecord]:
        return self._records.get(pred_id)

    def list_all(self, status: Optional[str] = None) -> List[PredictionRecord]:
        """Return all predictions, optionally filtered by status."""
        recs = list(self._records.values())
        if status:
            recs = [r for r in recs if r.status == status]
        return sorted(recs, key=lambda r: r.predicted_at, reverse=True)

    def get_due(self) -> List[PredictionRecord]:
        """Return pending predictions that have passed their deadline."""
        return [r for r in self._records.values() if r.is_expired()]

    # ---- Lifecycle ----------------------------------------------- #

    async def verify(
        self,
        pred_id: str,
        success: bool,
        notes: Optional[str] = None,
    ) -> Optional[PredictionRecord]:
        """
        Verify or falsify a prediction.

        Side effects:
          - Verified: strengthens related memories via synaptic binding
          - Falsified: reduces confidence on related memories + lesson learned
        """
        rec = self._records.get(pred_id)
        if rec is None:
            logger.warning(f"PredictionStore.verify: unknown id {pred_id!r}")
            return None

        rec.status = STATUS_VERIFIED if success else STATUS_FALSIFIED
        rec.outcome = success
        rec.verification_notes = notes
        rec.verified_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Prediction {pred_id} → {rec.status} | notes={notes or '—'}"
        )

        if self.engine is not None:
            if success:
                await self._strengthen_related(rec)
            else:
                await self._weaken_related(rec)
                await self._generate_lesson(rec)

        return rec

    async def expire_due(self) -> List[PredictionRecord]:
        """Mark overdue pending predictions as expired. Returns expired list."""
        due = self.get_due()
        for rec in due:
            rec.status = STATUS_EXPIRED
            logger.info(f"Prediction {rec.id} expired (deadline passed).")
        return due

    # ---- Engine integration -------------------------------------- #

    async def _strengthen_related(self, rec: PredictionRecord) -> None:
        """Verified prediction → strengthen synapses on related memories."""
        for mem_id in rec.related_memory_ids:
            try:
                node = await self.engine.get_memory(mem_id)
                if node:
                    si = getattr(self.engine, "synapse_index", None)
                    if si:
                        si.add_or_strengthen(rec.id, mem_id, delta=0.15)
                    logger.debug(f"Prediction {rec.id}: strengthened memory {mem_id[:8]}")
            except Exception as exc:
                logger.warning(f"Prediction {rec.id}: strengthen_related failed for memory {mem_id}: {exc}")

    async def _weaken_related(self, rec: PredictionRecord) -> None:
        """Falsified prediction → reduce confidence on related memories."""
        for mem_id in rec.related_memory_ids:
            try:
                node = await self.engine.get_memory(mem_id)
                if node:
                    from .bayesian_ltp import get_bayesian_updater
                    updater = get_bayesian_updater()
                    updater.observe_node_retrieval(node, helpful=False, eig_signal=0.5)
                    logger.debug(f"Prediction {rec.id}: weakened memory {mem_id[:8]}")
            except Exception as exc:
                logger.warning(f"Prediction {rec.id}: weaken_related failed for memory {mem_id}: {exc}")

    async def _generate_lesson(self, rec: PredictionRecord) -> None:
        """Ask SubconsciousAI to synthesize a 'lesson learned' for a falsified prediction."""
        try:
            subcon = getattr(self.engine, "subconscious_ai", None)
            if subcon is None:
                return
            prompt = (
                f"The following prediction was FALSIFIED: '{rec.content}'. "
                f"Confidence at creation: {rec.confidence_at_creation:.2f}. "
                f"Notes: {rec.verification_notes or 'none'}. "
                "In 1-2 sentences, what is the key lesson learned from this failure?"
            )
            lesson = await subcon.generate(prompt, max_tokens=128)
            # Store the lesson as a new memory
            await self.engine.store(
                lesson.strip(),
                metadata={
                    "type": "lesson_learned",
                    "source_prediction_id": rec.id,
                    "domain": "strategic",
                }
            )
            logger.info(f"Lesson learned generated for falsified prediction {rec.id}")
        except Exception as exc:
            logger.warning(f"Prediction {rec.id}: _generate_lesson failed: {exc}")

    # ---- Serialization ------------------------------------------- #

    def to_list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records.values()]

    def __len__(self) -> int:
        return len(self._records)
