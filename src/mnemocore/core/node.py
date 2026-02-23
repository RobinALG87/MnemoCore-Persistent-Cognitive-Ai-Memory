from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Any, Optional
import math

from .binary_hdv import BinaryHDV
from .config import get_config

if TYPE_CHECKING:
    from .provenance import ProvenanceRecord


@dataclass
class MemoryNode:
    """
    Holographic memory neuron (Phase 3.0+).
    Uses BinaryHDV for efficient storage and computation.

    Phase 4.3: Temporal Recall - supports episodic chaining and time-based indexing.
    Phase 6.0: Embedding Version Registry - tracks embedding model version.
    """

    id: str
    hdv: BinaryHDV
    content: str  # Original text/data
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Phase 3.0: Tiering & LTP
    tier: str = "hot"  # "hot", "warm", "cold"
    access_count: int = 1
    ltp_strength: float = 0.5  # Current retrieval strength

    # Legacy Free Energy signals (mapped to importance)
    epistemic_value: float = 0.0  # Reduces uncertainty?
    pragmatic_value: float = 0.0  # Helps achieve goals?

    # Phase 4.3: Episodic Chaining - links to temporally adjacent memories
    previous_id: Optional[str] = None  # UUID of the memory created immediately before this one

    # Phase 5.0 — Agent 1: Trust & Provenance
    provenance: Optional["ProvenanceRecord"] = field(default=None, repr=False)

    # Phase 5.0 — Agent 2: Adaptive Temporal Decay
    # Per-memory stability: S_i = S_base * (1 + k * access_count)
    # Starts at 1.0; increases logarithmically on access.
    stability: float = 1.0
    review_candidate: bool = False  # Set by ForgettingCurveManager when near decay threshold

    # Phase 6.0 — Embedding Version Registry
    # Tracks which embedding model version generated this vector
    embedding_model_id: str = "binary_hdv"  # Model identifier
    embedding_version: int = 1  # Model version number
    embedding_checksum: str = ""  # SHA-256 checksum of model config (empty = unknown/default)

    def access(self, update_weights: bool = True):
        """Retrieve memory (reconsolidation)"""
        now = datetime.now(timezone.utc)
        self.last_accessed = now

        if update_weights:
            self.access_count += 1
            # Decay old strength first? Or just recalculate?
            # We recalculate based on new access count
            self.calculate_ltp()

            # Phase 5.0: update per-memory stability on each successful access
            # S_i grows logarithmically so older frequently-accessed memories are more stable
            import math as _math
            self.stability = max(1.0, 1.0 + _math.log1p(self.access_count) * 0.5)

            # Legacy updates
            self.epistemic_value *= 1.01
            self.epistemic_value = min(self.epistemic_value, 1.0)

    def calculate_ltp(self) -> float:
        """
        Calculate Long-Term Potentiation (LTP) strength.
        Formula: S = I * log(1 + A) * e^(-lambda * T)
        """
        config = get_config()
        
        # I = Importance (derived from legacy values or default)
        importance = max(
            config.ltp.initial_importance,
            (self.epistemic_value + self.pragmatic_value) / 2
        )
        
        # A = Access count
        access_factor = math.log1p(self.access_count)
        
        # T = Time since creation (days)
        age = self.age_days()
        
        # Decay
        decay = math.exp(-config.ltp.decay_lambda * age)
        
        self.ltp_strength = importance * access_factor * decay
        
        # Clamp? No, it can grow. But maybe clamp for meaningful comparison.
        # Check permanence threshold
        if self.ltp_strength > config.ltp.permanence_threshold:
            # Prevent decay below threshold if verified permanent?
            # For now just let it be high.
            pass
            
        return self.ltp_strength

    def get_free_energy_score(self) -> float:
        """
        Legacy score, now aliased to LTP strength for compatibility.
        """
        # If LTP hasn't been calculated recently, do it now
        return self.calculate_ltp()

    def age_days(self) -> float:
        """Age of memory in days (for decay calculations)"""
        # Use timezone-aware now
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 86400.0

    @property
    def unix_timestamp(self) -> int:
        """Unix timestamp (seconds since epoch) for Qdrant indexing."""
        return int(self.created_at.timestamp())

    @property
    def iso_date(self) -> str:
        """ISO 8601 date string for human-readable time metadata."""
        return self.created_at.isoformat()

    def age_seconds(self) -> float:
        """Age of memory in seconds (for fine-grained chrono-weighting)."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds()

    def __lt__(self, other):
        # Sort by LTP strength descending? No, __lt__ is valid for sorting.
        # Default sort by ID is fine, but for priority queues we might want LTP.
        # Let's keep ID for stability and use key= attr for sorting.
        return self.id < other.id

    # ------------------------------------------------------------------
    # Phase 6.0: Embedding Version Management
    # ------------------------------------------------------------------

    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get embedding version information as a dictionary.

        Returns:
            Dict with embedding_model_id, embedding_version, embedding_checksum
        """
        return {
            "embedding_model_id": self.embedding_model_id,
            "embedding_version": self.embedding_version,
            "embedding_checksum": self.embedding_checksum,
        }

    def is_embedding_compatible(
        self,
        target_model_id: str,
        target_version: int,
        target_checksum: Optional[str] = None
    ) -> bool:
        """
        Check if this node's embedding is compatible with a target model.

        Args:
            target_model_id: Target model identifier
            target_version: Target model version
            target_checksum: Optional target checksum for exact match

        Returns:
            True if the embedding matches the target specification
        """
        if self.embedding_model_id != target_model_id:
            return False
        if self.embedding_version != target_version:
            return False
        if target_checksum is not None and self.embedding_checksum:
            return self.embedding_checksum == target_checksum
        return True

    def needs_migration(self, target_spec: "EmbeddingModelSpec") -> bool:
        """
        Check if this node needs to be migrated to a target model.

        Args:
            target_spec: Target EmbeddingModelSpec

        Returns:
            True if migration is needed
        """
        return not self.is_embedding_compatible(
            target_spec.model_id,
            target_spec.version,
            target_spec.checksum
        )

    def update_embedding_info(
        self,
        model_id: str,
        version: int,
        checksum: str
    ):
        """
        Update embedding version information (after re-embedding).

        Args:
            model_id: New embedding model ID
            version: New embedding version
            checksum: New embedding checksum
        """
        self.embedding_model_id = model_id
        self.embedding_version = version
        self.embedding_checksum = checksum

    @property
    def embedding_qualified_id(self) -> str:
        """
        Get the full qualified embedding identifier.

        Returns: "{model_id}:v{version}:{checksum[:8]}"
        """
        checksum_short = self.embedding_checksum[:8] if self.embedding_checksum else "unknown"
        return f"{self.embedding_model_id}:v{self.embedding_version}:{checksum_short}"
