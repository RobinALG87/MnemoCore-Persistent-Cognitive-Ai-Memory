from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import math

from .binary_hdv import BinaryHDV
from .config import get_config


@dataclass
class MemoryNode:
    """
    Holographic memory neuron (Phase 3.0+).
    Uses BinaryHDV for efficient storage and computation.

    Phase 4.3: Temporal Recall - supports episodic chaining and time-based indexing.
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

    def access(self, update_weights: bool = True):
        """Retrieve memory (reconsolidation)"""
        now = datetime.now(timezone.utc)
        self.last_accessed = now

        if update_weights:
            self.access_count += 1
            # Decay old strength first? Or just recalculate?
            # We recalculate based on new access count
            self.calculate_ltp()

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
