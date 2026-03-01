"""
SM-2 State – SuperMemo 2 Algorithm State
=========================================
SM-2 algorithm state for a single memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, TYPE_CHECKING

from .config import SM2_DEFAULT_EASINESS, SM2_MIN_EASINESS
from .profile import LearningProfile

if TYPE_CHECKING:
    pass


@dataclass
class SM2State:
    """
    SM-2 algorithm state for a single memory.

    Tracks the SuperMemo 2 parameters for each memory to enable
    optimal spaced repetition scheduling.
    """
    memory_id: str
    repetitions: int = 0  # Number of successful reviews
    interval: float = 0.0  # Current interval in days
    easiness_factor: float = SM2_DEFAULT_EASINESS
    last_review_quality: int = 0
    last_review_date: Optional[datetime] = None
    next_review_date: Optional[datetime] = None

    def calculate_next_review(
        self,
        quality: int,
        profile: LearningProfile,
        current_date: Optional[datetime] = None
    ) -> "SM2State":
        """
        Calculate next review state based on SM-2 algorithm.

        Args:
            quality: Review quality (0-5)
            profile: Agent's learning profile
            current_date: Date of this review

        Returns new SM2State with updated values.
        """
        now = current_date or datetime.now(timezone.utc)

        # Update easiness factor
        new_ef = profile.easiness_factor
        new_ef = new_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        new_ef = max(profile.sm2_min_easiness, new_ef)

        # Update repetitions
        if quality >= 3:
            new_reps = self.repetitions + 1
            # Calculate new interval
            new_interval = profile.calculate_sm2_interval(
                new_reps, self.interval, new_ef
            )
        else:
            # Failed review - restart
            new_reps = 0
            new_interval = 0.0  # Due immediately

        # Calculate next review date
        if new_interval == 0:
            next_date = now  # Due immediately
        else:
            next_date = now + timedelta(days=new_interval)

        return SM2State(
            memory_id=self.memory_id,
            repetitions=new_reps,
            interval=new_interval,
            easiness_factor=round(new_ef, 3),
            last_review_quality=quality,
            last_review_date=now,
            next_review_date=next_date,
        )

    def is_due(self, now: Optional[datetime] = None) -> bool:
        """Check if this memory is due for review."""
        if self.next_review_date is None:
            return True  # Never reviewed
        now = now or datetime.now(timezone.utc)
        return now >= self.next_review_date

    def days_until_due(self, now: Optional[datetime] = None) -> float:
        """Days until next review (negative if overdue)."""
        if self.next_review_date is None:
            return 0.0
        now = now or datetime.now(timezone.utc)
        delta = self.next_review_date - now
        return delta.total_seconds() / 86400.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "memory_id": self.memory_id,
            "repetitions": self.repetitions,
            "interval": round(self.interval, 2),
            "easiness_factor": self.easiness_factor,
            "last_review_quality": self.last_review_quality,
            "last_review_date": self.last_review_date.isoformat() if self.last_review_date else None,
            "next_review_date": self.next_review_date.isoformat() if self.next_review_date else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SM2State":
        """Deserialize from dictionary."""
        data = data.copy()
        for key in ("last_review_date", "next_review_date"):
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


__all__ = ["SM2State"]
