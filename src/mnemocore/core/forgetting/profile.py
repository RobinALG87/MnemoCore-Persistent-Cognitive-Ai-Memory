"""
Learning Profile – Per-Agent Learning Configuration
====================================================
Per-agent learning profile with personalized decay and repetition parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

from loguru import logger

from .config import (
    SM2_DEFAULT_EASINESS,
    SM2_MIN_EASINESS,
)

if TYPE_CHECKING:
    from ..node import MemoryNode


@dataclass
class LearningProfile:
    """
    Per-agent learning profile with personalized decay and repetition parameters.

    Each agent (e.g., different AI personas, users, or subsystems) can have
    different learning characteristics. Some agents may be "fast learners"
    requiring less frequent review, while others may need more reinforcement.

    Attributes:
        agent_id: Unique identifier for the agent
        base_decay: Base decay constant (higher = slower decay)
        easiness_factor: SM-2 easiness factor (higher = easier time remembering)
        review_frequency_multiplier: Multiplier for review intervals
        emotional_sensitivity: How much emotions affect memory (0-1)
        learning_style: Description of learning preferences
        stats: Cumulative statistics
        created_at: When this profile was created
        last_updated: Last time profile was modified
    """
    agent_id: str
    base_decay: float = 1.0  # Base stability modifier
    easiness_factor: float = SM2_DEFAULT_EASINESS
    review_frequency_multiplier: float = 1.0
    emotional_sensitivity: float = 0.7  # How much emotion affects decay
    learning_style: str = "balanced"
    stats: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # SM-2 specific parameters
    sm2_min_easiness: float = SM2_MIN_EASINESS
    sm2_interval_modifier: float = 1.0  # Adjusts interval growth

    def __post_init__(self):
        if not self.stats:
            self.stats = {
                "total_reviews": 0,
                "successful_reviews": 0,
                "avg_quality": 0.0,
                "memories_tracked": 0,
                "emotional_memories": 0,
            }

    def get_decay_modifier(self, node: Optional["MemoryNode"] = None) -> float:
        """
        Calculate the decay modifier for this agent.

        Returns a multiplier where:
        - > 1.0 means slower decay (better retention)
        - < 1.0 means faster decay (worse retention)
        """
        modifier = self.base_decay

        # Apply easiness factor influence
        modifier *= (self.easiness_factor / SM2_DEFAULT_EASINESS)

        return max(0.1, min(5.0, modifier))

    def get_emotional_modifier(self, salience: float) -> float:
        """
        Calculate emotional decay modifier based on salience.

        Higher salience = slower decay (emotional memories persist longer).
        """
        # Modifier ranges from 1.0 (no effect) to 2.0 (half decay rate)
        # based on emotional_sensitivity and salience
        boost = 1.0 + (self.emotional_sensitivity * salience)
        return min(2.0, boost)

    def update_stats(self, quality: int, is_emotional: bool = False) -> None:
        """Update learning statistics after a review."""
        self.stats["total_reviews"] += 1
        if quality >= 3:  # Successful recall
            self.stats["successful_reviews"] += 1

        # Update average quality using running average
        old_avg = self.stats.get("avg_quality", 0.0)
        n = self.stats["total_reviews"]
        new_avg = (old_avg * (n - 1) + quality) / n
        self.stats["avg_quality"] = round(new_avg, 3)

        if is_emotional:
            self.stats["emotional_memories"] += 1

        self.last_updated = datetime.now(timezone.utc)

    def adjust_easiness(self, quality: int) -> float:
        """
        Adjust SM-2 easiness factor based on review quality.

        EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        """
        q = quality
        ef = self.easiness_factor

        # SM-2 formula for easiness adjustment
        new_ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

        # Clamp to minimum
        self.easiness_factor = max(self.sm2_min_easiness, new_ef)
        self.last_updated = datetime.now(timezone.utc)

        return self.easiness_factor

    def calculate_sm2_interval(
        self,
        repetitions: int,
        previous_interval: float,
        easiness: Optional[float] = None
    ) -> float:
        """
        Calculate next review interval using SM-2 algorithm.

        I(1) = 1 day
        I(2) = 6 days
        I(n) = I(n-1) * EF for n > 2

        Returns interval in days.
        """
        ef = easiness or self.easiness_factor

        if repetitions == 0:
            return 0.0  # Due immediately
        elif repetitions == 1:
            return 1.0 * self.sm2_interval_modifier
        elif repetitions == 2:
            return 6.0 * self.sm2_interval_modifier
        else:
            return previous_interval * ef * self.sm2_interval_modifier

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "base_decay": self.base_decay,
            "easiness_factor": round(self.easiness_factor, 3),
            "review_frequency_multiplier": self.review_frequency_multiplier,
            "emotional_sensitivity": self.emotional_sensitivity,
            "learning_style": self.learning_style,
            "stats": self.stats,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningProfile":
        """Deserialize from dictionary."""
        data = data.copy()
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_updated" in data and isinstance(data["last_updated"], str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)

    @classmethod
    def for_agent(cls, agent_id: str, learning_style: str = "balanced") -> "LearningProfile":
        """Factory to create a profile with a specific learning style."""
        profiles = {
            "fast": {
                "base_decay": 1.5,
                "easiness_factor": 3.0,
                "review_frequency_multiplier": 1.5,
                "learning_style": "fast_learner",
            },
            "slow": {
                "base_decay": 0.7,
                "easiness_factor": 2.0,
                "review_frequency_multiplier": 0.8,
                "learning_style": "needs_reinforcement",
            },
            "visual": {
                "base_decay": 1.0,
                "easiness_factor": 2.7,
                "emotional_sensitivity": 0.9,
                "learning_style": "visual_emotional",
            },
            "analytical": {
                "base_decay": 1.2,
                "easiness_factor": 2.8,
                "emotional_sensitivity": 0.4,
                "learning_style": "analytical",
            },
            "balanced": {
                "base_decay": 1.0,
                "easiness_factor": SM2_DEFAULT_EASINESS,
                "emotional_sensitivity": 0.7,
                "learning_style": "balanced",
            },
        }

        config = profiles.get(learning_style, profiles["balanced"])
        return cls(agent_id=agent_id, **config)


__all__ = ["LearningProfile"]
