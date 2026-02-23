"""
Forgetting Curve Manager (Phase 5.0 - Enhanced)
================================================
Implements Ebbinghaus-based spaced repetition scheduling with SM-2 algorithm,
per-agent learning profiles, and emotion-aware decay for MnemoCore.

Enhanced Features:
  1. Per-agent LearningProfile with personalized decay constants
  2. SM-2 (SuperMemo 2) spaced repetition algorithm
  3. Emotional memory integration (via emotional_tag.py)
  4. ForgettingAnalytics dashboard for visualization

Key Algorithms:
  - SM-2: I(n+1) = I(n) * EF where EF is the easiness factor
  - Retention: R = e^(-T / S_i) * emotional_boost
  - Stability: S_i = S_base * (1 + k * access_count) * agent_modifier

Public API:
    manager = ForgettingCurveManager(engine)
    profile = manager.get_learning_profile(agent_id)
    await manager.record_review(node_id, quality=4)  # SM-2 quality 0-5
    analytics = ForgettingAnalytics(manager)
    dashboard = analytics.get_dashboard_data()
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from collections import defaultdict
from enum import Enum

from loguru import logger
import numpy as np

from .temporal_decay import AdaptiveDecayEngine, get_adaptive_decay_engine

if TYPE_CHECKING:
    from .node import MemoryNode
    from .engine import HAIMEngine


# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

# SM-2 Algorithm Constants
SM2_MIN_EASINESS: float = 1.3  # Minimum easiness factor (EF)
SM2_DEFAULT_EASINESS: float = 2.5  # Starting easiness factor
SM2_QUALITY_MIN: int = 0  # Worst recall quality
SM2_QUALITY_MAX: int = 5  # Perfect recall

# Retention thresholds
TARGET_RETENTION: float = 0.70  # Target retention at next review
MIN_EIG_TO_CONSOLIDATE: float = 0.3  # Minimum epistemic value to consolidate

# Emotional memory modifiers
EMOTIONAL_DECAY_REDUCTION: float = 0.5  # 50% slower decay for emotional memories
HIGH_SALIENCE_THRESHOLD: float = 0.5  # Salience threshold for "emotional" status

# Analytics defaults
ANALYTICS_HISTORY_SIZE: int = 1000  # Max historical records to keep


# ------------------------------------------------------------------ #
#  Enums                                                              #
# ------------------------------------------------------------------ #

class ReviewAction(Enum):
    """Actions that can be taken on a memory during review."""
    REVIEW = "review"  # Standard spaced repetition review
    CONSOLIDATE = "consolidate"  # Merge into stronger anchor
    EVICT = "evict"  # Remove from memory
    BOOST = "boost"  # Strengthen without full review (emotionally significant)


class ReviewQuality(Enum):
    """SM-2 quality ratings for memory recall."""
    COMPLETE_BLACKOUT = 0  # Total failure
    INCORRECT_BUT_RECOGNIZED = 1  # Wrong but familiar
    INCORRECT_EASY_RECALL = 2  # Wrong but easily remembered
    CORRECT_DIFFICULT = 3  # Correct but difficult
    CORRECT_HESITATION = 4  # Correct with hesitation
    PERFECT_RECALL = 5  # Perfect, instant recall


# ------------------------------------------------------------------ #
#  Learning Profile (Per-Agent)                                       #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  SM-2 Review State                                                  #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Review Schedule Entry                                             #
# ------------------------------------------------------------------ #

@dataclass
class ReviewEntry:
    """A scheduled review for a single memory."""
    memory_id: str
    agent_id: str
    due_at: datetime
    current_retention: float
    stability: float
    sm2_state: Optional[SM2State] = None
    emotional_salience: float = 0.0
    action: str = "review"

    def to_dict(self) -> Dict:
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "due_at": self.due_at.isoformat(),
            "current_retention": round(self.current_retention, 4),
            "stability": round(self.stability, 4),
            "sm2_state": self.sm2_state.to_dict() if self.sm2_state else None,
            "emotional_salience": round(self.emotional_salience, 4),
            "action": self.action,
        }


# ------------------------------------------------------------------ #
#  Forgetting Curve Manager                                          #
# ------------------------------------------------------------------ #

class ForgettingCurveManager:
    """
    Enhanced forgetting curve manager with SM-2 spaced repetition,
    per-agent learning profiles, and emotional memory integration.

    This manager extends the base ForgettingCurveManager with:
    1. Per-agent LearningProfile management
    2. SM-2 algorithm integration for optimal review scheduling
    3. Emotional salience integration for decay modification
    4. Richer analytics and tracking

    Usage:
        manager = ForgettingCurveManager(engine)
        profile = manager.get_or_create_profile(agent_id, learning_style="fast")
        entry = await manager.schedule_review(node, agent_id)
        await manager.record_review_result(node_id, quality=4)
    """

    def __init__(
        self,
        engine: Optional["HAIMEngine"] = None,
        decay_engine: Optional[AdaptiveDecayEngine] = None,
        target_retention: float = TARGET_RETENTION,
        min_eig_to_consolidate: float = MIN_EIG_TO_CONSOLIDATE,
        persistence_path: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.decay = decay_engine or get_adaptive_decay_engine()
        self.target_retention = target_retention
        self.min_eig_to_consolidate = min_eig_to_consolidate
        self.persistence_path = persistence_path

        # Per-agent learning profiles
        self._profiles: Dict[str, LearningProfile] = {}

        # SM-2 state tracking per memory
        self._sm2_states: Dict[str, SM2State] = {}

        # Review schedule
        self._schedule: List[ReviewEntry] = []

        # Review history for analytics
        self._review_history: List[Dict[str, Any]] = []

        # Load persisted state if available
        if persistence_path:
            self._load_state()

    # ------------------------------------------------------------------
    #  Learning Profile Management
    # ------------------------------------------------------------------

    def get_or_create_profile(
        self,
        agent_id: str,
        learning_style: str = "balanced"
    ) -> LearningProfile:
        """Get existing profile or create new one."""
        if agent_id not in self._profiles:
            self._profiles[agent_id] = LearningProfile.for_agent(agent_id, learning_style)
            logger.info(f"Created new LearningProfile for agent '{agent_id}' with style '{learning_style}'")
        return self._profiles[agent_id]

    def get_profile(self, agent_id: str) -> Optional[LearningProfile]:
        """Get existing profile without creating."""
        return self._profiles.get(agent_id)

    def update_profile(
        self,
        agent_id: str,
        **updates
    ) -> Optional[LearningProfile]:
        """Update profile parameters."""
        profile = self._profiles.get(agent_id)
        if not profile:
            return None

        for key, value in updates.items():
            if hasattr(profile, key) and not key.startswith("_"):
                setattr(profile, key, value)

        profile.last_updated = datetime.now(timezone.utc)
        return profile

    def list_profiles(self) -> List[LearningProfile]:
        """List all agent profiles."""
        return list(self._profiles.values())

    # ------------------------------------------------------------------
    #  Emotional Integration
    # ------------------------------------------------------------------

    def get_emotional_salience(self, node: "MemoryNode") -> float:
        """
        Extract emotional salience from a memory node.

        Checks for emotional metadata from emotional_tag.py.
        Returns a value in [0.0, 1.0].
        """
        if not node or not node.metadata:
            return 0.0

        # Check for direct salience value
        salience = node.metadata.get("emotional_salience")
        if salience is not None:
            return float(max(0.0, min(1.0, salience)))

        # Calculate from valence and arousal
        valence = node.metadata.get("emotional_valence", 0.0)
        arousal = node.metadata.get("emotional_arousal", 0.0)

        # Salience = |valence| * arousal (Russell's circumplex model)
        calculated = abs(float(valence)) * float(arousal)
        return max(0.0, min(1.0, calculated))

    def is_emotional_memory(self, node: "MemoryNode") -> bool:
        """Check if a memory has significant emotional content."""
        salience = self.get_emotional_salience(node)
        return salience >= HIGH_SALIENCE_THRESHOLD

    def apply_emotional_decay_modifier(
        self,
        retention: float,
        node: "MemoryNode",
        profile: Optional[LearningProfile] = None
    ) -> float:
        """
        Apply emotional modifier to retention calculation.

        Emotional memories decay slower (higher retention).
        """
        salience = self.get_emotional_salience(node)

        if salience < 0.1:
            return retention  # No significant emotion

        # Get profile or use default
        if profile:
            modifier = profile.get_emotional_modifier(salience)
        else:
            # Default emotional boost
            modifier = 1.0 + (salience * EMOTIONAL_DECAY_REDUCTION)

        # Apply modifier to retention (inverse effect on decay)
        # Higher modifier = higher retention = slower decay
        modified = min(1.0, retention * modifier)
        return modified

    # ------------------------------------------------------------------
    #  SM-2 Integration
    # ------------------------------------------------------------------

    def get_sm2_state(self, memory_id: str) -> SM2State:
        """Get or create SM-2 state for a memory."""
        if memory_id not in self._sm2_states:
            self._sm2_states[memory_id] = SM2State(memory_id=memory_id)
        return self._sm2_states[memory_id]

    def update_sm2_state(
        self,
        memory_id: str,
        quality: int,
        agent_id: str,
        review_date: Optional[datetime] = None,
        is_emotional: bool = False
    ) -> SM2State:
        """
        Update SM-2 state after a review.

        Args:
            memory_id: ID of the memory being reviewed
            quality: Review quality (0-5)
            agent_id: ID of the agent performing the review
            review_date: When the review occurred
            is_emotional: Whether the memory has emotional significance

        Returns the updated SM2State.
        """
        profile = self.get_or_create_profile(agent_id)
        old_state = self.get_sm2_state(memory_id)
        new_state = old_state.calculate_next_review(quality, profile, review_date)

        self._sm2_states[memory_id] = new_state

        # Update profile stats
        profile.update_stats(quality, is_emotional)

        # Record in history
        self._review_history.append({
            "memory_id": memory_id,
            "agent_id": agent_id,
            "quality": quality,
            "timestamp": (review_date or datetime.now(timezone.utc)).isoformat(),
            "was_emotional": is_emotional,
            "new_interval": new_state.interval,
            "new_repetitions": new_state.repetitions,
        })

        # Trim history
        if len(self._review_history) > ANALYTICS_HISTORY_SIZE:
            self._review_history = self._review_history[-ANALYTICS_HISTORY_SIZE:]

        return new_state

    def calculate_sm2_retention(
        self,
        memory_id: str,
        node: Optional["MemoryNode"] = None,
        profile: Optional[LearningProfile] = None
    ) -> float:
        """
        Calculate retention based on SM-2 state and time.

        Uses the elapsed time since last review vs the scheduled interval
        to estimate current retention probability.
        """
        state = self.get_sm2_state(memory_id)

        if state.repetitions == 0 or state.last_review_date is None:
            # Never successfully reviewed - use standard decay
            if node:
                return self.decay.retention(node)
            return 0.5

        # Calculate time elapsed as ratio of scheduled interval
        now = datetime.now(timezone.utc)
        elapsed = (now - state.last_review_date).total_seconds() / 86400.0

        if state.interval <= 0:
            return 0.5

        # Retention based on SM-2 interval position
        # At interval boundary, retention should be ~target_retention
        ratio = elapsed / state.interval

        # Use exponential decay from target at interval boundary
        # R(t) = target * e^(-lambda * (t/I - 1)) for t > I
        if ratio <= 1.0:
            # Within scheduled interval - retention should be decent
            retention = self.target_retention + (1 - self.target_retention) * (1 - ratio)
        else:
            # Past scheduled interval - exponential decay
            excess_ratio = ratio - 1.0
            retention = self.target_retention * math.exp(-excess_ratio)

        # Apply emotional modifier if node provided
        if node:
            retention = self.apply_emotional_decay_modifier(retention, node, profile)

        return max(0.0, min(1.0, retention))

    # ------------------------------------------------------------------
    #  Review Scheduling
    # ------------------------------------------------------------------

    def next_review_days(
        self,
        node: "MemoryNode",
        profile: Optional[LearningProfile] = None,
        sm2_state: Optional[SM2State] = None
    ) -> float:
        """
        Days until next review using combined SM-2 and adaptive decay.

        Prefers SM-2 interval if available, falls back to Ebbinghaus.
        """
        memory_id = getattr(node, "id", "")
        state = sm2_state or self.get_sm2_state(memory_id)

        # Use SM-2 interval if we have review history
        if state.repetitions > 0 and state.next_review_date:
            now = datetime.now(timezone.utc)
            if state.next_review_date > now:
                return (state.next_review_date - now).total_seconds() / 86400.0
            return 0.0  # Overdue

        # Fallback to adaptive decay calculation
        profile = profile or self.get_or_create_profile("default")
        s_i = self.decay.stability(node)
        s_i *= profile.get_decay_modifier(node)

        # Apply emotional modifier
        salience = self.get_emotional_salience(node)
        if salience > 0.1:
            s_i *= profile.get_emotional_modifier(salience)

        target = max(1e-6, min(self.target_retention, 0.999))
        return -s_i * math.log(target)

    def schedule_reviews(
        self,
        nodes: "List[MemoryNode]",
        agent_id: str = "default"
    ) -> List[ReviewEntry]:
        """
        Scan nodes and build schedule of upcoming reviews.

        Enhanced to use SM-2 intervals and emotional modifiers.
        """
        now = datetime.now(timezone.utc)
        profile = self.get_or_create_profile(agent_id)
        new_entries: List[ReviewEntry] = []

        for node in nodes:
            memory_id = getattr(node, "id", "")
            if not memory_id:
                continue

            # Get SM-2 state
            sm2_state = self.get_sm2_state(memory_id)

            # Calculate retention
            retention = self.calculate_sm2_retention(memory_id, node, profile)

            # Get stability with modifiers
            s_i = self.decay.stability(node)
            s_i *= profile.get_decay_modifier(node)

            # Get emotional salience
            salience = self.get_emotional_salience(node)

            # Calculate next review time
            days_until = self.next_review_days(node, profile, sm2_state)
            due_at = now + timedelta(days=days_until)

            # Determine action
            action = self._determine_action(node, retention, salience)

            entry = ReviewEntry(
                memory_id=memory_id,
                agent_id=agent_id,
                due_at=due_at,
                current_retention=retention,
                stability=s_i,
                sm2_state=sm2_state,
                emotional_salience=salience,
                action=action,
            )
            new_entries.append(entry)

            # Update node's review_candidate flag
            self.decay.update_review_candidate(node)

        # Merge into schedule
        scheduled_ids = {e.memory_id for e in self._schedule}
        for entry in new_entries:
            # Replace existing entry for this memory
            self._schedule = [e for e in self._schedule if e.memory_id != entry.memory_id]
            self._schedule.append(entry)

        self._schedule.sort(key=lambda e: e.due_at)

        logger.info(
            f"ForgettingCurveManager: scheduled {len(new_entries)} reviews for agent '{agent_id}'. "
            f"Total scheduled: {len(self._schedule)}"
        )

        return new_entries

    def _determine_action(
        self,
        node: "MemoryNode",
        retention: float,
        salience: float
    ) -> str:
        """
        Determine what action to take for a memory.

        Enhanced to consider emotional salience.
        """
        # Emotional memories get boost instead of review
        if salience >= HIGH_SALIENCE_THRESHOLD and retention < self.target_retention:
            return ReviewAction.BOOST.value

        # Standard eviction logic
        if self.decay.should_evict(node):
            eig = getattr(node, "epistemic_value", 0.0)
            if eig >= self.min_eig_to_consolidate:
                return ReviewAction.CONSOLIDATE.value
            return ReviewAction.EVICT.value

        return ReviewAction.REVIEW.value

    # ------------------------------------------------------------------
    #  Schedule Queries
    # ------------------------------------------------------------------

    def get_schedule(self, agent_id: Optional[str] = None) -> List[ReviewEntry]:
        """Return review schedule, optionally filtered by agent."""
        schedule = sorted(self._schedule, key=lambda e: e.due_at)
        if agent_id:
            return [e for e in schedule if e.agent_id == agent_id]
        return schedule

    def get_due_reviews(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ReviewEntry]:
        """Return entries that are due now, optionally filtered by agent."""
        now = datetime.now(timezone.utc)
        due = [e for e in self._schedule if e.due_at <= now]

        if agent_id:
            due = [e for e in due if e.agent_id == agent_id]

        return sorted(due, key=lambda e: e.due_at)[:limit]

    def get_actions_by_type(
        self,
        action: str,
        agent_id: Optional[str] = None
    ) -> List[ReviewEntry]:
        """Filter schedule by action type."""
        filtered = [e for e in self._schedule if e.action == action]
        if agent_id:
            filtered = [e for e in filtered if e.agent_id == agent_id]
        return filtered

    def remove_entry(self, memory_id: str) -> None:
        """Remove a memory from the review schedule."""
        self._schedule = [e for e in self._schedule if e.memory_id != memory_id]

    # ------------------------------------------------------------------
    #  Engine Integration
    # ------------------------------------------------------------------

    async def run_once(self, agent_id: str = "default") -> Dict:
        """
        Run a full scan over HOT + WARM nodes and update review schedule.

        Returns stats dict with counts per action.
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

        entries = self.schedule_reviews(nodes, agent_id)

        # Count actions
        from collections import Counter
        action_counts = dict(Counter(e.action for e in entries))

        logger.info(
            f"ForgettingCurveManager scan for agent '{agent_id}': {action_counts}"
        )

        # Persist state if path configured
        if self.persistence_path:
            self._save_state()

        return {
            "agent_id": agent_id,
            "nodes_scanned": len(nodes),
            "entries_scheduled": len(entries),
            "action_counts": action_counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def record_review_result(
        self,
        memory_id: str,
        quality: int,
        agent_id: str = "default"
    ) -> Optional[SM2State]:
        """
        Record the result of a memory review.

        Args:
            memory_id: ID of the reviewed memory
            quality: Review quality (0-5)
            agent_id: ID of the agent performing review

        Returns the updated SM2State.
        """
        state = self.update_sm2_state(memory_id, quality, agent_id)

        # Update the schedule entry
        for entry in self._schedule:
            if entry.memory_id == memory_id and entry.agent_id == agent_id:
                entry.due_at = state.next_review_date or datetime.now(timezone.utc)
                entry.sm2_state = state
                break

        # Persist state
        if self.persistence_path:
            self._save_state()

        logger.info(
            f"Recorded review for memory {memory_id[:8]}: quality={quality}, "
            f"next interval={state.interval:.1f}d"
        )

        return state

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Save manager state to disk."""
        if not self.persistence_path:
            return

        try:
            path = Path(self.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "profiles": {k: v.to_dict() for k, v in self._profiles.items()},
                "sm2_states": {k: v.to_dict() for k, v in self._sm2_states.items()},
                "schedule": [e.to_dict() for e in self._schedule],
                "review_history": self._review_history[-1000:],  # Limit history
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            logger.debug(f"Saved ForgettingCurveManager state to {path}")

        except Exception as e:
            logger.warning(f"Failed to save ForgettingCurveManager state: {e}")

    def _load_state(self) -> None:
        """Load manager state from disk."""
        if not self.persistence_path:
            return

        try:
            path = Path(self.persistence_path)
            if not path.exists():
                logger.debug(f"No persisted state found at {path}")
                return

            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Restore profiles
            for agent_id, profile_data in state.get("profiles", {}).items():
                self._profiles[agent_id] = LearningProfile.from_dict(profile_data)

            # Restore SM-2 states
            for memory_id, sm2_data in state.get("sm2_states", {}).items():
                self._sm2_states[memory_id] = SM2State.from_dict(sm2_data)

            # Restore schedule
            for entry_data in state.get("schedule", []):
                entry = ReviewEntry(
                    memory_id=entry_data["memory_id"],
                    agent_id=entry_data.get("agent_id", "default"),
                    due_at=datetime.fromisoformat(entry_data["due_at"]),
                    current_retention=entry_data["current_retention"],
                    stability=entry_data["stability"],
                    sm2_state=SM2State.from_dict(entry_data["sm2_state"]) if entry_data.get("sm2_state") else None,
                    emotional_salience=entry_data.get("emotional_salience", 0.0),
                    action=entry_data.get("action", "review"),
                )
                self._schedule.append(entry)

            # Restore history
            self._review_history = state.get("review_history", [])

            logger.info(
                f"Loaded ForgettingCurveManager state: {len(self._profiles)} profiles, "
                f"{len(self._sm2_states)} SM-2 states, {len(self._schedule)} scheduled reviews"
            )

        except Exception as e:
            logger.warning(f"Failed to load ForgettingCurveManager state: {e}")


# ------------------------------------------------------------------ #
#  Forgetting Analytics Dashboard                                     #
# ------------------------------------------------------------------ #

@dataclass
class RetentionCurve:
    """
    A single retention curve data point for visualization.
    """
    time_days: float
    retention: float
    memory_count: int
    emotional_count: int
    avg_quality: float


@dataclass
class AgentAnalytics:
    """
    Analytics data for a single agent.
    """
    agent_id: str
    total_memories: int
    avg_retention: float
    avg_stability: float
    emotional_memories: int
    due_for_review: int
    overdue_count: int
    sm2_stats: Dict[str, Any]
    learning_style: str


class ForgettingAnalytics:
    """
    Analytics and visualization dashboard for forgetting curve data.

    Provides:
    1. Retention curve visualization data
    2. Per-agent analytics summaries
    3. SM-2 performance metrics
    4. Emotional memory distribution
    5. Review schedule statistics
    """

    def __init__(
        self,
        manager: ForgettingCurveManager,
        engine: Optional["HAIMEngine"] = None
    ):
        self.manager = manager
        self.engine = engine

    def get_dashboard_data(
        self,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for visualization.

        Returns a dictionary with all analytics data ready for
        rendering in a dashboard UI.
        """
        schedule = self.manager.get_schedule(agent_id)

        return {
            "summary": self._get_summary_stats(schedule),
            "agents": self._get_agent_analytics(schedule),
            "retention_curve": self._calculate_retention_curve(agent_id),
            "sm2_performance": self._get_sm2_performance(agent_id),
            "emotional_distribution": self._get_emotional_distribution(schedule),
            "review_schedule": self._get_schedule_breakdown(schedule),
            "recommendations": self._generate_recommendations(schedule),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _get_summary_stats(self, schedule: List[ReviewEntry]) -> Dict[str, Any]:
        """Calculate overall summary statistics."""
        if not schedule:
            return {
                "total_scheduled": 0,
                "avg_retention": 0.0,
                "avg_stability": 0.0,
                "emotional_ratio": 0.0,
            }

        now = datetime.now(timezone.utc)
        emotional_count = sum(1 for e in schedule if e.emotional_salience >= HIGH_SALIENCE_THRESHOLD)

        return {
            "total_scheduled": len(schedule),
            "avg_retention": round(statistics.mean(e.current_retention for e in schedule), 4),
            "avg_stability": round(statistics.mean(e.stability for e in schedule), 4),
            "emotional_ratio": round(emotional_count / len(schedule), 4),
            "due_now": sum(1 for e in schedule if e.due_at <= now),
            "overdue": sum(1 for e in schedule if e.due_at < now),
        }

    def _get_agent_analytics(
        self,
        schedule: List[ReviewEntry]
    ) -> Dict[str, AgentAnalytics]:
        """Get per-agent analytics."""
        agent_entries: Dict[str, List[ReviewEntry]] = defaultdict(list)
        for entry in schedule:
            agent_entries[entry.agent_id].append(entry)

        analytics = {}
        now = datetime.now(timezone.utc)

        for agent_id, entries in agent_entries.items():
            profile = self.manager.get_profile(agent_id)
            sm2_states = [
                e.sm2_state for e in entries
                if e.sm2_state and e.sm2_state.repetitions > 0
            ]

            analytics[agent_id] = AgentAnalytics(
                agent_id=agent_id,
                total_memories=len(entries),
                avg_retention=round(statistics.mean(e.current_retention for e in entries), 4) if entries else 0.0,
                avg_stability=round(statistics.mean(e.stability for e in entries), 4) if entries else 0.0,
                emotional_memories=sum(1 for e in entries if e.emotional_salience >= HIGH_SALIENCE_THRESHOLD),
                due_for_review=sum(1 for e in entries if e.due_at <= now),
                overdue_count=sum(1 for e in entries if e.due_at < now),
                sm2_stats=self._calculate_sm2_stats(sm2_states),
                learning_style=profile.learning_style if profile else "unknown",
            )

        return analytics

    def _calculate_sm2_stats(self, states: List[SM2State]) -> Dict[str, Any]:
        """Calculate SM-2 specific statistics."""
        if not states:
            return {
                "active_items": 0,
                "avg_repetitions": 0.0,
                "avg_interval": 0.0,
                "avg_easiness": 0.0,
            }

        return {
            "active_items": len(states),
            "avg_repetitions": round(statistics.mean(s.repetitions for s in states), 2),
            "avg_interval": round(statistics.mean(s.interval for s in states), 2),
            "avg_easiness": round(statistics.mean(s.easiness_factor for s in states), 3),
            "mature_items": sum(1 for s in states if s.repetitions >= 3),
        }

    def _calculate_retention_curve(
        self,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """
        Calculate retention curve data points.

        Returns time-series data showing how retention decays over time.
        """
        schedule = self.manager.get_schedule(agent_id)

        if not schedule:
            return []

        # Group by time buckets (days)
        time_buckets: Dict[float, List[float]] = defaultdict(list)

        now = datetime.now(timezone.utc)

        for entry in schedule:
            # Calculate age in days
            if entry.sm2_state and entry.sm2_state.last_review_date:
                age = (now - entry.sm2_state.last_review_date).total_seconds() / 86400.0
            else:
                age = 0.0

            bucket = round(age, 1)
            time_buckets[bucket].append(entry.current_retention)

        # Calculate curve points
        curve_points = []
        for bucket_days in sorted(time_buckets.keys()):
            retentions = time_buckets[bucket_days]
            curve_points.append({
                "time_days": bucket_days,
                "retention": round(statistics.mean(retentions), 4),
                "count": len(retentions),
            })

        return curve_points

    def _get_sm2_performance(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get SM-2 algorithm performance metrics."""
        history = self.manager._review_history

        if agent_id:
            history = [h for h in history if h.get("agent_id") == agent_id]

        if not history:
            return {
                "total_reviews": 0,
                "avg_quality": 0.0,
                "success_rate": 0.0,
                "avg_interval_growth": 0.0,
            }

        qualities = [h.get("quality", 0) for h in history]
        successful = [q for q in qualities if q >= 3]

        # Calculate interval growth for memories with multiple reviews
        interval_growth = []
        memory_reviews: Dict[str, List[Dict]] = defaultdict(list)
        for h in history:
            memory_reviews[h["memory_id"]].append(h)

        for mem_history in memory_reviews.values():
            if len(mem_history) > 1:
                intervals = [h.get("new_interval", 0) for h in mem_history]
                if len(intervals) >= 2:
                    growth = (intervals[-1] - intervals[0]) / max(intervals[0], 0.1)
                    interval_growth.append(growth)

        return {
            "total_reviews": len(history),
            "avg_quality": round(statistics.mean(qualities), 3) if qualities else 0.0,
            "success_rate": round(len(successful) / len(qualities), 4) if qualities else 0.0,
            "avg_interval_growth": round(statistics.mean(interval_growth), 2) if interval_growth else 0.0,
        }

    def _get_emotional_distribution(self, schedule: List[ReviewEntry]) -> Dict[str, Any]:
        """Analyze emotional memory distribution."""
        if not schedule:
            return {
                "total": 0,
                "emotional": 0,
                "neutral": 0,
                "avg_salience": 0.0,
                "by_quartile": {},
            }

        saliences = [e.emotional_salience for e in schedule]
        emotional_count = sum(1 for s in saliences if s >= HIGH_SALIENCE_THRESHOLD)

        # Quartile distribution
        quartiles = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
        for s in saliences:
            if s < 0.25:
                quartiles["low"] += 1
            elif s < 0.5:
                quartiles["medium"] += 1
            elif s < 0.75:
                quartiles["high"] += 1
            else:
                quartiles["very_high"] += 1

        return {
            "total": len(schedule),
            "emotional": emotional_count,
            "neutral": len(schedule) - emotional_count,
            "avg_salience": round(statistics.mean(saliences), 4) if saliences else 0.0,
            "by_quartile": quartiles,
        }

    def _get_schedule_breakdown(self, schedule: List[ReviewEntry]) -> Dict[str, Any]:
        """Get review schedule breakdown by action and urgency."""
        now = datetime.now(timezone.utc)

        by_action = defaultdict(int)
        urgency = {"immediate": 0, "today": 0, "week": 0, "month": 0, "later": 0}

        for entry in schedule:
            by_action[entry.action] += 1

            delta = (entry.due_at - now).total_seconds() / 86400.0
            if delta <= 0:
                urgency["immediate"] += 1
            elif delta <= 1:
                urgency["today"] += 1
            elif delta <= 7:
                urgency["week"] += 1
            elif delta <= 30:
                urgency["month"] += 1
            else:
                urgency["later"] += 1

        return {
            "by_action": dict(by_action),
            "by_urgency": urgency,
        }

    def _generate_recommendations(self, schedule: List[ReviewEntry]) -> List[str]:
        """Generate actionable recommendations based on analytics."""
        recommendations = []
        now = datetime.now(timezone.utc)

        # Check for overdue items
        overdue = sum(1 for e in schedule if e.due_at < now)
        if overdue > 10:
            recommendations.append(
                f"High priority: {overdue} memories are overdue for review. "
                "Consider running a review session."
            )

        # Check for low retention items
        low_retention = sum(1 for e in schedule if e.current_retention < 0.3)
        if low_retention > 5:
            recommendations.append(
                f"Warning: {low_retention} memories have retention below 30%. "
                "These may need consolidation or re-learning."
            )

        # Check for emotional memory opportunities
        emotional_review = sum(
            1 for e in schedule
            if e.emotional_salience >= HIGH_SALIENCE_THRESHOLD and e.due_at <= now
        )
        if emotional_review > 3:
            recommendations.append(
                f"Opportunity: {emotional_review} emotionally significant memories "
                "are due for review. These may be particularly impactful."
            )

        # Check SM-2 health
        sm2_active = sum(1 for e in schedule if e.sm2_state and e.sm2_state.repetitions > 0)
        if sm2_active < len(schedule) * 0.5:
            recommendations.append(
                "SM-2 algorithm is underutilized. More consistent review sessions "
                "will improve long-term retention predictions."
            )

        return recommendations

    def export_retention_curve_csv(self, agent_id: Optional[str] = None) -> str:
        """
        Export retention curve as CSV string.

        Useful for external visualization tools.
        """
        curve = self._calculate_retention_curve(agent_id)

        lines = ["time_days,retention,count"]
        for point in curve:
            lines.append(
                f"{point['time_days']},{point['retention']},{point['count']}"
            )

        return "\n".join(lines)

    def get_agent_comparison(self) -> List[Dict[str, Any]]:
        """
        Compare learning performance across all agents.

        Returns a list of agent comparison metrics.
        """
        schedule = self.manager.get_schedule()
        agent_analytics = self._get_agent_analytics(schedule)

        comparison = []
        for agent_id, analytics in agent_analytics.items():
            profile = self.manager.get_profile(agent_id)

            comparison.append({
                "agent_id": agent_id,
                "learning_style": analytics.learning_style,
                "total_memories": analytics.total_memories,
                "avg_retention": analytics.avg_retention,
                "emotional_memories": analytics.emotional_memories,
                "due_for_review": analytics.due_for_review,
                "sm2_active_items": analytics.sm2_stats.get("active_items", 0),
                "sm2_avg_interval": analytics.sm2_stats.get("avg_interval", 0.0),
                "easiness_factor": profile.easiness_factor if profile else SM2_DEFAULT_EASINESS,
            })

        # Sort by average retention
        comparison.sort(key=lambda x: x["avg_retention"], reverse=True)

        return comparison

    def get_learning_progress_chart(
        self,
        agent_id: str,
        memory_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get learning progress data for a chart.

        If memory_id is provided, returns data for that specific memory.
        Otherwise, returns aggregate progress for the agent.
        """
        history = self.manager._review_history

        # Filter by agent
        if agent_id:
            history = [h for h in history if h.get("agent_id") == agent_id]

        # Filter by memory if specified
        if memory_id:
            history = [h for h in history if h.get("memory_id") == memory_id]

        # Build progress chart
        chart_data = []
        for i, entry in enumerate(history):
            chart_data.append({
                "review_number": i + 1,
                "quality": entry.get("quality", 0),
                "interval": entry.get("new_interval", 0.0),
                "repetitions": entry.get("new_repetitions", 0),
                "timestamp": entry.get("timestamp", ""),
            })

        return chart_data


# ------------------------------------------------------------------ #
#  Convenience Functions                                               #
# ------------------------------------------------------------------ #

def create_forgetting_manager(
    engine: Optional["HAIMEngine"] = None,
    persistence_path: Optional[str] = None,
    **kwargs
) -> ForgettingCurveManager:
    """
    Factory function to create a configured ForgettingCurveManager.

    Args:
        engine: HAIMEngine instance
        persistence_path: Optional path for state persistence
        **kwargs: Additional arguments for ForgettingCurveManager

    Returns:
        Configured ForgettingCurveManager instance
    """
    return ForgettingCurveManager(
        engine=engine,
        persistence_path=persistence_path,
        **kwargs
    )


def create_learning_profile(
    agent_id: str,
    learning_style: str = "balanced",
    **custom_params
) -> LearningProfile:
    """
    Factory function to create a LearningProfile.

    Args:
        agent_id: Unique identifier for the agent
        learning_style: Predefined style or "custom"
        **custom_params: Custom parameters if learning_style="custom"

    Returns:
        Configured LearningProfile
    """
    if learning_style != "custom":
        return LearningProfile.for_agent(agent_id, learning_style)

    # Custom profile
    return LearningProfile(agent_id=agent_id, **custom_params)


def quality_from_confidence(confidence: float) -> int:
    """
    Convert a confidence score to SM-2 quality rating.

    Args:
        confidence: Confidence score in [0.0, 1.0]

    Returns:
        Quality rating in [0, 5]
    """
    if confidence >= 0.95:
        return 5  # Perfect recall
    elif confidence >= 0.8:
        return 4  # Correct with hesitation
    elif confidence >= 0.6:
        return 3  # Correct but difficult
    elif confidence >= 0.4:
        return 2  # Incorrect but easy recall
    elif confidence >= 0.2:
        return 1  # Incorrect but recognized
    else:
        return 0  # Complete blackout
