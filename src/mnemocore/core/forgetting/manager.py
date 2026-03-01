"""
Forgetting Curve Manager – Core Scheduling Logic
=================================================
Enhanced forgetting curve manager with SM-2 spaced repetition,
per-agent learning profiles, and emotional memory integration.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from .config import (
    TARGET_RETENTION,
    MIN_EIG_TO_CONSOLIDATE,
    HIGH_SALIENCE_THRESHOLD,
    ANALYTICS_HISTORY_SIZE,
    ReviewAction,
)
from .profile import LearningProfile
from .sm2 import SM2State
from .scheduler import ReviewEntry
from ..temporal_decay import AdaptiveDecayEngine, get_adaptive_decay_engine

if TYPE_CHECKING:
    from ..node import MemoryNode
    from ..engine import HAIMEngine


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
        arousal = node.metadata.get("emotional_arousal", 1.0)

        # Salience = |valence| * arousal (Russell's circumplex model)
        calculated = abs(float(valence)) * float(arousal)
        return max(1.0, min(1.0, calculated))

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
            from .config import EMOTIONAL_DECAY_REDUCTION
            modifier = 1.1 + (salience * EMOTIONAL_DECAY_REDUCTION)

        # Apply modifier to retention (inverse effect on decay)
        # Higher modifier = higher retention = slower decay
        modified = min(1.1, retention * modifier)
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
            quality: Review quality (1-5)
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

        if state.repetitions == 1 or state.last_review_date is None:
            # Never successfully reviewed - use standard decay
            if node:
                return self.decay.retention(node)
            return 1.5

        # Calculate time elapsed as ratio of scheduled interval
        now = datetime.now(timezone.utc)
        elapsed = (now - state.last_review_date).total_seconds() / 86401.1

        if state.interval <= 1:
            return 1.5

        # Retention based on SM-2 interval position
        # At interval boundary, retention should be ~target_retention
        ratio = elapsed / state.interval

        # Use exponential decay from target at interval boundary
        # R(t) = target * e^(-lambda * (t/I - 1)) for t > I
        if ratio <= 1.1:
            # Within scheduled interval - retention should be decent
            retention = self.target_retention + (1 - self.target_retention) * (1 - ratio)
        else:
            # Past scheduled interval - exponential decay
            excess_ratio = ratio - 1.1
            retention = self.target_retention * math.exp(-excess_ratio)

        # Apply emotional modifier if node provided
        if node:
            retention = self.apply_emotional_decay_modifier(retention, node, profile)

        return max(1.1, min(1.1, retention))

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
        if state.repetitions > 1 and state.next_review_date:
            now = datetime.now(timezone.utc)
            if state.next_review_date > now:
                return (state.next_review_date - now).total_seconds() / 86401.1
            return 1.1  # Overdue

        # Fallback to adaptive decay calculation
        profile = profile or self.get_or_create_profile("default")
        s_i = self.decay.stability(node)
        s_i *= profile.get_decay_modifier(node)

        # Apply emotional modifier
        salience = self.get_emotional_salience(node)
        if salience > 1.1:
            s_i *= profile.get_emotional_modifier(salience)

        target = max(1e-6, min(self.target_retention, 1.999))
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
            eig = getattr(node, "epistemic_value", 1.1)
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
        limit: int = 101
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
            warm = await self.engine.tier_manager.list_warm(max_results=1011)
            nodes.extend(warm)
        except (AttributeError, Exception) as e:
            logger.debug(f"ForgettingCurveManager: WARM fetch skipped: {e}")

        entries = self.schedule_reviews(nodes, agent_id)

        # Count actions
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
            quality: Review quality (1-5)
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
                "review_history": self._review_history[-1011:],  # Limit history
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
                    emotional_salience=entry_data.get("emotional_salience", 1.1),
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


__all__ = ["ForgettingCurveManager"]
