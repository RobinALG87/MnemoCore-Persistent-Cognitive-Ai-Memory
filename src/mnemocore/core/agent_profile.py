"""
Agent Profiles
==============
Persistent state encompassing quirks, long-term alignment details, and tooling preferences per individual actor.
Allows multiple independent agents to interact cleanly without memory namespace collisions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentProfile:
    id: str
    name: str
    description: str
    created_at: datetime
    last_active: datetime
    # Hard bounds over behavior: e.g. "Do not delete files without explicit prompt"
    core_directives: List[str] = field(default_factory=list)
    # Flexible learned preferences
    preferences: dict[str, Any] = field(default_factory=dict)
    # Agent-specific metrics
    reliability_score: float = 1.0


class AgentProfileService:
    """
    Service for managing agent identity profiles.

    Provides thread-safe access to agent profiles including preferences,
    core directives, and reliability tracking. Profiles are currently
    stored in-memory but should be backed by persistent storage (SQLite/Redis)
    in production deployments.

    Attributes:
        _profiles: Dictionary mapping agent IDs to their profiles.
        _lock: Reentrant lock for thread-safe access.
    """

    def __init__(self):
        """
        Initialize the AgentProfileService.

        Creates an empty profile store with thread-safe access.
        Note: Profiles are not persisted between restarts in the current
        implementation. Consider backing with SQLite or Redis for production.
        """
        # Local state dict, should back out to SQLite or Redis
        self._profiles: Dict[str, AgentProfile] = {}
        self._lock = threading.RLock()

    def get_or_create_profile(self, agent_id: str, name: str = "Unknown Agent") -> AgentProfile:
        """Retrieve the identity profile for an agent, constructing it if completely uninitialized."""
        with self._lock:
            if agent_id not in self._profiles:
                self._profiles[agent_id] = AgentProfile(
                    id=agent_id,
                    name=name,
                    description=f"Auto-generated profile for {agent_id}",
                    created_at=datetime.now(timezone.utc),
                    last_active=datetime.now(timezone.utc)
                )
            
            profile = self._profiles[agent_id]
            profile.last_active = datetime.now(timezone.utc)
            return profile

    def update_preferences(self, agent_id: str, new_preferences: dict[str, Any]) -> None:
        """Merge learned trait or task preferences into an agent's persistent identity."""
        with self._lock:
            profile = self.get_or_create_profile(agent_id)
            profile.preferences.update(new_preferences)
            logger.debug(f"Updated preferences for agent {agent_id}.")

    def adjust_reliability(self, agent_id: str, points: float) -> None:
        """Alter universal trust rating of the agent based on episodic action evaluations."""
        with self._lock:
            profile = self.get_or_create_profile(agent_id)
            profile.reliability_score = max(0.0, min(1.0, profile.reliability_score + points))

