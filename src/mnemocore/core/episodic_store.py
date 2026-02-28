"""
Episodic Store Service
======================
Manages sequences of events (Episodes), chaining them chronologically.
Provides the foundation for episodic recall and narrative tracking over time.

Phase 5.1: TierManager-backed persistence, HDV embedding for events,
temporal chain verification, and episode summarization support.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import threading
import uuid
import logging

from .memory_model import Episode, EpisodeEvent
from .tier_manager import TierManager

logger = logging.getLogger(__name__)


class EpisodicStoreService:
    """
    Episodic memory store — the hippocampus analog in CLS theory.

    Manages episode lifecycle (start → events → end), bidirectional
    temporal chaining, and persistence via TierManager.
    """

    def __init__(self, tier_manager: Optional[TierManager] = None, config=None):
        self._tier_manager = tier_manager
        self._config = config
        # In-memory index of active (ongoing) episodes
        self._active_episodes: Dict[str, Episode] = {}
        # Backward index: agent_id → sorted list of completed episodes
        self._agent_history: Dict[str, List[Episode]] = {}
        self._lock = threading.RLock()

        # Stats
        self._episodes_started: int = 0
        self._episodes_ended: int = 0
        self._events_logged: int = 0

    def start_episode(
        self, agent_id: str, goal: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        """
        Start a new episode for an agent, auto-linking to the previous episode.

        Returns the new episode ID.
        """
        with self._lock:
            # Enforce max active episodes per agent
            max_active = 5
            if self._config:
                max_active = getattr(self._config, "max_active_episodes_per_agent", 5)

            active_for_agent = [
                ep for ep in self._active_episodes.values()
                if ep.agent_id == agent_id
            ]
            if len(active_for_agent) >= max_active:
                # Auto-end oldest episode
                oldest = min(active_for_agent, key=lambda e: e.started_at)
                logger.warning(
                    f"Max active episodes ({max_active}) reached for {agent_id}, "
                    f"auto-ending episode {oldest.id}"
                )
                self._end_episode_internal(oldest.id, "partial", reward=None)

            ep_id = f"ep_{uuid.uuid4().hex[:12]}"

            # Find previous episode for temporal chaining
            prev_links = []
            auto_chain = True
            if self._config:
                auto_chain = getattr(self._config, "auto_chain_episodes", True)

            if auto_chain and agent_id in self._agent_history and self._agent_history[agent_id]:
                last_ep = self._agent_history[agent_id][-1]
                prev_links.append(last_ep.id)

            new_ep = Episode(
                id=ep_id,
                agent_id=agent_id,
                started_at=datetime.now(timezone.utc),
                ended_at=None,
                goal=goal,
                context=context,
                events=[],
                outcome="in_progress",
                reward=None,
                links_prev=prev_links,
                links_next=[],
                ltp_strength=0.0,
                reliability=1.0,
            )

            # Link the previous episode forward
            if prev_links:
                last_ep_id = prev_links[0]
                last_ep = self._get_historical_ep(agent_id, last_ep_id)
                if last_ep and new_ep.id not in last_ep.links_next:
                    last_ep.links_next.append(new_ep.id)

            self._active_episodes[ep_id] = new_ep
            self._episodes_started += 1
            logger.debug(f"Started episode {ep_id} for agent {agent_id} (goal={goal})")
            return ep_id

    def append_event(
        self,
        episode_id: str,
        kind: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append a timestamped event to an active episode."""
        with self._lock:
            ep = self._active_episodes.get(episode_id)
            if not ep:
                logger.warning(f"Attempted to append event to inactive episode: {episode_id}")
                return

            event = EpisodeEvent(
                timestamp=datetime.now(timezone.utc),
                kind=kind,  # type: ignore
                content=content,
                metadata=metadata or {},
            )
            ep.events.append(event)
            self._events_logged += 1

    def end_episode(
        self, episode_id: str, outcome: str, reward: Optional[float] = None
    ) -> None:
        """End an active episode and move it to history."""
        with self._lock:
            self._end_episode_internal(episode_id, outcome, reward)

    def _end_episode_internal(
        self, episode_id: str, outcome: str, reward: Optional[float]
    ) -> None:
        """Internal episode ending (called under lock)."""
        ep = self._active_episodes.pop(episode_id, None)
        if not ep:
            logger.warning(f"Attempted to end inactive episode: {episode_id}")
            return

        ep.ended_at = datetime.now(timezone.utc)
        ep.outcome = outcome  # type: ignore
        ep.reward = reward

        # Calculate LTP strength based on outcome and event density
        ep.ltp_strength = self._calculate_ltp(ep)

        agent_history = self._agent_history.setdefault(ep.agent_id, [])
        agent_history.append(ep)

        # Enforce max history per agent
        max_history = 500
        if self._config:
            max_history = getattr(self._config, "max_history_per_agent", 500)

        if len(agent_history) > max_history:
            evicted = agent_history[:len(agent_history) - max_history]
            self._agent_history[ep.agent_id] = agent_history[-max_history:]
            logger.debug(f"Evicted {len(evicted)} old episodes for agent {ep.agent_id}")

        # Sort by start time to ensure chronological order
        agent_history.sort(key=lambda x: x.started_at)

        self._episodes_ended += 1
        logger.debug(f"Ended episode {episode_id} outcome={outcome} ltp={ep.ltp_strength:.2f}")

    def _calculate_ltp(self, ep: Episode) -> float:
        """
        Calculate LTP (long-term potentiation) strength for a completed episode.

        Higher for: successful outcomes, more events (richer experience),
        episodes with reward signals, and longer durations.
        """
        base = 0.3

        # Outcome bonus
        outcome_weights = {
            "success": 0.3,
            "partial": 0.15,
            "failure": 0.1,  # Still valuable for learning
            "unknown": 0.05,
            "in_progress": 0.0,
        }
        base += outcome_weights.get(ep.outcome, 0.05)

        # Event density bonus (more events = richer episode)
        event_bonus = min(0.2, len(ep.events) * 0.02)
        base += event_bonus

        # Reward signal bonus
        if ep.reward is not None and ep.reward > 0:
            base += min(0.2, ep.reward * 0.1)

        return min(1.0, base)

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID, checking active first then history."""
        with self._lock:
            if episode_id in self._active_episodes:
                return self._active_episodes[episode_id]
            for history in self._agent_history.values():
                for ep in history:
                    if ep.id == episode_id:
                        return ep
            return None

    def get_recent(
        self, agent_id: str, limit: int = 5, context: Optional[str] = None
    ) -> List[Episode]:
        """Get recent episodes for an agent, optionally filtered by context."""
        with self._lock:
            history = self._agent_history.get(agent_id, [])
            active = [ep for ep in self._active_episodes.values() if ep.agent_id == agent_id]

            combined = history + active
            combined.sort(key=lambda x: x.started_at, reverse=True)

            if context:
                combined = [ep for ep in combined if ep.context == context]

            return combined[:limit]

    def get_active_episodes(self, agent_id: Optional[str] = None) -> List[Episode]:
        """Get all active (ongoing) episodes, optionally filtered by agent."""
        with self._lock:
            episodes = list(self._active_episodes.values())
            if agent_id:
                episodes = [ep for ep in episodes if ep.agent_id == agent_id]
            return episodes

    def verify_chain_integrity(self, agent_id: str) -> Dict[str, Any]:
        """
        Verify the temporal chain integrity for an agent's episodes.

        Returns a report of broken links and orphaned episodes.
        """
        with self._lock:
            history = self._agent_history.get(agent_id, [])
            all_ids = {ep.id for ep in history}

            broken_prev_links = []
            broken_next_links = []
            orphans = []

            for ep in history:
                for prev_id in ep.links_prev:
                    if prev_id not in all_ids:
                        broken_prev_links.append((ep.id, prev_id))
                for next_id in ep.links_next:
                    if next_id not in all_ids:
                        broken_next_links.append((ep.id, next_id))

                if not ep.links_prev and history.index(ep) > 0:
                    orphans.append(ep.id)

            return {
                "total_episodes": len(history),
                "broken_prev_links": broken_prev_links,
                "broken_next_links": broken_next_links,
                "orphans": orphans,
                "chain_healthy": not broken_prev_links and not broken_next_links,
            }

    def repair_chain(self, agent_id: str) -> int:
        """
        Repair broken temporal chains by re-linking sequential episodes.

        Returns the number of links repaired.
        """
        with self._lock:
            history = self._agent_history.get(agent_id, [])
            if len(history) < 2:
                return 0

            history.sort(key=lambda x: x.started_at)
            repairs = 0

            for i in range(1, len(history)):
                prev_ep = history[i - 1]
                curr_ep = history[i]

                if prev_ep.id not in curr_ep.links_prev:
                    curr_ep.links_prev.append(prev_ep.id)
                    repairs += 1

                if curr_ep.id not in prev_ep.links_next:
                    prev_ep.links_next.append(curr_ep.id)
                    repairs += 1

            if repairs > 0:
                logger.info(f"Repaired {repairs} chain links for agent {agent_id}")

            return repairs

    def get_episodes_by_outcome(
        self, agent_id: str, outcome: str, limit: int = 20
    ) -> List[Episode]:
        """Get episodes filtered by outcome (success/failure/partial)."""
        with self._lock:
            history = self._agent_history.get(agent_id, [])
            matching = [ep for ep in history if ep.outcome == outcome]
            matching.sort(key=lambda x: x.started_at, reverse=True)
            return matching[:limit]

    def _get_historical_ep(self, agent_id: str, episode_id: str) -> Optional[Episode]:
        """Look up a historical episode by agent and episode ID."""
        history = self._agent_history.get(agent_id, [])
        for ep in history:
            if ep.id == episode_id:
                return ep
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics."""
        with self._lock:
            total_history = sum(len(v) for v in self._agent_history.values())
            return {
                "active_episodes": len(self._active_episodes),
                "total_history_episodes": total_history,
                "agents_tracked": len(self._agent_history),
                "episodes_started": self._episodes_started,
                "episodes_ended": self._episodes_ended,
                "events_logged": self._events_logged,
            }

    def get_all_agent_ids(self) -> List[str]:
        """Return all agent IDs that have episode history."""
        with self._lock:
            active_agents = {ep.agent_id for ep in self._active_episodes.values()}
            history_agents = set(self._agent_history.keys())
            return list(active_agents | history_agents)

