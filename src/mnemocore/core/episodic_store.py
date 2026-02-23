"""
Episodic Store Service
======================
Manages sequences of events (Episodes), chaining them chronologically.
Provides the foundation for episodic recall and narrative tracking over time.
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
    def __init__(self, tier_manager: Optional[TierManager] = None):
        self._tier_manager = tier_manager
        # In-memory index of active episodes; eventually backed by SQLite/Qdrant
        self._active_episodes: Dict[str, Episode] = {}
        # Simple backward index map from agent to sorted list of historical episodes
        self._agent_history: Dict[str, List[Episode]] = {}
        self._lock = threading.RLock()

    def start_episode(
        self, agent_id: str, goal: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        with self._lock:
            ep_id = f"ep_{uuid.uuid4().hex[:12]}"
            
            # Find previous absolute episode for this agent to populate links_prev
            prev_links = []
            if agent_id in self._agent_history and self._agent_history[agent_id]:
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
            return ep_id

    def append_event(
        self,
        episode_id: str,
        kind: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            ep = self._active_episodes.get(episode_id)
            if not ep:
                logger.warning(f"Attempted to append event to inactive or not found episode: {episode_id}")
                return

            event = EpisodeEvent(
                timestamp=datetime.now(timezone.utc),
                kind=kind, # type: ignore
                content=content,
                metadata=metadata or {},
            )
            ep.events.append(event)

    def end_episode(
        self, episode_id: str, outcome: str, reward: Optional[float] = None
    ) -> None:
        with self._lock:
            ep = self._active_episodes.pop(episode_id, None)
            if not ep:
                logger.warning(f"Attempted to end inactive or not found episode: {episode_id}")
                return

            ep.ended_at = datetime.now(timezone.utc)
            ep.outcome = outcome # type: ignore
            ep.reward = reward

            agent_history = self._agent_history.setdefault(ep.agent_id, [])
            agent_history.append(ep)
            
            # Sort by start time just to ensure chronological order is preserved
            agent_history.sort(key=lambda x: x.started_at)
            
            logger.debug(f"Ended episode {episode_id} with outcome {outcome}")

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        with self._lock:
            # Check active first
            if episode_id in self._active_episodes:
                return self._active_episodes[episode_id]
            # Then check history
            for history in self._agent_history.values():
                for ep in history:
                    if ep.id == episode_id:
                        return ep
            return None

    def get_recent(
        self, agent_id: str, limit: int = 5, context: Optional[str] = None
    ) -> List[Episode]:
        with self._lock:
            history = self._agent_history.get(agent_id, [])
            
            # Active episodes count too
            active = [ep for ep in self._active_episodes.values() if ep.agent_id == agent_id]
            
            combined = history + active
            combined.sort(key=lambda x: x.started_at, reverse=True)
            
            if context:
                combined = [ep for ep in combined if ep.context == context]
                
            return combined[:limit]

    def _get_historical_ep(self, agent_id: str, episode_id: str) -> Optional[Episode]:
        history = self._agent_history.get(agent_id, [])
        for ep in history:
            if ep.id == episode_id:
                return ep
        return None

