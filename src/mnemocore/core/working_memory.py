"""
Working Memory Service
======================
Manages short-term operational state (STM/WM) for individual agents or sessions.
Provides fast caching, item eviction (via LRU + importance), and contextual focus.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import threading
import logging

from .memory_model import WorkingMemoryItem, WorkingMemoryState

logger = logging.getLogger(__name__)


class WorkingMemoryService:
    def __init__(self, max_items_per_agent: int = 64):
        self.max_items_per_agent = max_items_per_agent
        self._states: Dict[str, WorkingMemoryState] = {}
        self._lock = threading.RLock()

    def _get_or_create_state(self, agent_id: str) -> WorkingMemoryState:
        with self._lock:
            if agent_id not in self._states:
                self._states[agent_id] = WorkingMemoryState(
                    agent_id=agent_id, max_items=self.max_items_per_agent
                )
            return self._states[agent_id]

    def push_item(self, agent_id: str, item: WorkingMemoryItem) -> None:
        """Push a new item into the agent's working memory, pruning if necessary."""
        with self._lock:
            state = self._get_or_create_state(agent_id)
            # Prevent exact duplicate IDs from being appended twice
            existing_idx = next(
                (i for i, x in enumerate(state.items) if x.id == item.id), None
            )
            if existing_idx is not None:
                state.items[existing_idx] = item
            else:
                state.items.append(item)
            
            self._prune(agent_id)

    def get_state(self, agent_id: str) -> Optional[WorkingMemoryState]:
        """Retrieve the current working memory state for an agent."""
        with self._lock:
            return self._states.get(agent_id)

    def clear(self, agent_id: str) -> None:
        """Clear the working memory for a specific agent."""
        with self._lock:
            if agent_id in self._states:
                self._states[agent_id].items.clear()

    def prune_all(self) -> None:
        """Prune TTL-expired items and overflows across all agents. Typically called by Pulse."""
        with self._lock:
            for agent_id in list(self._states.keys()):
                self._prune(agent_id)

    def _prune(self, agent_id: str) -> None:
        """Internal method to prune a specific agent's state based on TTL and capacity limits."""
        state = self._states.get(agent_id)
        if not state:
            return

        now = datetime.now(timezone.utc)
        active_items = []

        # 1. Filter out expired items based on TTL
        for item in state.items:
            expected_expiry = item.created_at + timedelta(seconds=item.ttl_seconds)
            if now < expected_expiry:
                active_items.append(item)

        # 2. If still over capacity, sort by importance (descending) then by freshness (descending)
        if len(active_items) > state.max_items:
            # We want to keep the highest importance / newest items
            active_items.sort(
                key=lambda x: (x.importance, x.created_at.timestamp()), reverse=True
            )
            active_items = active_items[: state.max_items]

            # Re-sort temporally for the final state list (oldest to newest)
            active_items.sort(key=lambda x: x.created_at.timestamp())

        state.items = active_items

    def promote_item(self, agent_id: str, item_id: str, bonus: float = 0.1) -> None:
        """Locally boost the importance of an item and refresh its creation time if accessed."""
        with self._lock:
            state = self._states.get(agent_id)
            if not state:
                return
            for item in state.items:
                if item.id == item_id:
                    item.importance = min(1.0, item.importance + bonus)
                    # Extend TTL window by effectively refreshing created_at (LRU-like behavior)
                    item.created_at = datetime.now(timezone.utc)
                    break

