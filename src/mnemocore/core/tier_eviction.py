"""
Tier Eviction Policies Module (Phase 6 Refactor)
=================================================
Eviction policies for managing tier capacity and promoting/demoting memories.

Implements:
- LRU (Least Recently Used)
- Importance-weighted eviction (LTP-based)
- Decay-triggered archival
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from loguru import logger

from mnemocore.core.config import HAIMConfig

if TYPE_CHECKING:
    from mnemocore.core.node import MemoryNode


class EvictionPolicy(Enum):
    """Available eviction policies."""
    LRU = "lru"
    LTP_LOWEST = "ltp_lowest"  # Evict lowest LTP strength
    IMPORTANCE = "importance"  # Combine LTP + epistemic/pragmatic values
    DECAY_TRIGGERED = "decay_triggered"  # Age-based with decay


class EvictionStrategy(ABC):
    """
    Abstract base class for eviction strategies.

    Each strategy determines which memories should be evicted
    when a tier reaches capacity.
    """

    @abstractmethod
    def select_victim(
        self,
        candidates: List["MemoryNode"],
        exclude_ids: Optional[List[str]] = None,
    ) -> Optional["MemoryNode"]:
        """
        Select a victim node for eviction.

        Args:
            candidates: List of candidate nodes to evict from.
            exclude_ids: IDs to protect from eviction.

        Returns:
            The node to evict, or None if no valid victim.
        """
        pass

    @abstractmethod
    def should_archive(self, node: "MemoryNode", threshold: float) -> bool:
        """
        Determine if a node should be archived to COLD tier.

        Args:
            node: The node to evaluate.
            threshold: Archive threshold (LTP or age-based).

        Returns:
            True if the node should be archived.
        """
        pass


class LRUEvictionStrategy(EvictionStrategy):
    """
    Least Recently Used eviction strategy.

    Evicts the node with the oldest last_accessed timestamp.
    """

    def select_victim(
        self,
        candidates: List["MemoryNode"],
        exclude_ids: Optional[List[str]] = None,
    ) -> Optional["MemoryNode"]:
        if not candidates:
            return None

        exclude_set = set(exclude_ids) if exclude_ids else set()
        valid_candidates = [n for n in candidates if n.id not in exclude_set]

        if not valid_candidates:
            return None

        victim = min(valid_candidates, key=lambda n: n.last_accessed)
        logger.debug(
            f"LRU evicting {victim.id} (last_accessed: {victim.last_accessed})"
        )
        return victim

    def should_archive(self, node: "MemoryNode", threshold: float) -> bool:
        # LRU doesn't directly map to archival; use age as proxy
        return node.age_days() > threshold


class LTPEvictionStrategy(EvictionStrategy):
    """
    LTP-based eviction strategy.

    Evicts the node with the lowest LTP strength.
    This prioritizes keeping important memories in faster tiers.
    """

    def select_victim(
        self,
        candidates: List["MemoryNode"],
        exclude_ids: Optional[List[str]] = None,
    ) -> Optional["MemoryNode"]:
        if not candidates:
            return None

        exclude_set = set(exclude_ids) if exclude_ids else set()
        valid_candidates = [n for n in candidates if n.id not in exclude_set]

        if not valid_candidates:
            return None

        victim = min(valid_candidates, key=lambda n: n.ltp_strength)
        logger.debug(
            f"LTP evicting {victim.id} (ltp_strength: {victim.ltp_strength:.4f})"
        )
        return victim

    def should_archive(self, node: "MemoryNode", threshold: float) -> bool:
        return node.ltp_strength < threshold


class ImportanceEvictionStrategy(EvictionStrategy):
    """
    Importance-weighted eviction strategy.

    Combines LTP strength with epistemic and pragmatic values
    to determine the least important memory.
    """

    def select_victim(
        self,
        candidates: List["MemoryNode"],
        exclude_ids: Optional[List[str]] = None,
    ) -> Optional["MemoryNode"]:
        if not candidates:
            return None

        exclude_set = set(exclude_ids) if exclude_ids else set()
        valid_candidates = [n for n in candidates if n.id not in exclude_set]

        if not valid_candidates:
            return None

        def importance_score(node: "MemoryNode") -> float:
            # Combine LTP with epistemic/pragmatic values
            base = node.ltp_strength
            epistemic = node.epistemic_value
            pragmatic = node.pragmatic_value
            # Weighted combination favoring memories with higher values
            return base * 0.7 + (epistemic + pragmatic) / 2 * 0.3

        victim = min(valid_candidates, key=importance_score)
        logger.debug(
            f"Importance evicting {victim.id} "
            f"(score: {importance_score(victim):.4f})"
        )
        return victim

    def should_archive(self, node: "MemoryNode", threshold: float) -> bool:
        # Archive if combined score is below threshold
        score = (
            node.ltp_strength * 0.7
            + (node.epistemic_value + node.pragmatic_value) / 2 * 0.3
        )
        return score < threshold


class DecayTriggeredStrategy(EvictionStrategy):
    """
    Decay-triggered eviction strategy.

    Uses time-based decay combined with access patterns.
    Memories that haven't been accessed recently and have
    decayed below threshold are candidates for eviction.
    """

    def __init__(self, decay_lambda: float = 0.01):
        self.decay_lambda = decay_lambda

    def select_victim(
        self,
        candidates: List["MemoryNode"],
        exclude_ids: Optional[List[str]] = None,
    ) -> Optional["MemoryNode"]:
        if not candidates:
            return None

        exclude_set = set(exclude_ids) if exclude_ids else set()
        valid_candidates = [n for n in candidates if n.id not in exclude_set]

        if not valid_candidates:
            return None

        def decay_score(node: "MemoryNode") -> float:
            # Lower score = better eviction candidate
            # Combine age and LTP decay
            import math

            age_days = node.age_days()
            decay_factor = math.exp(-self.decay_lambda * age_days)
            return node.ltp_strength * decay_factor

        victim = min(valid_candidates, key=decay_score)
        logger.debug(
            f"Decay evicting {victim.id} (decay_score: {decay_score(victim):.4f})"
        )
        return victim

    def should_archive(self, node: "MemoryNode", threshold: float) -> bool:
        import math

        age_days = node.age_days()
        decay_factor = math.exp(-self.decay_lambda * age_days)
        effective_strength = node.ltp_strength * decay_factor
        return effective_strength < threshold


class TierEvictionManager:
    """
    Manager for tier eviction policies.

    Coordinates eviction decisions across all tiers based on
    configured policies and thresholds.
    """

    # Policy registry
    _strategies: Dict[EvictionPolicy, EvictionStrategy] = {
        EvictionPolicy.LRU: LRUEvictionStrategy(),
        EvictionPolicy.LTP_LOWEST: LTPEvictionStrategy(),
        EvictionPolicy.IMPORTANCE: ImportanceEvictionStrategy(),
        EvictionPolicy.DECAY_TRIGGERED: None,  # Created dynamically with config
    }

    def __init__(self, config: HAIMConfig):
        self.config = config
        self.hot_policy = EvictionPolicy(
            config.tiers_hot.eviction_policy
        ) if config.tiers_hot.eviction_policy in [
            e.value for e in EvictionPolicy
        ] else EvictionPolicy.LTP_LOWEST

        self.warm_policy = EvictionPolicy(
            config.tiers_warm.eviction_policy
        ) if config.tiers_warm.eviction_policy in [
            e.value for e in EvictionPolicy
        ] else EvictionPolicy.LTP_LOWEST

        # Initialize decay-triggered strategy with configured lambda
        if EvictionPolicy.DECAY_TRIGGERED not in self._strategies or \
           self._strategies[EvictionPolicy.DECAY_TRIGGERED] is None:
            self._strategies[EvictionPolicy.DECAY_TRIGGERED] = \
                DecayTriggeredStrategy(decay_lambda=config.ltp.decay_lambda)

    def get_hot_strategy(self) -> EvictionStrategy:
        """Get the eviction strategy for HOT tier."""
        return self._strategies.get(self.hot_policy, self._strategies[EvictionPolicy.LTP_LOWEST])

    def get_warm_strategy(self) -> EvictionStrategy:
        """Get the eviction strategy for WARM tier."""
        return self._strategies.get(self.warm_strategy, self._strategies[EvictionPolicy.LTP_LOWEST])

    def select_hot_victim(
        self,
        candidates: List["MemoryNode"],
        exclude_ids: Optional[List[str]] = None,
    ) -> Optional["MemoryNode"]:
        """
        Select a victim for eviction from HOT tier.

        Args:
            candidates: Nodes to consider for eviction.
            exclude_ids: IDs to protect from eviction.

        Returns:
            The node to evict, or None.
        """
        strategy = self.get_hot_strategy()
        victim = strategy.select_victim(candidates, exclude_ids)
        if victim:
            victim.tier = "warm"  # Mark for demotion
        return victim

    def should_demote_to_warm(self, node: "MemoryNode") -> bool:
        """
        Check if a node should be demoted from HOT to WARM.

        Uses hysteresis to prevent thrashing between tiers.
        """
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.demote_delta
        return node.ltp_strength < (threshold - delta)

    def should_promote_to_hot(self, node: "MemoryNode") -> bool:
        """
        Check if a node should be promoted from WARM to HOT.

        Uses hysteresis to prevent thrashing between tiers.
        """
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.promote_delta
        return node.ltp_strength > (threshold + delta)

    def should_archive_to_cold(self, node: "MemoryNode") -> bool:
        """
        Check if a node should be archived from WARM to COLD.
        """
        threshold = self.config.tiers_warm.ltp_threshold_min

        # Check LTP threshold
        if node.ltp_strength < threshold:
            return True

        # Check age-based archival
        archive_days = self.config.tiers_warm.archive_threshold_days
        if node.age_days() > archive_days:
            return True

        return False

    def get_warm_candidates_for_archive(
        self,
        nodes: List["MemoryNode"],
        batch_size: int = 100,
    ) -> List["MemoryNode"]:
        """
        Get candidates for archival from WARM to COLD.

        Returns up to batch_size nodes that qualify for archival.
        """
        candidates = [n for n in nodes if self.should_archive_to_cold(n)]
        # Sort by LTP strength (lowest first) to archive least important first
        candidates.sort(key=lambda n: n.ltp_strength)
        return candidates[:batch_size]


__all__ = [
    "EvictionPolicy",
    "EvictionStrategy",
    "LRUEvictionStrategy",
    "LTPEvictionStrategy",
    "ImportanceEvictionStrategy",
    "DecayTriggeredStrategy",
    "TierEvictionManager",
]
