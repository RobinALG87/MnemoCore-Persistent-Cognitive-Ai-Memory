from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from .binary_hdv import BinaryHDV, majority_bundle
from .config import ContextConfig


class TopicTracker:
    """
    Phase 12.2: Contextual Awareness
    Tracks the rolling conversational context using an HDV moving average.
    Detects topic shifts and resets the context when appropriate.
    """

    def __init__(self, config: ContextConfig, dimension: int):
        self.config = config
        self.dimension = dimension
        self.context_vector: Optional[BinaryHDV] = None
        self.history: List[BinaryHDV] = []

    def add_query(self, query_hdv: BinaryHDV) -> Tuple[bool, float]:
        """
        Adds a new query to the tracker.
        Returns (is_shift, similarity).
        If is_shift is True, it means a topic boundary was detected.
        """
        if not self.config.enabled:
            return False, 1.0

        if self.context_vector is None:
            self.context_vector = query_hdv
            self.history = [query_hdv]
            return False, 1.0

        similarity = self.context_vector.similarity(query_hdv)

        # Detect shift
        if similarity < self.config.shift_threshold:
            logger.info(
                f"Topic shift detected! Similarity {similarity:.3f} < {self.config.shift_threshold}"
            )
            self.reset(query_hdv)
            return True, similarity

        # Update rolling context
        self.history.append(query_hdv)
        if len(self.history) > self.config.rolling_window_size:
            self.history.pop(0)

        # Recompute the majority bundle for the current window
        self.context_vector = majority_bundle(self.history)
        return False, similarity

    def reset(self, new_context: Optional[BinaryHDV] = None):
        """Resets the topic tracker, optionally seeding it with a new query."""
        if new_context:
            self.context_vector = new_context
            self.history = [new_context]
        else:
            self.context_vector = None
            self.history = []

    def get_context(self) -> Optional[BinaryHDV]:
        """Returns the current topic context vector."""
        return self.context_vector
