from typing import List, Optional

import numpy as np
from loguru import logger

from .binary_hdv import BinaryHDV, majority_bundle
from .config import PreferenceConfig


class PreferenceStore:
    """
    Phase 12.3: Preference Learning
    Maintains a persistent vector representing implicit user preferences
    based on logged decisions or positive feedback.
    """

    def __init__(self, config: PreferenceConfig, dimension: int):
        self.config = config
        self.dimension = dimension
        # The preference vector represents the "ideal" or "preferred" region
        self.preference_vector: Optional[BinaryHDV] = None
        self.decision_history: List[BinaryHDV] = []

    def log_decision(self, context_hdv: BinaryHDV, outcome: float) -> None:
        """
        Logs a decision or feedback event.
        `outcome`: positive value (e.g. 1.0) for good feedback, negative (-1.0) for bad feedback.
        If outcome is positive, the preference vector shifts slightly toward `context_hdv`.
        If outcome is negative, the preference vector shifts away (invert context_hdv).
        """
        if not self.config.enabled:
            return

        target_hdv = context_hdv if outcome >= 0 else context_hdv.invert()
        self.decision_history.append(target_hdv)

        # Maintain history size
        if len(self.decision_history) > self.config.history_limit:
            self.decision_history.pop(0)

        # Update preference vector via majority bundling of recent positive shifts
        self.preference_vector = majority_bundle(self.decision_history)
        logger.debug(
            f"Logged decision (outcome={outcome:.2f}). Preference vector updated."
        )

    def bias_score(self, target_hdv: BinaryHDV, base_score: float) -> float:
        """
        Biases a retrieval score using the preference vector if one exists.
        Formula: new_score = base_score + (learning_rate * preference_similarity)
        """
        if not self.config.enabled or self.preference_vector is None:
            return base_score

        pref_sim = self.preference_vector.similarity(target_hdv)

        # We apply the learning_rate as the max potential boost an item can get from mapping exactly to preferences
        return base_score + (self.config.learning_rate * pref_sim)
