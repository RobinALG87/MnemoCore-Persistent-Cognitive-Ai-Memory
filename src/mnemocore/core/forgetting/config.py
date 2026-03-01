"""
Forgetting Curve Constants and Enums
=====================================
Shared constants and enums for the forgetting curve system.
"""

from enum import Enum


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


__all__ = [
    # Constants
    "SM2_MIN_EASINESS",
    "SM2_DEFAULT_EASINESS",
    "SM2_QUALITY_MIN",
    "SM2_QUALITY_MAX",
    "TARGET_RETENTION",
    "MIN_EIG_TO_CONSOLIDATE",
    "EMOTIONAL_DECAY_REDUCTION",
    "HIGH_SALIENCE_THRESHOLD",
    "ANALYTICS_HISTORY_SIZE",
    # Enums
    "ReviewAction",
    "ReviewQuality",
]
