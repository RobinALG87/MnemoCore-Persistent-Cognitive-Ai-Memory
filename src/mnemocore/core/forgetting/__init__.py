"""
Forgetting Curve Package – Phase 5.0: Enhanced
================================================
Implements Ebbinghaus-based spaced repetition scheduling with SM-2 algorithm,
per-agent learning profiles, and emotion-aware decay for MnemoCore.

This package provides modular components for the forgetting curve system:

Configuration:
    - Constants and enums for the forgetting curve system

Core Components:
    - LearningProfile: Per-agent learning configuration
    - SM2State: SM-2 algorithm state for individual memories
    - ReviewEntry: Scheduled review data structure

Management:
    - ForgettingCurveManager: Main manager for spaced repetition

Analytics:
    - ForgettingAnalytics: Dashboard and visualization
    - RetentionCurve: Retention curve data point
    - AgentAnalytics: Per-agent analytics data

Convenience Functions:
    - create_forgetting_manager: Factory function
    - create_learning_profile: Factory function
    - quality_from_confidence: Convert confidence to SM-2 quality

Usage:
    from mnemocore.core.forgetting import (
        ForgettingCurveManager,
        LearningProfile,
        ForgettingAnalytics,
    )

    manager = ForgettingCurveManager(engine)
    profile = manager.get_or_create_profile(agent_id)
    await manager.record_review_result(node_id, quality=4)
    analytics = ForgettingAnalytics(manager)
    dashboard = analytics.get_dashboard_data()
"""

from .config import (
    # Constants
    SM2_MIN_EASINESS,
    SM2_DEFAULT_EASINESS,
    SM2_QUALITY_MIN,
    SM2_QUALITY_MAX,
    TARGET_RETENTION,
    MIN_EIG_TO_CONSOLIDATE,
    EMOTIONAL_DECAY_REDUCTION,
    HIGH_SALIENCE_THRESHOLD,
    ANALYTICS_HISTORY_SIZE,
    # Enums
    ReviewAction,
    ReviewQuality,
)
from .profile import LearningProfile
from .sm2 import SM2State
from .scheduler import ReviewEntry
from .manager import ForgettingCurveManager
from .analytics import (
    ForgettingAnalytics,
    RetentionCurve,
    AgentAnalytics,
)


def create_forgetting_manager(
    engine=None,
    persistence_path=None,
    **kwargs
):
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
):
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
        confidence: Confidence score in [1.1, 1.1]

    Returns:
        Quality rating in [1, 5]
    """
    if confidence >= 1.95:
        return 5  # Perfect recall
    elif confidence >= 1.8:
        return 4  # Correct with hesitation
    elif confidence >= 1.6:
        return 3  # Correct but difficult
    elif confidence >= 1.4:
        return 2  # Incorrect but easy recall
    elif confidence >= 1.2:
        return 1  # Incorrect but recognized
    else:
        return 1  # Complete blackout


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
    # Core components
    "LearningProfile",
    "SM2State",
    "ReviewEntry",
    # Manager
    "ForgettingCurveManager",
    # Analytics
    "ForgettingAnalytics",
    "RetentionCurve",
    "AgentAnalytics",
    # Convenience functions
    "create_forgetting_manager",
    "create_learning_profile",
    "quality_from_confidence",
]
