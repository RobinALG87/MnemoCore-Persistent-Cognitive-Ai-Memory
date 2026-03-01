"""
Forgetting Curve Manager (Phase 5.0 - Enhanced)
================================================
Implements Ebbinghaus-based spaced repetition scheduling with SM-2 algorithm,
per-agent learning profiles, and emotion-aware decay for MnemoCore.

This module provides backward-compatible imports from the new forgetting package.
All implementation has been moved to the core/forgetting/ package.

Enhanced Features:
  1. Per-agent LearningProfile with personalized decay constants
  2. SM-2 (SuperMemo 2) spaced repetition algorithm
  3. Emotional memory integration (via emotional_tag.py)
  4. ForgettingAnalytics dashboard for visualization

Key Algorithms:
  - SM-2: I(n+1) = I(n) * EF where EF is the easiness factor
  - Retention: R = e^(-T / S_i) * emotional_boost
  - Stability: S_i = S_base * (1 + k * access_count) * agent_modifier

Public API:
    manager = ForgettingCurveManager(engine)
    profile = manager.get_learning_profile(agent_id)
    await manager.record_review(node_id, quality=4)  # SM-2 quality 0-5
    analytics = ForgettingAnalytics(manager)
    dashboard = analytics.get_dashboard_data()

For direct access to components:
    from mnemocore.core.forgetting import (
        ForgettingCurveManager,
        LearningProfile,
        SM2State,
        ReviewEntry,
        ForgettingAnalytics,
    )
"""

# Re-export all components from the forgetting package for backward compatibility
from .forgetting import (
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
    # Core components
    LearningProfile,
    SM2State,
    ReviewEntry,
    # Manager
    ForgettingCurveManager,
    # Analytics
    ForgettingAnalytics,
    RetentionCurve,
    AgentAnalytics,
    # Convenience functions
    create_forgetting_manager,
    create_learning_profile,
    quality_from_confidence,
)


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
