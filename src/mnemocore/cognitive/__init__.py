"""
Cognitive Memory Module
=======================
Phase 6.0 - Higher-order cognitive processes including reconstructive memory,
schema-based inference, and associative synthesis.

Phase 7.0 - Episodic Future Thinking (EFT) for scenario simulation and
prospective cognitive processing.

Phase 7.5 - Forgetting Analytics with SM-2 spaced repetition integration.
"""

from .memory_reconstructor import (
    ReconstructiveRecall,
    ReconstructedMemory,
    ReconstructionConfig,
    ReconstructionResult,
)

from .context_optimizer import (
    ContextWindowPrioritizer,
    TokenCounter,
    SemanticChunker,
    ContextBuilder,
    RankedMemory,
    ModelContextLimits,
    ModelProvider,
    ChunkConfig,
    ScoringWeights,
    OptimizationResult,
    create_prioritizer,
    rank_memories,
    MODEL_LIMITS,
)

from .future_thinking import (
    EFTConfig,
    ScenarioType,
    ScenarioNode,
    ScenarioStore,
    EpisodeFutureSimulator,
    AttentionIntegration,
    create_future_thinking_pipeline,
)

from .associations import (
    AssociationType,
    AssociationDirection,
    AssociationEdge,
    AssociationConfig,
    GraphMetrics,
    AssociationStrengthener,
    AssociationsNetwork,
    AssociationRecallIntegrator,
    create_associations_network,
    create_network_from_nodes,
    reinforce_associations,
    find_related_memories,
)

from .forgetting_analytics import (
    ForgettingAnalyticsCognitive,
    ChartData,
    DashboardWidget,
    ChartType,
    create_forgetting_analytics,
    get_dashboard_html,
)

__all__ = [
    "ReconstructiveRecall",
    "ReconstructedMemory",
    "ReconstructionConfig",
    "ReconstructionResult",
    "ContextWindowPrioritizer",
    "TokenCounter",
    "SemanticChunker",
    "ContextBuilder",
    "RankedMemory",
    "ModelContextLimits",
    "ModelProvider",
    "ChunkConfig",
    "ScoringWeights",
    "OptimizationResult",
    "create_prioritizer",
    "rank_memories",
    "MODEL_LIMITS",
    # Phase 7.0: Episodic Future Thinking
    "EFTConfig",
    "ScenarioType",
    "ScenarioNode",
    "ScenarioStore",
    "EpisodeFutureSimulator",
    "AttentionIntegration",
    "create_future_thinking_pipeline",
    # Phase 6.0: Association Network
    "AssociationType",
    "AssociationDirection",
    "AssociationEdge",
    "AssociationConfig",
    "GraphMetrics",
    "AssociationStrengthener",
    "AssociationsNetwork",
    "AssociationRecallIntegrator",
    "create_associations_network",
    "create_network_from_nodes",
    "reinforce_associations",
    "find_related_memories",
    # Phase 7.5: Forgetting Analytics
    "ForgettingAnalyticsCognitive",
    "ChartData",
    "DashboardWidget",
    "ChartType",
    "create_forgetting_analytics",
    "get_dashboard_html",
]
