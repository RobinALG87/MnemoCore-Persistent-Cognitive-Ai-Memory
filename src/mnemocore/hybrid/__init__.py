"""Deterministic, exact-scope hybrid retrieval over AgentMemory."""

from .contracts import (
    SCORING_VERSION,
    ExactScopeError,
    HybridRecallResult,
    RetrievalObservability,
    RetrievalRequest,
)
from .factory import RuntimeFactory, RuntimeMetadata
from .retrieval import BinaryHDV, DeterministicHybridRetriever, lexical_similarity
from .runtime import HybridMemoryRuntime, PlanApplyReceipt, SyncHybridMemoryRuntime

__all__ = [
    "BinaryHDV",
    "DeterministicHybridRetriever",
    "ExactScopeError",
    "HybridMemoryRuntime",
    "PlanApplyReceipt",
    "HybridRecallResult",
    "RetrievalRequest",
    "RetrievalObservability",
    "RuntimeFactory",
    "RuntimeMetadata",
    "SCORING_VERSION",
    "SyncHybridMemoryRuntime",
    "lexical_similarity",
]
