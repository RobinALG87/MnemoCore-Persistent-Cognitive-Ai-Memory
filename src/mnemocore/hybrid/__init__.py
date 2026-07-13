"""Deterministic, exact-scope hybrid retrieval over AgentMemory."""

from .contracts import ExactScopeError, HybridRecallResult, RetrievalRequest, SCORING_VERSION
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
    "SCORING_VERSION",
    "SyncHybridMemoryRuntime",
    "lexical_similarity",
]
