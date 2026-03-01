"""
MnemoCore Core Module
=====================
The central cognitive processing engine and HDV operations.

This module provides the foundational components for MnemoCore's memory system:

Vector Operations (Binary HDV):
    - BinaryHDV: 16,384-dimensional binary vectors with XOR binding
    - TextEncoder: Convert text to HDV representations
    - Batch operations: Vectorized distance computations

Memory Storage:
    - MemoryNode: Individual memory unit with HDV, content, metadata
    - SynapticConnection: Weighted links between memories
    - HAIMEngine: Central coordinator for all memory operations

Tier Management:
    - TierManager: Orchestrates HOT/WARM/COLD tier transitions
    - HotTierStorage: In-memory fast access layer
    - WarmTierStorage: Redis or memory-mapped storage
    - ColdTierStorage: Qdrant or filesystem archive

Cognitive Services (Phase 5):
    - WorkingMemory: Active slot buffer (7+/-2 items)
    - EpisodicStore: Temporal episode chains
    - SemanticStore: Concept persistence
    - ProceduralStore: Skill library
    - MetaMemory: Anomaly detection

Advanced Features (Phase 4+):
    - BayesianLTPUpdater: Bayesian reliability model for memory strength
    - SemanticConsolidationWorker: Nightly clustering and pruning
    - ImmunologyLoop: Corruption detection and attractor convergence
    - GapDetector/GapFiller: Knowledge gap identification and filling
    - HNSWIndexManager: Approximate nearest neighbor search

Research Services (Phase 6):
    - StrategyBank: 5-phase strategy distillation loop
    - KnowledgeGraph: Spreading activation, community detection
    - MemoryScheduler: Priority job queue with interrupts
    - SAMEP: Multi-agent memory exchange protocol

Configuration:
    All settings are loaded from config.yaml via the config module.
    See docs/CONFIGURATION.md for complete reference.

Example:
    from mnemocore.core import HAIMEngine, BinaryHDV, TierManager

    engine = HAIMEngine()
    node_id = await engine.store("Hello world", metadata={"source": "user"})
    results = await engine.query("hello", k=5)
"""

from .binary_hdv import BinaryHDV
# Backward-compatibility re-export – import HDV lazily to avoid triggering
# the module-level DeprecationWarning at package import time.

def __getattr__(name):
    if name == 'HDV':
        from .hdv import HDV  # Deprecated - kept for backward compatibility
        return HDV
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from .node import MemoryNode
from .synapse import SynapticConnection
from .engine import HAIMEngine
from .exceptions import (
    MnemoCoreError,
    StorageError,
    StorageConnectionError,
    StorageTimeoutError,
    DataCorruptionError,
    VectorError,
    DimensionMismatchError,
    VectorOperationError,
    ConfigurationError,
    CircuitOpenError,
    MemoryOperationError,
)

# ── Phase 4.0 ────────────────────────────────────────────────────────
from .attention import XORAttentionMasker, AttentionConfig, AttentionResult
from .bayesian_ltp import BayesianLTPUpdater, BayesianState, get_bayesian_updater
from .semantic_consolidation import SemanticConsolidationWorker, SemanticConsolidationConfig
from .immunology import ImmunologyLoop, ImmunologyConfig
from .gap_detector import GapDetector, GapDetectorConfig, GapRecord
from .gap_filler import GapFiller, GapFillerConfig
from .hnsw_index import HNSWIndexManager
from .synapse_index import SynapseIndex

# ── Phase 6 Refactor ──────────────────────────────────────────────────
from .tier_manager import TierManager
from .tier_storage import (
    TierInterface,
    HotTierStorage,
    WarmTierStorage,
    ColdTierStorage,
)
from .tier_eviction import (
    EvictionPolicy,
    EvictionStrategy,
    LRUEvictionStrategy,
    LTPEvictionStrategy,
    ImportanceEvictionStrategy,
    DecayTriggeredStrategy,
    TierEvictionManager,
)
from .tier_scoring import (
    TierScore,
    SearchResult,
    LTPCalculator,
    RecencyScorer,
    ImportanceScorer,
    TierScoringManager,
    TemporalScorer,
    SearchScorer,
)

# ── Phase 6.0: Embedding Version Registry ───────────────────────────────
from .embedding_registry import (
    EmbeddingModelSpec,
    MigrationTask,
    MigrationPlan,
    MigrationStatus,
    Priority,
    EmbeddingRegistry,
    MigrationPlanner,
    ReEmbeddingWorker,
    EmbeddingVersionManager,
    create_vector_metadata,
    verify_vector_compatibility,
)

# ── Phase 6.0: Engine Refactor Modules ──────────────────────────────────
from .engine_core import EngineCoreOperations
from .engine_lifecycle import EngineLifecycleManager
from .engine_coordinator import EngineCoordinator

__all__ = [
    "BinaryHDV",
    "HDV",
    "MemoryNode",
    "SynapticConnection",
    "HAIMEngine",
    # Exceptions
    "MnemoCoreError",
    "StorageError",
    "StorageConnectionError",
    "StorageTimeoutError",
    "DataCorruptionError",
    "VectorError",
    "DimensionMismatchError",
    "VectorOperationError",
    "ConfigurationError",
    "CircuitOpenError",
    "MemoryOperationError",
    # Phase 4.0
    "XORAttentionMasker",
    "AttentionConfig",
    "AttentionResult",
    "BayesianLTPUpdater",
    "BayesianState",
    "get_bayesian_updater",
    "SemanticConsolidationWorker",
    "SemanticConsolidationConfig",
    "ImmunologyLoop",
    "ImmunologyConfig",
    "GapDetector",
    "GapDetectorConfig",
    "GapRecord",
    "GapFiller",
    "GapFillerConfig",
    "HNSWIndexManager",
    "SynapseIndex",
    # Phase 6 Refactor
    "TierManager",
    "TierInterface",
    "HotTierStorage",
    "WarmTierStorage",
    "ColdTierStorage",
    "EvictionPolicy",
    "EvictionStrategy",
    "LRUEvictionStrategy",
    "LTPEvictionStrategy",
    "ImportanceEvictionStrategy",
    "DecayTriggeredStrategy",
    "TierEvictionManager",
    "TierScore",
    "SearchResult",
    "LTPCalculator",
    "RecencyScorer",
    "ImportanceScorer",
    "TierScoringManager",
    "TemporalScorer",
    "SearchScorer",
    # Phase 6.0: Embedding Version Registry
    "EmbeddingModelSpec",
    "MigrationTask",
    "MigrationPlan",
    "MigrationStatus",
    "Priority",
    "EmbeddingRegistry",
    "MigrationPlanner",
    "ReEmbeddingWorker",
    "EmbeddingVersionManager",
    "create_vector_metadata",
    "verify_vector_compatibility",
    # Phase 6.0: Engine Refactor Modules
    "EngineCoreOperations",
    "EngineLifecycleManager",
    "EngineCoordinator",
]
