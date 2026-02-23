from .binary_hdv import BinaryHDV
# Backward-compatibility re-export – import HDV lazily to avoid triggering
# the module-level DeprecationWarning at package import time.
from .binary_hdv import BinaryHDV

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
