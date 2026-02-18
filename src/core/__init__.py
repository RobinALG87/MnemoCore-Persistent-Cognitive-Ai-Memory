from .binary_hdv import BinaryHDV
from .hdv import HDV  # Deprecated - kept for backward compatibility
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
]
