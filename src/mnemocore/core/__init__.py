# Backward-compatibility re-export – import HDV lazily to avoid triggering
# the module-level DeprecationWarning at package import time.
from .binary_hdv import BinaryHDV


def __getattr__(name):
    if name == "HDV":
        from .hdv import HDV  # Deprecated - kept for backward compatibility

        return HDV
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── Phase 4.0 ────────────────────────────────────────────────────────
from .attention import AttentionConfig, AttentionResult, XORAttentionMasker
from .bayesian_ltp import BayesianLTPUpdater, BayesianState, get_bayesian_updater
from .engine import HAIMEngine
from .exceptions import (
    CircuitOpenError,
    ConfigurationError,
    DataCorruptionError,
    DimensionMismatchError,
    MemoryOperationError,
    MnemoCoreError,
    StorageConnectionError,
    StorageError,
    StorageTimeoutError,
    VectorError,
    VectorOperationError,
)
from .gap_detector import GapDetector, GapDetectorConfig, GapRecord
from .gap_filler import GapFiller, GapFillerConfig
from .hnsw_index import HNSWIndexManager
from .immunology import ImmunologyConfig, ImmunologyLoop
from .node import MemoryNode
from .semantic_consolidation import (
    SemanticConsolidationConfig,
    SemanticConsolidationWorker,
)
from .synapse import SynapticConnection
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
