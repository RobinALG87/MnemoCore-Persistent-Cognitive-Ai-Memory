"""
MnemoCore Storage Layer
=======================

Backup, snapshotting, import/export, vector compression, and hybrid search functionality.

Modules:
    backup_manager: Automated Qdrant snapshots with WAL (Write-Ahead Log)
    memory_exporter: Export memories to JSON/Parquet formats
    memory_importer: Import memories with schema validation and deduplication
    vector_compression: Product Quantization (PQ) and Scalar Quantization (INT8)
    hybrid_search: Dense + Sparse retrieval with RRF (Phase 4.6)
"""

from .backup_manager import (
    BackupManager,
    SnapshotInfo,
    BackupConfig,
    RecoverableBackup,
)
from .memory_exporter import (
    MemoryExporter,
    ExportFormat,
    ExportResult,
    ExportOptions,
)
from .memory_importer import (
    MemoryImporter,
    ImportResult,
    ImportOptions,
    DeduplicationStrategy,
)

# Vector compression imports - handle gracefully
try:
    from .vector_compression import (
        VectorCompressionConfig,
        ProductQuantization,
        ScalarQuantizer,
        VectorCompressor,
        CompressionMetadata,
        CompressedVector,
        CompressionMethod,
        create_compressor,
        get_compression_ratio,
    )
    _vector_compression_available = True
except ImportError as e:
    import logging
    logging.warning(f"Vector compression module import failed: {e}")
    _vector_compression_available = False

# Phase 4.6: Hybrid Search
try:
    from mnemocore.core.hybrid_search import (
        HybridSearchEngine,
        HybridSearchConfig,
        SearchResult,
        SparseEncoder,
        ReciprocalRankFusion,
    )
    _hybrid_search_available = True
except ImportError as e:
    import logging
    logging.warning(f"Hybrid search module import failed: {e}")
    _hybrid_search_available = False

# Binary vector compression (Phase 6)
try:
    from .binary_vector_compression import (
        BinaryCompressionConfig,
        BinaryCompressionMethod,
        BinaryProductQuantization,
        BinaryScalarQuantizer,
        BinaryVectorCompressor,
        BinaryCompressionMetadata,
        CompressedBinaryVector,
        create_binary_compressor,
        get_binary_compression_ratio,
    )
    _binary_vector_compression_available = True
except ImportError as e:
    import logging
    logging.warning(f"Binary vector compression module import failed: {e}")
    _binary_vector_compression_available = False

__all__ = [
    # Backup Manager
    "BackupManager",
    "SnapshotInfo",
    "BackupConfig",
    "RecoverableBackup",
    # Exporter
    "MemoryExporter",
    "ExportFormat",
    "ExportResult",
    "ExportOptions",
    # Importer
    "MemoryImporter",
    "ImportResult",
    "ImportOptions",
    "DeduplicationStrategy",
]

# Add vector compression exports if available
if _vector_compression_available:
    __all__.extend([
        "VectorCompressionConfig",
        "ProductQuantization",
        "ScalarQuantizer",
        "VectorCompressor",
        "CompressionMetadata",
        "CompressedVector",
        "CompressionMethod",
        "create_compressor",
        "get_compression_ratio",
    ])

# Add hybrid search exports if available
if _hybrid_search_available:
    __all__.extend([
        "HybridSearchEngine",
        "HybridSearchConfig",
        "SearchResult",
        "SparseEncoder",
        "ReciprocalRankFusion",
    ])

# Add binary vector compression exports if available
if _binary_vector_compression_available:
    __all__.extend([
        "BinaryCompressionConfig",
        "BinaryCompressionMethod",
        "BinaryProductQuantization",
        "BinaryScalarQuantizer",
        "BinaryVectorCompressor",
        "BinaryCompressionMetadata",
        "CompressedBinaryVector",
        "create_binary_compressor",
        "get_binary_compression_ratio",
    ])
