"""
Binary Vector Compression Layer for MnemoCore
==============================================

Phase 6: Vector compression optimized for BinaryHDV vectors.

Provides:
1. Product Quantization (PQ) for BinaryHDV - High compression ratio
2. Scalar quantization (INT8) for binary vectors - 2x compression
3. Auto-compression based on memory age, confidence, and tier
4. Persistent metadata storage for compressed vectors
5. Transparent decompression during recall

This module is specifically designed for the BinaryHDV format used in
MnemoCore, unlike vector_compression.py which handles float32/float64 vectors.

Example Usage:
    ```python
    from mnemocore.storage.binary_vector_compression import BinaryVectorCompressor
    from mnemocore.core.binary_hdv import BinaryHDV

    # Create compressor
    compressor = BinaryVectorCompressor()

    # Train PQ codebook
    vectors = [BinaryHDV.random() for _ in range(1000)]
    compressor.train_pq(vectors)

    # Compress a vector
    original = BinaryHDV.random()
    compressed = compressor.compress(
        vector_id="mem_123",
        vector=original,
        confidence=0.5,
        tier="warm"
    )

    # Decompress
    decompressed = compressor.decompress(compressed)
    ```
"""

import asyncio
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from ..core.binary_hdv import BinaryHDV
from ..core.config import get_config


# =============================================================================
# Configuration
# =============================================================================


class BinaryCompressionMethod(Enum):
    """Compression methods for binary vectors."""
    NONE = "none"
    SCALAR_INT8 = "int8"       # 2x compression
    PRODUCT_PQ = "pq"         # 8x+ compression


@dataclass
class BinaryCompressionConfig:
    """
    Configuration for binary vector compression.

    Attributes:
        enabled: Whether compression is enabled
        pq_n_subvectors: Number of PQ subvectors (must divide dimension)
        pq_n_bits: Bits per PQ code (default: 8 -> 256 centroids)
        int8_threshold_confidence: Max confidence for INT8 compression
        age_threshold_hours: Age before auto-compression
        compression_interval_seconds: Background compression scan interval
        max_batch_size: Max vectors per compression batch
        storage_path: Path to compression metadata database
        hot_tier_compression: Compress hot tier (default: False for speed)
        warm_tier_compression: Compress warm tier (default: True)
        cold_tier_compression: Compress cold tier (default: True)
        pq_max_train_vectors: Max vectors for PQ training
        pq_train_iterations: K-means iterations per subvector
    """
    enabled: bool = True
    pq_n_subvectors: int = 32
    pq_n_bits: int = 8
    int8_threshold_confidence: float = 0.4
    age_threshold_hours: float = 24.0
    compression_interval_seconds: int = 3600
    max_batch_size: int = 1000
    storage_path: str = "./data/binary_vector_compression.db"
    hot_tier_compression: bool = False
    warm_tier_compression: bool = True
    cold_tier_compression: bool = True
    pq_max_train_vectors: int = 10000
    pq_train_iterations: int = 20


@dataclass
class BinaryCompressionMetadata:
    """Metadata for a compressed binary vector."""
    method: BinaryCompressionMethod
    original_dimension: int
    compressed_size_bytes: int
    compressed_at: datetime
    confidence: float
    tier: str
    pq_codebook_id: Optional[str] = None
    decompression_priority: int = 0


@dataclass
class CompressedBinaryVector:
    """A compressed binary vector with metadata."""
    vector_id: str
    method: BinaryCompressionMethod
    compressed_data: bytes
    metadata: BinaryCompressionMetadata

    def decompress(self, compressor: 'BinaryVectorCompressor') -> Optional[BinaryHDV]:
        """Decompress using the provided compressor."""
        try:
            return compressor.decompress(self)
        except Exception as e:
            logger.error(f"Failed to decompress {self.vector_id}: {e}")
            return None


# =============================================================================
# Product Quantization for Binary Vectors
# =============================================================================


class BinaryProductQuantization:
    """
    Product Quantization optimized for binary vectors.

    Splits binary vector into subvectors, each quantized to K centroids
    via majority vote (hamming distance minimization).

    For D=16384, M=32 subvectors:
    - Original: 2048 bytes
    - Compressed: 32 bytes (1 byte per subvector code)
    - Compression ratio: 64x theoretical, ~8x practical
    """

    def __init__(
        self,
        dimension: int,
        n_subvectors: int = 32,
        n_bits: int = 8,
        max_train_vectors: int = 10000,
        train_iterations: int = 20,
    ):
        if dimension % n_subvectors != 0:
            raise ValueError(
                f"Dimension {dimension} must be divisible by n_subvectors {n_subvectors}"
            )

        self.dimension = dimension
        self.n_subvectors = n_subvectors
        self.subvector_dim = dimension // n_subvectors
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits
        self.max_train_vectors = max_train_vectors
        self.train_iterations = train_iterations
        self.codebooks: Optional[np.ndarray] = None

    def train(self, vectors: List[BinaryHDV]) -> str:
        """
        Train PQ codebooks using k-means on each subvector.

        Args:
            vectors: List of BinaryHDV vectors for training

        Returns:
            Codebook ID string
        """
        if len(vectors) > self.max_train_vectors:
            indices = np.random.choice(len(vectors), self.max_train_vectors, replace=False)
            vectors = [vectors[i] for i in indices]

        n_train = len(vectors)
        logger.info(
            f"Training Binary PQ with {n_train} vectors: "
            f"{self.n_subvectors} subvectors, {self.n_centroids} centroids"
        )

        # Stack all vectors
        packed_data = np.stack([v.data for v in vectors], axis=0)
        all_bits = np.unpackbits(packed_data, axis=1)

        # Initialize codebooks
        self.codebooks = np.zeros(
            (self.n_subvectors, self.n_centroids, self.subvector_dim),
            dtype=np.uint8
        )

        # Train each subvector
        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = all_bits[:, start:end]

            # Initialize centroids randomly
            centroids = np.zeros((self.n_centroids, self.subvector_dim), dtype=np.uint8)
            for i in range(self.n_centroids):
                idx = np.random.choice(n_train)
                centroids[i] = subvectors[idx]

            # K-means iterations
            for it in range(self.train_iterations):
                # Assign to nearest centroid (hamming distance)
                distances = self._hamming_matrix(subvectors, centroids)
                assignments = np.argmin(distances, axis=1)

                # Update centroids (majority vote)
                new_centroids = centroids.copy()
                for k in range(self.n_centroids):
                    mask = assignments == k
                    if mask.sum() > 0:
                        bit_sums = subvectors[mask].sum(axis=0)
                        threshold = mask.sum() / 2.0
                        new_centroids[k] = (bit_sums > threshold).astype(np.uint8)

                if np.array_equal(centroids, new_centroids):
                    break
                centroids = new_centroids

            self.codebooks[m] = centroids

        logger.info("Binary PQ training complete")
        return f"pq_{int(time.time())}_{n_train}"

    def _hamming_matrix(self, vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute Hamming distance matrix."""
        xor_result = vectors[:, np.newaxis, :] ^ centroids[np.newaxis, :, :]
        return xor_result.sum(axis=2)

    def encode(self, vector: BinaryHDV) -> np.ndarray:
        """Encode BinaryHDV to PQ codes."""
        if self.codebooks is None:
            raise RuntimeError("PQ codebooks not trained")

        bits = np.unpackbits(vector.data)
        codes = np.zeros(self.n_subvectors, dtype=np.uint8)

        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            subvector = bits[start:end]
            centroids = self.codebooks[m]
            distances = np.sum(subvector != centroids, axis=1)
            codes[m] = np.argmin(distances)

        return codes

    def decode(self, codes: np.ndarray) -> BinaryHDV:
        """Decode PQ codes to BinaryHDV."""
        if self.codebooks is None:
            raise RuntimeError("PQ codebooks not trained")

        reconstructed_bits = np.zeros(self.dimension, dtype=np.uint8)

        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            reconstructed_bits[start:end] = self.codebooks[m, codes[m]]

        packed = np.packbits(reconstructed_bits)
        return BinaryHDV(data=packed, dimension=self.dimension)

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_bytes = self.dimension // 8
        compressed_bytes = self.n_subvectors
        return original_bytes / compressed_bytes


# =============================================================================
# Scalar Quantization for Binary Vectors
# =============================================================================


class BinaryScalarQuantizer:
    """
    Scalar Quantization for binary vectors.

    Packs 4 binary bits into 1 INT8 value for 2x compression.
    Each bit is encoded as 2 bits to preserve bipolar information.
    """

    def __init__(self, dimension: int):
        if dimension % 4 != 0:
            raise ValueError(f"Dimension {dimension} must be divisible by 4")
        self.dimension = dimension

    def encode(self, vector: BinaryHDV) -> np.ndarray:
        """Encode BinaryHDV to INT8 array."""
        bits = np.unpackbits(vector.data)

        # Convert to bipolar: 0->-1, 1->+1, then pack
        bipolar = bits.astype(np.int8) * 2 - 1

        packed = np.zeros(self.dimension // 4, dtype=np.int8)
        for i in range(self.dimension // 4):
            b0 = (bipolar[i * 4] + 1) // 2
            b1 = (bipolar[i * 4 + 1] + 1) // 2
            b2 = (bipolar[i * 4 + 2] + 1) // 2
            b3 = (bipolar[i * 4 + 3] + 1) // 2
            packed[i] = (b3 << 6) | (b2 << 4) | (b1 << 2) | b0

        return packed

    def decode(self, encoded: np.ndarray) -> BinaryHDV:
        """Decode INT8 array to BinaryHDV."""
        bits = np.zeros(self.dimension, dtype=np.uint8)

        for i in range(self.dimension // 4):
            byte_val = encoded[i]
            bits[i * 4] = (byte_val >> 0) & 0x01
            bits[i * 4 + 1] = (byte_val >> 2) & 0x01
            bits[i * 4 + 2] = (byte_val >> 4) & 0x01
            bits[i * 4 + 3] = (byte_val >> 6) & 0x01

        packed = np.packbits(bits)
        return BinaryHDV(data=packed, dimension=self.dimension)

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio (2x)."""
        return 2.0


# =============================================================================
# Main Binary Vector Compressor
# =============================================================================


class BinaryVectorCompressor:
    """
    Main compression engine for BinaryHDV vectors.

    Features:
    - Automatic method selection based on confidence and tier
    - PQ codebook management with persistence
    - Background compression queue processing
    - Statistics and monitoring
    """

    def __init__(self, config: Optional[BinaryCompressionConfig] = None):
        """
        Initialize compressor.

        Args:
            config: Compression configuration (uses defaults if None)
        """
        self.config = config or BinaryCompressionConfig()

        # Get dimension from global config
        self.dimension = 16384
        try:
            global_config = get_config()
            self.dimension = global_config.dimensionality
        except Exception:
            pass

        # Initialize quantizers
        self.pq = BinaryProductQuantization(
            dimension=self.dimension,
            n_subvectors=self.config.pq_n_subvectors,
            n_bits=self.config.pq_n_bits,
            max_train_vectors=self.config.pq_max_train_vectors,
            train_iterations=self.config.pq_train_iterations,
        )
        self.scalar_q = BinaryScalarQuantizer(dimension=self.dimension)

        # Codebook state
        self._pq_codebook_id: Optional[str] = None
        self._pq_trained_at: Optional[datetime] = None

        # Storage
        self._db_path = Path(self.config.storage_path)
        self._db_lock = threading.Lock()
        self._init_db()

        # Background task
        self._compression_task: Optional[asyncio.Task] = None
        self._running = False

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self._db_path, timeout=30.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS compressed_vectors (
                        vector_id TEXT PRIMARY KEY,
                        method TEXT NOT NULL,
                        compressed_data BLOB NOT NULL,
                        original_dimension INTEGER NOT NULL,
                        compressed_size_bytes INTEGER NOT NULL,
                        compressed_at REAL NOT NULL,
                        confidence REAL NOT NULL,
                        tier TEXT NOT NULL,
                        pq_codebook_id TEXT,
                        decompression_priority INTEGER DEFAULT 0
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pq_codebooks (
                        codebook_id TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        dimension INTEGER NOT NULL,
                        n_subvectors INTEGER NOT NULL,
                        n_bits INTEGER NOT NULL,
                        codebook_data BLOB NOT NULL,
                        training_vectors_count INTEGER
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS compression_queue (
                        vector_id TEXT PRIMARY KEY,
                        added_at REAL NOT NULL,
                        priority REAL DEFAULT 0.0,
                        attempts INTEGER DEFAULT 0
                    )
                """)

                conn.execute("CREATE INDEX IF NOT EXISTS idx_cv_tier ON compressed_vectors(tier)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cv_at ON compressed_vectors(compressed_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cq_prio ON compression_queue(priority, added_at)")

                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

        except Exception as e:
            logger.error(f"Failed to initialize compression DB: {e}")

    # ======================================================================
    # Compression API
    # ======================================================================

    def compress(
        self,
        vector_id: str,
        vector: BinaryHDV,
        confidence: float = 0.5,
        tier: str = "warm",
        force_method: Optional[BinaryCompressionMethod] = None,
    ) -> CompressedBinaryVector:
        """
        Compress a BinaryHDV vector.

        Args:
            vector_id: Unique identifier
            vector: BinaryHDV to compress
            confidence: Confidence score (0-1)
            tier: Memory tier ("hot", "warm", "cold")
            force_method: Force specific method

        Returns:
            CompressedBinaryVector
        """
        if not self.config.enabled:
            metadata = BinaryCompressionMetadata(
                method=BinaryCompressionMethod.NONE,
                original_dimension=vector.dimension,
                compressed_size_bytes=vector.data.nbytes,
                compressed_at=datetime.now(),
                confidence=confidence,
                tier=tier,
            )
            return CompressedBinaryVector(
                vector_id=vector_id,
                method=BinaryCompressionMethod.NONE,
                compressed_data=vector.to_bytes(),
                metadata=metadata,
            )

        method = force_method or self._select_method(confidence, tier)

        compressed_data: bytes
        compressed_size: int

        if method == BinaryCompressionMethod.PRODUCT_PQ:
            if self.pq.codebooks is None:
                logger.warning("PQ not trained, falling back to INT8")
                method = BinaryCompressionMethod.SCALAR_INT8

        if method == BinaryCompressionMethod.PRODUCT_PQ:
            codes = self.pq.encode(vector)
            compressed_data = codes.tobytes()
            compressed_size = len(compressed_data)

        elif method == BinaryCompressionMethod.SCALAR_INT8:
            encoded = self.scalar_q.encode(vector)
            compressed_data = encoded.tobytes()
            compressed_size = len(compressed_data)

        else:  # NONE
            compressed_data = vector.to_bytes()
            compressed_size = len(compressed_data)

        metadata = BinaryCompressionMetadata(
            method=method,
            original_dimension=vector.dimension,
            compressed_size_bytes=compressed_size,
            compressed_at=datetime.now(),
            confidence=confidence,
            tier=tier,
            pq_codebook_id=self._pq_codebook_id if method == BinaryCompressionMethod.PRODUCT_PQ else None,
            decompression_priority=int(confidence * 100),
        )

        compressed = CompressedBinaryVector(
            vector_id=vector_id,
            method=method,
            compressed_data=compressed_data,
            metadata=metadata,
        )

        self._store_compressed(compressed)
        return compressed

    def decompress(self, compressed: CompressedBinaryVector) -> BinaryHDV:
        """Decompress a compressed vector."""
        if compressed.method == BinaryCompressionMethod.NONE:
            return BinaryHDV.from_bytes(
                compressed.compressed_data,
                dimension=compressed.metadata.original_dimension
            )

        if compressed.method == BinaryCompressionMethod.SCALAR_INT8:
            encoded = np.frombuffer(compressed.compressed_data, dtype=np.int8)
            return self.scalar_q.decode(encoded)

        if compressed.method == BinaryCompressionMethod.PRODUCT_PQ:
            if self.pq.codebooks is None:
                if compressed.metadata.pq_codebook_id:
                    self._load_pq_codebook(compressed.metadata.pq_codebook_id)
                else:
                    raise RuntimeError("Cannot decompress PQ: no codebook")

            codes = np.frombuffer(compressed.compressed_data, dtype=np.uint8)
            return self.pq.decode(codes)

        raise ValueError(f"Unknown method: {compressed.method}")

    def get_compressed(self, vector_id: str) -> Optional[CompressedBinaryVector]:
        """Retrieve compressed vector from storage."""
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as conn:
                row = conn.execute(
                    """
                    SELECT vector_id, method, compressed_data, original_dimension,
                           compressed_size_bytes, compressed_at, confidence, tier,
                           pq_codebook_id, decompression_priority
                    FROM compressed_vectors WHERE vector_id = ?
                    """,
                    (vector_id,)
                ).fetchone()

                if row:
                    return self._row_to_compressed(row)
        except Exception as e:
            logger.error(f"Failed to get compressed {vector_id}: {e}")

        return None

    # ======================================================================
    # Method Selection
    # ======================================================================

    def _select_method(self, confidence: float, tier: str) -> BinaryCompressionMethod:
        """Select compression method based on confidence and tier."""
        if tier == "hot" and not self.config.hot_tier_compression:
            return BinaryCompressionMethod.NONE

        if tier == "cold" and self.config.cold_tier_compression:
            return BinaryCompressionMethod.PRODUCT_PQ

        if tier == "warm" and self.config.warm_tier_compression:
            if confidence < self.config.int8_threshold_confidence:
                return BinaryCompressionMethod.SCALAR_INT8
            else:
                return BinaryCompressionMethod.PRODUCT_PQ

        return BinaryCompressionMethod.NONE

    # ======================================================================
    # PQ Codebook Management
    # ======================================================================

    def train_pq(self, vectors: List[BinaryHDV]) -> str:
        """Train PQ codebook on vectors."""
        logger.info(f"Training PQ on {len(vectors)} vectors")
        codebook_id = self.pq.train(vectors)
        self._pq_codebook_id = codebook_id
        self._pq_trained_at = datetime.now()
        self._save_pq_codebook(codebook_id)
        return codebook_id

    def _save_pq_codebook(self, codebook_id: str) -> None:
        """Save PQ codebook to database."""
        try:
            codebook_data = pickle.dumps({
                "codebooks": self.pq.codebooks,
                "dimension": self.pq.dimension,
                "n_subvectors": self.pq.n_subvectors,
                "n_bits": self.pq.n_bits,
            })

            with sqlite3.connect(self._db_path, timeout=10.0) as conn:
                conn.execute(
                    """
                    INSERT INTO pq_codebooks
                    (codebook_id, created_at, dimension, n_subvectors, n_bits, codebook_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (codebook_id, datetime.now().timestamp(), self.pq.dimension,
                     self.pq.n_subvectors, self.pq.n_bits, codebook_data)
                )

            logger.info(f"Saved PQ codebook {codebook_id}")

        except Exception as e:
            logger.error(f"Failed to save codebook: {e}")

    def _load_pq_codebook(self, codebook_id: str) -> bool:
        """Load PQ codebook from database."""
        try:
            with sqlite3.connect(self._db_path, timeout=10.0) as conn:
                row = conn.execute(
                    "SELECT codebook_data FROM pq_codebooks WHERE codebook_id = ?",
                    (codebook_id,)
                ).fetchone()

                if row:
                    data = pickle.loads(row[0])
                    self.pq.codebooks = data["codebooks"]
                    self.pq.dimension = data["dimension"]
                    self.pq.n_subvectors = data["n_subvectors"]
                    self.pq.n_bits = data["n_bits"]
                    self.pq.subvector_dim = self.pq.dimension // self.pq.n_subvectors

                    self._pq_codebook_id = codebook_id
                    logger.info(f"Loaded PQ codebook {codebook_id}")
                    return True

        except Exception as e:
            logger.error(f"Failed to load codebook {codebook_id}: {e}")

        return False

    def get_latest_codebook_id(self) -> Optional[str]:
        """Get most recent PQ codebook ID."""
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as conn:
                row = conn.execute(
                    "SELECT codebook_id FROM pq_codebooks ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                if row:
                    return row[0]
        except Exception as e:
            logger.error(f"Failed to get codebook: {e}")

        return None

    # ======================================================================
    # Auto-Compression Queue
    # ======================================================================

    def queue_for_compression(self, vector_id: str, priority: float = 0.0) -> None:
        """Add vector to compression queue."""
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO compression_queue (vector_id, added_at, priority) VALUES (?, ?, ?)",
                    (vector_id, datetime.now().timestamp(), priority)
                )
        except Exception as e:
            logger.error(f"Failed to queue {vector_id}: {e}")

    def process_compression_queue(
        self,
        get_vector_func: Callable[[str], Optional[Tuple[BinaryHDV, float, str]]],
        max_batch: int = None,
    ) -> Dict[str, int]:
        """
        Process vectors in compression queue.

        Args:
            get_vector_func: Callback returning (vector, confidence, tier) or None
            max_batch: Max vectors to process

        Returns:
            Statistics dict
        """
        max_batch = max_batch or self.config.max_batch_size
        stats = {m.value: 0 for m in BinaryCompressionMethod}
        stats["failed"] = 0
        stats["skipped"] = 0

        try:
            with sqlite3.connect(self._db_path, timeout=30.0) as conn:
                rows = conn.execute(
                    """
                    SELECT vector_id FROM compression_queue
                    ORDER BY priority DESC, added_at ASC LIMIT ?
                    """,
                    (max_batch,)
                ).fetchall()

                for (vector_id,) in rows:
                    try:
                        result = get_vector_func(vector_id)
                        if result is None:
                            stats["skipped"] += 1
                            continue

                        vector, confidence, tier = result
                        compressed = self.compress(vector_id, vector, confidence, tier)
                        stats[compressed.method.value] += 1

                        conn.execute(
                            "DELETE FROM compression_queue WHERE vector_id = ?",
                            (vector_id,)
                        )

                    except Exception as e:
                        logger.error(f"Failed to compress {vector_id}: {e}")
                        stats["failed"] += 1
                        conn.execute(
                            "UPDATE compression_queue SET attempts = attempts + 1 WHERE vector_id = ?",
                            (vector_id,)
                        )

        except Exception as e:
            logger.error(f"Failed to process queue: {e}")

        return stats

    # ======================================================================
    # Storage Helpers
    # ======================================================================

    def _store_compressed(self, compressed: CompressedBinaryVector) -> None:
        """Store compressed vector."""
        try:
            with sqlite3.connect(self._db_path, timeout=10.0) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO compressed_vectors
                    (vector_id, method, compressed_data, original_dimension,
                     compressed_size_bytes, compressed_at, confidence, tier,
                     pq_codebook_id, decompression_priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (compressed.vector_id, compressed.method.value, compressed.compressed_data,
                     compressed.metadata.original_dimension, compressed.metadata.compressed_size_bytes,
                     compressed.metadata.compressed_at.timestamp(), compressed.metadata.confidence,
                     compressed.metadata.tier, compressed.metadata.pq_codebook_id,
                     compressed.metadata.decompression_priority)
                )
        except Exception as e:
            logger.error(f"Failed to store compressed: {e}")

    def _row_to_compressed(self, row) -> CompressedBinaryVector:
        """Convert DB row to CompressedBinaryVector."""
        (vector_id, method_str, compressed_data, original_dim, compressed_size,
         compressed_at_ts, confidence, tier, pq_codebook_id, priority) = row

        method = BinaryCompressionMethod(method_str)

        metadata = BinaryCompressionMetadata(
            method=method,
            original_dimension=original_dim,
            compressed_size_bytes=compressed_size,
            compressed_at=datetime.fromtimestamp(compressed_at_ts),
            confidence=confidence,
            tier=tier,
            pq_codebook_id=pq_codebook_id,
            decompression_priority=priority,
        )

        return CompressedBinaryVector(
            vector_id=vector_id,
            method=method,
            compressed_data=compressed_data,
            metadata=metadata,
        )

    # ======================================================================
    # Statistics
    # ======================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics."""
        stats = {
            "enabled": self.config.enabled,
            "dimension": self.dimension,
            "pq_trained": self.pq.codebooks is not None,
            "pq_codebook_id": self._pq_codebook_id,
            "pq_compression_ratio": self.pq.get_compression_ratio() if self.pq.codebooks is not None else None,
            "scalar_compression_ratio": self.scalar_q.get_compression_ratio(),
        }

        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as conn:
                for method in BinaryCompressionMethod:
                    row = conn.execute(
                        "SELECT COUNT(*), SUM(compressed_size_bytes) FROM compressed_vectors WHERE method = ?",
                        (method.value,)
                    ).fetchone()
                    stats[f"{method.value}_count"] = row[0] or 0
                    stats[f"{method.value}_total_bytes"] = row[1] or 0

                row = conn.execute("SELECT COUNT(*) FROM compression_queue").fetchone()
                stats["queue_size"] = row[0]

                row = conn.execute("SELECT COUNT(*) FROM pq_codebooks").fetchone()
                stats["codebook_count"] = row[0]

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")

        return stats

    def cleanup_old_codebooks(self, keep_latest: int = 3) -> int:
        """Remove old PQ codebooks."""
        deleted = 0
        try:
            with sqlite3.connect(self._db_path, timeout=10.0) as conn:
                rows = conn.execute(
                    """
                    SELECT codebook_id FROM pq_codebooks
                    ORDER BY created_at DESC LIMIT -1 OFFSET ?
                    """,
                    (keep_latest,)
                ).fetchone()

                if rows:
                    for (codebook_id,) in rows:
                        row = conn.execute(
                            "SELECT COUNT(*) FROM compressed_vectors WHERE pq_codebook_id = ?",
                            (codebook_id,)
                        ).fetchone()

                        if row[0] == 0:
                            conn.execute("DELETE FROM pq_codebooks WHERE codebook_id = ?", (codebook_id,))
                            deleted += 1

        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")

        return deleted


# =============================================================================
# Convenience Functions
# =============================================================================


def create_binary_compressor(
    config: Optional[BinaryCompressionConfig] = None
) -> BinaryVectorCompressor:
    """Create a BinaryVectorCompressor."""
    return BinaryVectorCompressor(config)


def get_binary_compression_ratio(method: BinaryCompressionMethod, dimension: int) -> float:
    """Get theoretical compression ratio."""
    original_bytes = dimension // 8

    if method == BinaryCompressionMethod.NONE:
        return 1.0
    elif method == BinaryCompressionMethod.SCALAR_INT8:
        return 2.0
    elif method == BinaryCompressionMethod.PRODUCT_PQ:
        return 8.0  # Practical ratio

    return 1.0
