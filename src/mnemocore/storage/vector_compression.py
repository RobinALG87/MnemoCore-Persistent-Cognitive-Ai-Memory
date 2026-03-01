"""
Vector Compression for MnemoCore
==================================

Provides Product Quantization (PQ) and Scalar Quantization (INT8)
for efficient vector storage and faster distance computation.

Features:
    - Product Quantization (PQ) for high compression ratios
    - Scalar Quantization (INT8) for balanced compression/accuracy
    - Codec training and serialization
    - Batch compression/decompression
    - SIMD-friendly distance computation

Usage:
    ```python
    from mnemocore.storage import create_compressor, CompressionMethod

    # Create compressor
    compressor = create_compressor(
        method=CompressionMethod.PRODUCT_QUANTIZATION,
        dimension=16384,
        n_subvectors=256,
    )

    # Train on vectors
    await compressor.fit(vectors)

    # Compress vectors
    compressed = compressor.compress(vectors)

    # Decompress
    decompressed = compressor.decompress(compressed)
    ```
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from mnemocore.core.exceptions import ValidationError

# Magic bytes for safe binary format identification
VECTOR_FORMAT_MAGIC = b"MNCV"  # MnemoCore Vector
VECTOR_FORMAT_VERSION = 1


# =============================================================================
# Enums and Configuration
# =============================================================================


class CompressionMethod(Enum):
    """Vector compression methods."""
    NONE = "none"
    SCALAR_INT8 = "scalar_int8"       # 8-bit scalar quantization
    PRODUCT_QUANTIZATION = "pq"       # Product Quantization
    BINARY = "binary"                 # Binary quantization (sign)


@dataclass
class CompressionMetadata:
    """Metadata about a compression operation."""
    method: CompressionMethod
    original_dimension: int
    compressed_size_bytes: int
    compression_ratio: float
    reconstruction_error: Optional[float] = None
    training_samples: int = 0
    subvectors: Optional[int] = None
    n_bits: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "original_dimension": self.original_dimension,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_ratio": self.compression_ratio,
            "reconstruction_error": self.reconstruction_error,
            "training_samples": self.training_samples,
            "subvectors": self.subvectors,
            "n_bits": self.n_bits,
        }


@dataclass
class CompressedVector:
    """A compressed vector representation."""
    data: np.ndarray
    metadata: CompressionMetadata

    def to_bytes(self) -> bytes:
        """
        Serialize to bytes using safe binary format.

        Format:
        - 4 bytes: Magic bytes "MNCV"
        - 1 byte: Format version
        - 4 bytes: Metadata JSON length (uint32, big-endian)
        - N bytes: Metadata JSON (UTF-8)
        - 4 bytes: Data dtype length (uint32, big-endian)
        - M bytes: Data dtype string (UTF-8)
        - 4 bytes: Data shape length (uint32, big-endian)
        - P bytes: Data shape as JSON array
        - Rest: Raw numpy data bytes (tobytes())
        """
        metadata_json = json.dumps(self.metadata.to_dict())

        # Convert numpy data to bytes
        data_bytes = self.data.tobytes()
        dtype_str = str(self.data.dtype)
        shape_json = json.dumps(list(self.data.shape))

        parts = [
            VECTOR_FORMAT_MAGIC,
            struct.pack(">B", VECTOR_FORMAT_VERSION),
            struct.pack(">I", len(metadata_json)),
            metadata_json.encode("utf-8"),
            struct.pack(">I", len(dtype_str)),
            dtype_str.encode("utf-8"),
            struct.pack(">I", len(shape_json)),
            shape_json.encode("utf-8"),
            data_bytes,
        ]

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CompressedVector":
        """
        Deserialize from bytes using safe binary format.

        Raises:
            ValidationError: If the format is invalid or corrupted.
        """
        if len(raw) < 4:
            raise ValidationError("data", "Invalid compressed vector: too short")

        # Check magic bytes
        if raw[:4] != VECTOR_FORMAT_MAGIC:
            raise ValidationError(
                "data",
                f"Invalid compressed vector: bad magic bytes. "
                "This may indicate corrupted data or an unsafe pickle payload was rejected."
            )

        offset = 4

        # Version
        version = struct.unpack(">B", raw[offset:offset + 1])[0]
        offset += 1
        if version != VECTOR_FORMAT_VERSION:
            raise ValidationError(
                "data",
                f"Unsupported compressed vector version: {version}"
            )

        # Metadata JSON
        metadata_len = struct.unpack(">I", raw[offset:offset + 4])[0]
        offset += 4
        metadata_json = raw[offset:offset + metadata_len].decode("utf-8")
        offset += metadata_len
        metadata_dict = json.loads(metadata_json)

        # Data dtype
        dtype_len = struct.unpack(">I", raw[offset:offset + 4])[0]
        offset += 4
        dtype_str = raw[offset:offset + dtype_len].decode("utf-8")
        offset += dtype_len

        # Validate dtype against allowlist to prevent injection
        _SAFE_DTYPES = {"float32", "float64", "float16", "uint8", "int8", "int16", "int32", "int64", "uint16", "uint32", "uint64"}
        if dtype_str not in _SAFE_DTYPES:
            raise ValidationError(
                "data",
                f"Untrusted dtype: {dtype_str!r}. Allowed: {_SAFE_DTYPES}"
            )

        # Data shape
        shape_len = struct.unpack(">I", raw[offset:offset + 4])[0]
        offset += 4
        shape_json = raw[offset:offset + shape_len].decode("utf-8")
        offset += shape_len
        shape = tuple(json.loads(shape_json))

        # Raw data
        data_bytes = raw[offset:]
        data = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str)).reshape(shape)
        # Make a copy since frombuffer creates a read-only view
        data = data.copy()

        return cls(
            data=data,
            metadata=CompressionMetadata(**metadata_dict),
        )


@dataclass
class VectorCompressionConfig:
    """Configuration for vector compression."""
    method: CompressionMethod = CompressionMethod.SCALAR_INT8

    # PQ settings
    n_subvectors: int = 256          # Number of PQ subvectors
    n_pq_bits: int = 8               # Bits per PQ code

    # Scalar quantization settings
    scalar_bits: int = 8             # Bits for scalar quantization

    # Training settings
    training_samples: int = 10000    # Max samples for training
    max_iterations: int = 100        # Max iterations for k-means

    # Binary settings
    binary_threshold: float = 0.0    # Threshold for binarization


# =============================================================================
# Scalar Quantization
# =============================================================================


class ScalarQuantizer:
    """
    Scalar Quantization (INT8) for vectors.

    Compresses float32/float64 vectors to int8 for 4x compression.
    Uses per-dimension min/max scaling for accuracy.
    """

    def __init__(self, bits: int = 8):
        """
        Initialize scalar quantizer.

        Args:
            bits: Number of bits per value (typically 8)
        """
        self.bits = bits
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.dimension: Optional[int] = None
        self.is_fitted = False

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizer":
        """
        Fit quantizer to vectors.

        Args:
            vectors: Array of shape (n_vectors, dimension)

        Returns:
            Self for chaining
        """
        if vectors.ndim != 2:
            raise ValidationError(
                "vectors",
                f"Expected 2D array, got shape {vectors.shape}",
            )

        self.dimension = vectors.shape[1]
        self.min = vectors.min(axis=0)
        self.max = vectors.max(axis=0)

        # Compute scale for quantization
        range_val = self.max - self.min
        range_val[range_val == 0] = 1.0  # Avoid division by zero

        self.scale = range_val / ((1 << self.bits) - 1)
        self.is_fitted = True

        return self

    def compress(self, vectors: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compress vectors to int8.

        Args:
            vectors: Array of shape (n_vectors, dimension)

        Returns:
            Tuple of (compressed vectors, metadata dict)
        """
        if not self.is_fitted:
            raise ValidationError("quantizer", "Quantizer not fitted")

        # Normalize to [0, 1]
        normalized = (vectors - self.min) / (self.max - self.min)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Quantize to uint8
        quantized = (normalized * 255).astype(np.uint8)

        metadata = {
            "min": self.min,
            "max": self.max,
            "scale": self.scale,
            "dimension": self.dimension,
            "bits": self.bits,
        }

        return quantized, metadata

    def decompress(
        self,
        compressed: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Decompress uint8 vectors back to float32.

        Args:
            compressed: Compressed uint8 array
            metadata: Compression metadata (uses fitted if None)

        Returns:
            Decompressed float32 array
        """
        if metadata is None:
            min_val = self.min
            max_val = self.max
        else:
            min_val = metadata["min"]
            max_val = metadata["max"]

        # Dequantize
        normalized = compressed.astype(np.float32) / 255.0
        decompressed = normalized * (max_val - min_val) + min_val

        return decompressed

    def compute_distance(
        self,
        query: np.ndarray,
        compressed_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Compute approximate distances using compressed vectors.

        Uses SIMD-friendly operations for efficiency.

        Args:
            query: Query vector (float32)
            compressed_vectors: Compressed vectors (uint8)

        Returns:
            Array of distances
        """
        # Compress query
        query_compressed, _ = self.compress(query.reshape(1, -1))
        query_compressed = query_compressed[0]

        # Use squared Euclidean distance approximation
        # This works well for uint8 due to bounded range
        diff = compressed_vectors.astype(np.int16) - query_compressed.astype(np.int16)
        distances = np.sum(diff ** 2, axis=1)

        return distances.astype(np.float32)


# =============================================================================
# Product Quantization
# =============================================================================


class ProductQuantization:
    """
    Product Quantization (PQ) for high-compression vector storage.

    Splits vectors into subvectors and quantizes each independently.
    Provides excellent compression ratios with good accuracy.
    """

    def __init__(
        self,
        dimension: int,
        n_subvectors: int = 256,
        n_bits: int = 8,
    ):
        """
        Initialize PQ codec.

        Args:
            dimension: Original vector dimension
            n_subvectors: Number of subvectors
            n_bits: Bits per codebook entry
        """
        if dimension % n_subvectors != 0:
            raise ValidationError(
                "dimension",
                f"Dimension {dimension} must be divisible by n_subvectors {n_subvectors}",
            )

        self.dimension = dimension
        self.n_subvectors = n_subvectors
        self.subvector_dim = dimension // n_subvectors
        self.n_bits = n_bits
        self.codebook_size = 1 << n_bits  # 256 for n_bits=8

        # Codebooks: one per subvector
        self.codebooks: Optional[np.ndarray] = None  # Shape: (n_subvectors, codebook_size, subvector_dim)
        self.is_fitted = False

    def fit(
        self,
        vectors: np.ndarray,
        max_iterations: int = 100,
    ) -> "ProductQuantization":
        """
        Train PQ codebooks using k-means.

        Args:
            vectors: Training vectors (n_samples, dimension)
            max_iterations: Maximum k-means iterations

        Returns:
            Self for chaining
        """
        n_samples = min(vectors.shape[0], 10000)
        sample_indices = np.random.choice(
            vectors.shape[0],
            n_samples,
            replace=False,
        )
        training_vectors = vectors[sample_indices]

        self.codebooks = np.zeros(
            (self.n_subvectors, self.codebook_size, self.subvector_dim),
            dtype=np.float32,
        )

        # Train each subvector independently
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = start_idx + self.subvector_dim

            subvectors = training_vectors[:, start_idx:end_idx]

            # Initialize codebook with random samples
            codebook = subvectors[
                np.random.choice(subvectors.shape[0], self.codebook_size, replace=False)
            ]

            # K-means clustering
            for _ in range(max_iterations):
                # Assign to nearest codebook entry
                distances = self._compute_distances_to_codebook(subvectors, codebook)
                assignments = np.argmin(distances, axis=1)

                # Update codebook
                new_codebook = codebook.copy()
                for j in range(self.codebook_size):
                    mask = assignments == j
                    if mask.sum() > 0:
                        new_codebook[j] = subvectors[mask].mean(axis=0)

                # Check convergence
                if np.allclose(codebook, new_codebook, atol=1e-4):
                    break

                codebook = new_codebook

            self.codebooks[i] = codebook
            logger.debug(f"Trained PQ subvector {i+1}/{self.n_subvectors}")

        self.is_fitted = True
        return self

    def _compute_distances_to_codebook(
        self,
        vectors: np.ndarray,
        codebook: np.ndarray,
    ) -> np.ndarray:
        """Compute distances from vectors to codebook entries."""
        # vectors: (n, sub_dim)
        # codebook: (codebook_size, sub_dim)
        # result: (n, codebook_size)
        return np.sum((vectors[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2, axis=2)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors to PQ codes.

        Args:
            vectors: Float32 vectors (n, dimension)

        Returns:
            uint8 codes (n, n_subvectors)
        """
        if not self.is_fitted:
            raise ValidationError("pq", "PQ not fitted")

        n_vectors = vectors.shape[0]
        codes = np.zeros((n_vectors, self.n_subvectors), dtype=np.uint8)

        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = start_idx + self.subvector_dim

            subvectors = vectors[:, start_idx:end_idx]
            codebook = self.codebooks[i]

            distances = self._compute_distances_to_codebook(subvectors, codebook)
            codes[:, i] = np.argmin(distances, axis=1).astype(np.uint8)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes back to vectors.

        Args:
            codes: uint8 codes (n, n_subvectors)

        Returns:
            Reconstructed vectors (n, dimension)
        """
        if not self.is_fitted:
            raise ValidationError("pq", "PQ not fitted")

        n_vectors = codes.shape[0]
        vectors = np.zeros((n_vectors, self.dimension), dtype=np.float32)

        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = start_idx + self.subvector_dim

            # Lookup codebook entries
            codebook = self.codebooks[i]
            vectors[:, start_idx:end_idx] = codebook[codes[:, i]]

        return vectors

    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """
        Precompute distance table for asymmetric distance computation.

        Args:
            query: Query vector (dimension,)

        Returns:
            Distance table (n_subvectors, codebook_size)
        """
        if not self.is_fitted:
            raise ValidationError("pq", "PQ not fitted")

        distance_table = np.zeros(
            (self.n_subvectors, self.codebook_size),
            dtype=np.float32,
        )

        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = start_idx + self.subvector_dim

            subvector = query[start_idx:end_idx]
            codebook = self.codebooks[i]

            distances = np.sum((codebook - subvector) ** 2, axis=1)
            distance_table[i, :] = distances

        return distance_table

    def compute_distances(
        self,
        distance_table: np.ndarray,
        codes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute distances using asymmetric distance computation.

        Args:
            distance_table: From compute_distance_table
            codes: Encoded vectors (n, n_subvectors)

        Returns:
            Distances (n,)
        """
        # Sum distances from table using codes as indices
        distances = np.zeros(codes.shape[0], dtype=np.float32)

        for i in range(self.n_subvectors):
            distances += distance_table[i, codes[:, i]]

        return distances

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_bytes = self.dimension * 4  # float32
        compressed_bytes = self.n_subvectors * 1  # uint8 per subvector
        return original_bytes / compressed_bytes


# =============================================================================
# Binary Quantization
# =============================================================================


class BinaryQuantizer:
    """
    Binary quantization for vectors.

    Converts float vectors to binary (sign bits) for maximum compression.
    Suitable for cosine similarity on normalized vectors.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Initialize binary quantizer.

        Args:
            threshold: Threshold for binarization (default 0)
        """
        self.threshold = threshold
        self.dimension: Optional[int] = None
        self.is_fitted = False

    def fit(self, vectors: np.ndarray) -> "BinaryQuantizer":
        """Fit quantizer (just stores dimension)."""
        self.dimension = vectors.shape[1]
        self.is_fitted = True
        return self

    def compress(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compress vectors to binary using vectorized operations.

        Args:
            vectors: Float32 vectors of shape (n_vectors, dimension)

        Returns:
            Binary vectors (packed as uint8) of shape (n_vectors, packed_dim)
        """
        # Binarize
        binary = (vectors > self.threshold).astype(np.uint8)

        # Vectorized bit packing using np.packbits
        # np.packbits packs along the last axis by default
        packed = np.packbits(binary, axis=1)

        return packed

    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        """
        Decompress binary vectors back to float32 using vectorized operations.

        Args:
            compressed: Packed uint8 array of shape (n_vectors, packed_dim)

        Returns:
            Float32 array of shape (n_vectors, dimension)
        """
        # Vectorized unpacking
        unpacked = np.unpackbits(compressed, axis=1)

        # Truncate to original dimension and convert to float32
        vectors = unpacked[:, :self.dimension].astype(np.float32)

        return vectors

    def hamming_distance(
        self,
        query_compressed: np.ndarray,
        vectors_compressed: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Hamming distance between compressed vectors using vectorized popcount.

        Args:
            query_compressed: Single compressed query (1D or 2D with shape (1, packed_dim))
            vectors_compressed: Multiple compressed vectors (n_vectors, packed_dim)

        Returns:
            Hamming distances as 1D array of shape (n_vectors,)
        """
        # Ensure query is broadcastable
        if query_compressed.ndim == 1:
            query_compressed = query_compressed[np.newaxis, :]

        # XOR and count set bits (vectorized)
        xor_result = np.bitwise_xor(vectors_compressed, query_compressed)

        # Vectorized popcount using lookup table
        # Pre-computed popcount for all 256 possible byte values
        popcount_table = np.array([
            bin(i).count("1") for i in range(256)
        ], dtype=np.int32)

        # Apply lookup table to all XOR results and sum along axis 1
        distances = popcount_table[xor_result].sum(axis=1)

        return distances


# =============================================================================
# Unified Vector Compressor
# =============================================================================


class VectorCompressor:
    """
    Unified interface for vector compression.

    Supports multiple compression methods with automatic method
    selection and batch processing.
    """

    def __init__(self, config: VectorCompressionConfig):
        """
        Initialize vector compressor.

        Args:
            config: Compression configuration
        """
        self.config = config
        self._quantizer: Optional[
            Union[ScalarQuantizer, ProductQuantization, BinaryQuantizer]
        ] = None
        self._init_compressor()

    def _init_compressor(self):
        """Initialize the underlying compressor."""
        if self.config.method == CompressionMethod.SCALAR_INT8:
            self._quantizer = ScalarQuantizer(bits=self.config.scalar_bits)
        elif self.config.method == CompressionMethod.PRODUCT_QUANTIZATION:
            # Dimension must be set for PQ
            raise ValidationError(
                "config",
                "Use fit() to initialize PQ with dimension",
            )
        elif self.config.method == CompressionMethod.BINARY:
            self._quantizer = BinaryQuantizer(threshold=self.config.binary_threshold)
        else:
            self._quantizer = None

    async def fit(self, vectors: np.ndarray) -> "VectorCompressor":
        """
        Train compressor on vectors.

        Args:
            vectors: Training vectors (n_samples, dimension)

        Returns:
            Self for chaining
        """
        if self.config.method == CompressionMethod.PRODUCT_QUANTIZATION:
            dimension = vectors.shape[1]
            self._quantizer = ProductQuantization(
                dimension=dimension,
                n_subvectors=self.config.n_subvectors,
                n_bits=self.config.n_pq_bits,
            )
            self._quantizer.fit(vectors, max_iterations=self.config.max_iterations)
        else:
            if self._quantizer is None:
                raise ValidationError("compressor", "Invalid compression method")
            self._quantizer.fit(vectors)

        return self

    def compress(self, vectors: np.ndarray) -> CompressedVector:
        """
        Compress vectors.

        Args:
            vectors: Float32 vectors to compress

        Returns:
            CompressedVector with data and metadata
        """
        if self._quantizer is None:
            raise ValidationError("compressor", "Compressor not initialized")

        if isinstance(self._quantizer, ScalarQuantizer):
            compressed, _ = self._quantizer.compress(vectors)
        elif isinstance(self._quantizer, ProductQuantization):
            compressed = self._quantizer.encode(vectors)
        elif isinstance(self._quantizer, BinaryQuantizer):
            compressed = self._quantizer.compress(vectors)
        else:
            raise ValidationError("compressor", "Unknown quantizer type")

        original_size = vectors.nbytes
        compressed_size = compressed.nbytes
        compression_ratio = original_size / compressed_size

        metadata = CompressionMetadata(
            method=self.config.method,
            original_dimension=vectors.shape[1],
            compressed_size_bytes=compressed_size,
            compression_ratio=compression_ratio,
        )

        return CompressedVector(data=compressed, metadata=metadata)

    def decompress(self, compressed: CompressedVector) -> np.ndarray:
        """
        Decompress vectors.

        Args:
            compressed: CompressedVector

        Returns:
            Decompressed float32 vectors
        """
        if self._quantizer is None:
            raise ValidationError("compressor", "Compressor not initialized")

        if isinstance(self._quantizer, ScalarQuantizer):
            return self._quantizer.decompress(compressed.data)
        elif isinstance(self._quantizer, ProductQuantization):
            return self._quantizer.decode(compressed.data)
        elif isinstance(self._quantizer, BinaryQuantizer):
            return self._quantizer.decompress(compressed.data)
        else:
            raise ValidationError("compressor", "Unknown quantizer type")

    def save(self, path: Union[str, Path]):
        """
        Save compressor to file using safe format.

        Uses a directory containing:
        - config.json: Configuration metadata
        - state.json: Quantizer state metadata
        - data.npy: NumPy arrays (codebooks, min/max, etc.)

        This replaces the unsafe pickle-based serialization.
        """
        path = Path(path)
        save_dir = path if path.suffix == "" else path.with_suffix("")
        save_dir.mkdir(parents=True, exist_ok=True)

        config_data = {
            "method": self.config.method.value,
            "n_subvectors": self.config.n_subvectors,
            "n_pq_bits": self.config.n_pq_bits,
            "scalar_bits": self.config.scalar_bits,
            "binary_threshold": self.config.binary_threshold,
        }

        state_data = {"type": None}
        arrays_to_save = {}

        if isinstance(self._quantizer, ScalarQuantizer):
            state_data = {
                "type": "scalar",
                "dimension": self._quantizer.dimension,
                "bits": self._quantizer.bits,
            }
            arrays_to_save["min"] = self._quantizer.min
            arrays_to_save["max"] = self._quantizer.max
            arrays_to_save["scale"] = self._quantizer.scale
        elif isinstance(self._quantizer, ProductQuantization):
            state_data = {
                "type": "pq",
                "dimension": self._quantizer.dimension,
                "n_subvectors": self._quantizer.n_subvectors,
                "n_bits": self._quantizer.n_bits,
            }
            arrays_to_save["codebooks"] = self._quantizer.codebooks
        elif isinstance(self._quantizer, BinaryQuantizer):
            state_data = {
                "type": "binary",
                "dimension": self._quantizer.dimension,
                "threshold": self._quantizer.threshold,
            }

        # Save config and state as JSON
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        with open(save_dir / "state.json", "w") as f:
            json.dump(state_data, f, indent=2)

        # Save numpy arrays using np.savez (safe format)
        if arrays_to_save:
            np.savez(save_dir / "data.npz", **arrays_to_save)

        logger.info(f"Saved compressor to {save_dir}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorCompressor":
        """
        Load compressor from file (safe format).

        Raises:
            ValidationError: If the format is invalid or files are corrupted.
        """
        path = Path(path)
        load_dir = path if path.suffix == "" else path.with_suffix("")

        if not load_dir.exists():
            raise ValidationError("path", f"Compressor directory not found: {load_dir}")

        config_path = load_dir / "config.json"
        state_path = load_dir / "state.json"
        data_path = load_dir / "data.npz"

        if not config_path.exists() or not state_path.exists():
            raise ValidationError(
                "path",
                f"Invalid compressor format: missing config.json or state.json in {load_dir}"
            )

        # Load config
        with open(config_path, "r") as f:
            config_data = json.load(f)

        config = VectorCompressionConfig(
            method=CompressionMethod(config_data["method"]),
            n_subvectors=config_data.get("n_subvectors", 256),
            n_pq_bits=config_data.get("n_pq_bits", 8),
            scalar_bits=config_data.get("scalar_bits", 8),
            binary_threshold=config_data.get("binary_threshold", 0.0),
        )

        compressor = cls(config)

        # Load state
        with open(state_path, "r") as f:
            state = json.load(f)

        # Load arrays if present
        arrays = {}
        if data_path.exists():
            arrays = dict(np.load(data_path))

        if state["type"] == "scalar":
            q = ScalarQuantizer(bits=state["bits"])
            q.min = arrays["min"]
            q.max = arrays["max"]
            q.scale = arrays["scale"]
            q.dimension = state["dimension"]
            q.is_fitted = True
            compressor._quantizer = q
        elif state["type"] == "pq":
            q = ProductQuantization(
                dimension=state["dimension"],
                n_subvectors=state["n_subvectors"],
                n_bits=state["n_bits"],
            )
            q.codebooks = arrays["codebooks"]
            q.is_fitted = True
            compressor._quantizer = q
        elif state["type"] == "binary":
            q = BinaryQuantizer(threshold=state["threshold"])
            q.dimension = state["dimension"]
            q.is_fitted = True
            compressor._quantizer = q

        logger.info(f"Loaded compressor from {load_dir}")
        return compressor


# =============================================================================
# Convenience Functions
# =============================================================================


def create_compressor(
    method: CompressionMethod = CompressionMethod.SCALAR_INT8,
    dimension: Optional[int] = None,
    **config_kwargs,
) -> VectorCompressor:
    """
    Create a vector compressor.

    Args:
        method: Compression method
        dimension: Vector dimension (required for PQ)
        **config_kwargs: Additional configuration

    Returns:
        VectorCompressor instance
    """
    config = VectorCompressionConfig(method=method, **config_kwargs)

    if method == CompressionMethod.PRODUCT_QUANTIZATION and dimension is not None:
        # For PQ, we need to initialize with dimension
        compressor = VectorCompressor(config)
        # Create temporary PQ to validate dimension
        if "n_subvectors" in config_kwargs:
            n_sub = config_kwargs["n_subvectors"]
            if dimension % n_sub != 0:
                raise ValidationError(
                    "dimension",
                    f"Dimension {dimension} must be divisible by n_subvectors {n_sub}",
                )
        return compressor

    return VectorCompressor(config)


def get_compression_ratio(
    method: CompressionMethod,
    dimension: int,
    **config_kwargs,
) -> float:
    """
    Calculate theoretical compression ratio.

    Args:
        method: Compression method
        dimension: Original vector dimension
        **config_kwargs: Method-specific config

    Returns:
        Compression ratio (original_size / compressed_size)
    """
    original_bytes = dimension * 4  # float32

    if method == CompressionMethod.SCALAR_INT8:
        compressed_bytes = dimension * 1  # int8
    elif method == CompressionMethod.PRODUCT_QUANTIZATION:
        n_subvectors = config_kwargs.get("n_subvectors", 256)
        compressed_bytes = n_subvectors * 1  # uint8 per subvector
    elif method == CompressionMethod.BINARY:
        compressed_bytes = (dimension + 7) // 8  # packed bits
    else:
        compressed_bytes = original_bytes

    return original_bytes / compressed_bytes


# Type aliases for backward compatibility
ScalarQuantizer8 = ScalarQuantizer
PQCodec = ProductQuantization
BinaryVectorQuantizer = BinaryQuantizer
