"""
Binary Hyperdimensional Vector (Binary HDV) Core
=================================================
Phase 3.0 implementation of binary VSA operations.

Based on Kanerva's Hyperdimensional Computing theory (2009).
Uses standard mathematical operations (XOR, Hamming distance, majority bundling)
that are fundamental VSA primitives — not derived from any proprietary implementation.

Key design choices:
  - D = 16,384 bits (2^14) — configurable via config.yaml
  - Storage: packed as np.uint8 arrays (D/8 bytes = 2,048 bytes per vector)
  - Similarity: Hamming distance (popcount of XOR result)
  - Binding: element-wise XOR (self-inverse, commutative)
  - Bundling: element-wise majority vote (thresholded sum)
  - Sequence: circular bit-shift (permutation)

All batch operations are NumPy-vectorized (no Python loops for distance computation).
"""

import hashlib
import sqlite3
from typing import List, Optional, Tuple

import numpy as np
import re
from pathlib import Path
from loguru import logger
from .config import get_config

# ------------------------------------------------------------------
# Hardware Acceleration Detection (Tier 1: PyTorch, Tier 2: Numba)
# ------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        TORCH_DEVICE = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        TORCH_DEVICE = torch.device('mps')
    else:
        TORCH_DEVICE = torch.device('cpu')
    _TORCH_POPCOUNT_TABLE: Optional[torch.Tensor] = None
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Cached lookup table for popcount (bits set per byte value 0-255)
_POPCOUNT_TABLE: Optional[np.ndarray] = None


def _build_popcount_table() -> np.ndarray:
    """Build or return cached popcount lookup table for bytes (0-255)."""
    global _POPCOUNT_TABLE
    if _POPCOUNT_TABLE is None:
        _POPCOUNT_TABLE = np.array(
            [bin(i).count("1") for i in range(256)], dtype=np.int32
        )
    return _POPCOUNT_TABLE

def _get_torch_popcount_table():
    global _TORCH_POPCOUNT_TABLE
    if _TORCH_POPCOUNT_TABLE is None and TORCH_AVAILABLE:
        _TORCH_POPCOUNT_TABLE = torch.tensor(
            [bin(i).count("1") for i in range(256)], 
            dtype=torch.int32, 
            device=TORCH_DEVICE
        )
    return _TORCH_POPCOUNT_TABLE

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _numba_batch_hamming(query_data, database, popcount_table):
        N = database.shape[0]
        distances = np.zeros(N, dtype=np.int32)
        for i in prange(N):
            dist = 0
            for j in range(database.shape[1]):
                dist += popcount_table[query_data[j] ^ database[i, j]]
            distances[i] = dist
        return distances
    
    @njit(parallel=True, fastmath=True)
    def _numba_batch_hamming_matrix(database, popcount_table):
        N = database.shape[0]
        distances = np.zeros((N, N), dtype=np.int32)
        for i in prange(N):
            for j in range(i + 1, N):
                dist = 0
                for k in range(database.shape[1]):
                    dist += popcount_table[database[i, k] ^ database[j, k]]
                distances[i, j] = dist
                distances[j, i] = dist
        return distances


class BinaryHDV:
    """
    A binary hyperdimensional vector stored as a packed uint8 array.

    The vector has `dimension` logical bits, stored in `dimension // 8` bytes.
    Each byte holds 8 bits in big-endian bit order (MSB first within each byte).

    Attributes:
        data: np.ndarray of dtype uint8, shape (dimension // 8,)
        dimension: int, number of logical bits
    """

    __slots__ = ("data", "dimension")

    def __init__(self, data: np.ndarray, dimension: int):
        """
        Args:
            data: Packed uint8 array of shape (dimension // 8,).
            dimension: Number of logical bits.
        """
        assert data.dtype == np.uint8, f"Expected uint8, got {data.dtype}"
        assert data.shape == (dimension // 8,), (
            f"Shape mismatch: expected ({dimension // 8},), got {data.shape}"
        )
        self.data = data
        self.dimension = dimension

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def random(cls, dimension: int = 16384) -> "BinaryHDV":
        """Generate a random binary vector (uniform i.i.d. bits)."""
        assert dimension % 8 == 0, "Dimension must be multiple of 8"
        n_bytes = dimension // 8
        data = np.random.randint(0, 256, size=n_bytes, dtype=np.uint8)
        return cls(data=data, dimension=dimension)

    @classmethod
    def zeros(cls, dimension: int = 16384) -> "BinaryHDV":
        """All-zero vector."""
        n_bytes = dimension // 8
        return cls(data=np.zeros(n_bytes, dtype=np.uint8), dimension=dimension)

    @classmethod
    def ones(cls, dimension: int = 16384) -> "BinaryHDV":
        """All-one vector (every bit set)."""
        n_bytes = dimension // 8
        return cls(
            data=np.full(n_bytes, 0xFF, dtype=np.uint8), dimension=dimension
        )

    @classmethod
    def from_seed(cls, seed: str, dimension: int = 16384) -> "BinaryHDV":
        """
        Deterministic vector from a string seed.
        Uses SHA-3 (SHAKE-256) for high-performance deterministic expansion.
        """
        n_bytes = dimension // 8
        # SHAKE-256 can generate arbitrary length digests in one pass
        digest = hashlib.shake_256(seed.encode()).digest(n_bytes)
        data = np.frombuffer(digest, dtype=np.uint8).copy()
        return cls(data=data, dimension=dimension)

    # ------------------------------------------------------------------
    # Core VSA operations
    # ------------------------------------------------------------------

    def xor_bind(self, other: "BinaryHDV") -> "BinaryHDV":
        """
        Binding via element-wise XOR.

        Properties:
          - Self-inverse: a ⊕ a = 0
          - Commutative: a ⊕ b = b ⊕ a
          - Associative: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
          - Preserves distance: hamming(a⊕c, b⊕c) = hamming(a, b)
        """
        assert self.dimension == other.dimension
        return BinaryHDV(
            data=np.bitwise_xor(self.data, other.data),
            dimension=self.dimension,
        )

    def permute(self, shift: int = 1) -> "BinaryHDV":
        """
        Circular bit-shift for sequence/role encoding.

        Shifts all bits by `shift` positions to the right (with wrap-around).
        Works at the byte level with bit carry for efficiency.
        """
        if shift == 0:
            return BinaryHDV(data=self.data.copy(), dimension=self.dimension)

        # Normalize shift to positive value within dimension
        shift = shift % self.dimension

        bits = np.unpackbits(self.data)
        bits = np.roll(bits, shift)
        return BinaryHDV(
            data=np.packbits(bits), dimension=self.dimension
        )

    def invert(self) -> "BinaryHDV":
        """Bitwise NOT — produces the maximally distant vector."""
        return BinaryHDV(
            data=np.bitwise_not(self.data), dimension=self.dimension
        )

    def hamming_distance(self, other: "BinaryHDV") -> int:
        """
        Hamming distance: count of differing bits.

        Uses lookup table for speed (replacing unpackbits).
        Range: [0, dimension].
        """
        assert self.dimension == other.dimension
        xor_result = np.bitwise_xor(self.data, other.data)
        # Optimized: use precomputed popcount table instead of unpacking bits
        return int(_build_popcount_table()[xor_result].sum())

    def normalized_distance(self, other: "BinaryHDV") -> float:
        """Hamming distance normalized to [0.0, 1.0]."""
        return self.hamming_distance(other) / self.dimension

    def similarity(self, other: "BinaryHDV") -> float:
        """
        Similarity score in [0.0, 1.0].
        1.0 = identical, 0.0 = maximally different.
        0.5 = random/orthogonal (expected for unrelated vectors).
        """
        return 1.0 - self.normalized_distance(other)

    # ------------------------------------------------------------------
    # Compatibility shims for legacy HDV API
    # ------------------------------------------------------------------

    def bind(self, other: "BinaryHDV") -> "BinaryHDV":
        """
        Alias for xor_bind(). Compatibility shim for legacy HDV API.

        Deprecated: Use xor_bind() directly for new code.
        """
        return self.xor_bind(other)

    def unbind(self, other: "BinaryHDV") -> "BinaryHDV":
        """
        Alias for xor_bind(). Since XOR is self-inverse, unbind = bind.

        Compatibility shim for legacy HDV API.
        """
        return self.xor_bind(other)

    def cosine_similarity(self, other: "BinaryHDV") -> float:
        """
        Alias for similarity(). Compatibility shim for legacy HDV API.

        Note: For binary vectors, this returns Hamming-based similarity,
        not true cosine similarity. The values are comparable for most use cases.
        """
        return self.similarity(other)

    def normalize(self) -> "BinaryHDV":
        """
        No-op for binary vectors. Compatibility shim for legacy HDV API.

        Binary vectors are already "normalized" in the sense that they
        consist only of 0s and 1s. Returns a copy of the vector.
        """
        return BinaryHDV(data=self.data.copy(), dimension=self.dimension)

    def __xor__(self, other: "BinaryHDV") -> "BinaryHDV":
        """Alias for xor_bind(). Enables v1 ^ v2 syntax."""
        return self.xor_bind(other)

    def to_bytes(self) -> bytes:
        """Serialize to raw bytes (for storage)."""
        return self.data.tobytes()

    @classmethod
    def from_bytes(cls, raw: bytes, dimension: int = 16384) -> "BinaryHDV":
        """Deserialize from raw bytes."""
        data = np.frombuffer(raw, dtype=np.uint8).copy()
        return cls(data=data, dimension=dimension)

    def __repr__(self) -> str:
        # Optimized: use precomputed popcount table
        popcount = int(_build_popcount_table()[self.data].sum())
        return f"BinaryHDV(dim={self.dimension}, popcount={popcount}/{self.dimension})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryHDV):
            return NotImplemented
        return self.dimension == other.dimension and np.array_equal(
            self.data, other.data
        )


# ======================================================================
# Batch operations (NumPy-vectorized, no Python loops)
# ======================================================================


def batch_hamming_distance(
    query: BinaryHDV, database: np.ndarray
) -> np.ndarray:
    """
    Compute Hamming distance between a query vector and all vectors in a database.
    Auto-scales from GPU (PyTorch) -> CPU JIT (Numba) -> CPU (NumPy).
    """
    # Tier 1: PyTorch (GPU)
    if TORCH_AVAILABLE and TORCH_DEVICE.type != 'cpu':
        try:
            q_t = torch.from_numpy(query.data).to(TORCH_DEVICE, non_blocking=True)
            db_t = torch.from_numpy(database).to(TORCH_DEVICE, non_blocking=True)
            xor_res = torch.bitwise_xor(db_t, q_t).to(torch.long)
            bit_counts = _get_torch_popcount_table()[xor_res]
            return bit_counts.sum(dim=1).cpu().numpy()
        except Exception as e:
            logger.debug(f"Torch batch_hamming failed, falling back: {e}")

    # Tier 2: Numba (JIT CPU)
    if NUMBA_AVAILABLE:
        try:
            return _numba_batch_hamming(query.data, database, _build_popcount_table())
        except Exception as e:
            logger.debug(f"Numba batch_hamming failed, falling back: {e}")

    # Tier 3: NumPy (Pure CPU Fallback)
    xor_result = np.bitwise_xor(database, query.data)
    popcount_table = _build_popcount_table()
    bit_counts = popcount_table[xor_result]  # (N, D//8)
    return bit_counts.sum(axis=1)


def batch_hamming_distance_matrix(
    database: np.ndarray,
) -> np.ndarray:
    """
    Compute the full pairwise Hamming distance matrix for a database.
    Auto-scales from GPU (PyTorch) -> CPU JIT (Numba) -> CPU (NumPy).
    """
    # Tier 1: PyTorch (GPU)
    if TORCH_AVAILABLE and TORCH_DEVICE.type != 'cpu':
        try:
            db_t = torch.from_numpy(database).to(TORCH_DEVICE, non_blocking=True)
            N = database.shape[0]
            # Expanding for pairwise XOR (can be memory intensive for huge N)
            # Use chunks if N is very large, but for now standard tensor ops:
            if N < 5000:
                xor_res = torch.bitwise_xor(db_t.unsqueeze(1), db_t.unsqueeze(0)).to(torch.long)
                bit_counts = _get_torch_popcount_table()[xor_res]
                return bit_counts.sum(dim=2).cpu().numpy()
        except Exception as e:
            logger.debug(f"Torch batch matrix failed, falling back: {e}")

    # Tier 2: Numba (JIT CPU)
    if NUMBA_AVAILABLE:
        try:
            return _numba_batch_hamming_matrix(database, _build_popcount_table())
        except Exception as e:
            logger.debug(f"Numba batch matrix failed, falling back: {e}")

    # Tier 3: NumPy (Pure CPU Fallback)
    N = database.shape[0]
    popcount_table = _build_popcount_table()
    distances = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        xor_result = np.bitwise_xor(database[i], database[i + 1 :])
        bit_counts = popcount_table[xor_result].sum(axis=1)
        distances[i, i + 1 :] = bit_counts
        distances[i + 1 :, i] = bit_counts
    return distances


def majority_bundle(
    vectors: List[BinaryHDV], randomize_ties: bool = False
) -> BinaryHDV:
    """
    Bundle multiple vectors via element-wise majority vote.

    For each bit position, the result bit is 1 if more than half of the
    input vectors have a 1 at that position.

    Args:
        vectors: List of BinaryHDV vectors to bundle.
        randomize_ties: If True, break ties randomly. If False (default),
                        ties default to 0 for deterministic results.

    This is the standard VSA bundling operation (superposition).
    """
    assert len(vectors) > 0, "Cannot bundle empty list"
    dimension = vectors[0].dimension

    # Unpack all vectors to bits
    # Optimization: Stack packed data first, then unpack all at once
    # This avoids K calls to unpackbits and list comprehension overhead
    packed_data = np.stack([v.data for v in vectors], axis=0)  # (K, D//8)
    all_bits = np.unpackbits(packed_data, axis=1)  # (K, D)

    # Sum along vectors axis: count of 1-bits per position
    sums = all_bits.sum(axis=0)  # (D,)

    # Majority vote: > half means 1
    threshold = len(vectors) / 2.0

    result_bits = np.zeros(dimension, dtype=np.uint8)
    result_bits[sums > threshold] = 1

    # Handle ties
    if randomize_ties:
        ties = sums == threshold
        if ties.any():
            result_bits[ties] = np.random.randint(
                0, 2, size=ties.sum(), dtype=np.uint8
            )

    return BinaryHDV(data=np.packbits(result_bits), dimension=dimension)


def top_k_nearest(
    query: BinaryHDV, database: np.ndarray, k: int = 10
) -> List[Tuple[int, int]]:
    """
    Find k nearest neighbors by Hamming distance.

    Args:
        query: Query vector.
        database: 2D array of shape (N, D//8) packed binary vectors.
        k: Number of nearest neighbors.

    Returns:
        List of (index, distance) tuples, sorted by distance ascending.
    """
    distances = batch_hamming_distance(query, database)
    k = min(k, len(distances))

    # argpartition is O(N) vs O(N log N) for full sort — much faster for large N
    indices = np.argpartition(distances, k)[:k]
    selected_distances = distances[indices]

    # Sort the k results by distance
    sort_order = np.argsort(selected_distances)
    sorted_indices = indices[sort_order]
    sorted_distances = selected_distances[sort_order]

    return [(int(idx), int(dist)) for idx, dist in zip(sorted_indices, sorted_distances)]


# ======================================================================
# Persistent Vector Cache (SQLite)
# ======================================================================


class PersistentVectorCache:
    """
    SQLite-backed persistent cache for BinaryHDV vectors.
    Reduces redundant CPU-intensive encoding operations.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Ensure the vectors table exists."""
        try:
            # Create directories if they don't exist
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS vectors "
                    "(key TEXT PRIMARY KEY, vector BLOB, dimension INTEGER)"
                )
                conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
                conn.execute("PRAGMA synchronous=NORMAL")
        except Exception as e:
            logger.warning(f"Failed to initialize PersistentVectorCache at {self.db_path}: {e}")

    def get(self, key: str, dimension: int) -> Optional[BinaryHDV]:
        """Retrieve a vector from the cache."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                res = conn.execute(
                    "SELECT vector FROM vectors WHERE key = ? AND dimension = ?",
                    (key, dimension)
                ).fetchone()
                if res:
                    return BinaryHDV.from_bytes(res[0], dimension=dimension)
        except Exception as e:
            logger.debug(f"Cache get failed for {key}: {e}")
        return None

    def set(self, key: str, vector: BinaryHDV):
        """Store a vector in the cache."""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO vectors (key, vector, dimension) VALUES (?, ?, ?)",
                    (key, vector.to_bytes(), vector.dimension)
                )
        except Exception as e:
            logger.debug(f"Cache set failed for {key}: {e}")


# ======================================================================
# Text encoding pipeline
# ======================================================================


class TextEncoder:
    """
    Encode text to binary HDV using token-level random vectors with
    position-permutation binding.

    Method: For text "hello world", we compute:
        HDV = bundle(token("hello") ⊕ permute(pos, 0),
                     token("world") ⊕ permute(pos, 1))

    Token vectors are deterministic (seeded from the token string),
    ensuring the same word always maps to the same base vector.
    """

    def __init__(self, dimension: int = 16384):
        self.dimension = dimension
        self._token_cache: dict[str, BinaryHDV] = {}
        
        # Phase 4.6: Persistent Caching
        config = get_config()
        self.perf_config = getattr(config, 'performance', None)
        self.persistent_cache = None
        
        if self.perf_config and self.perf_config.vector_cache_enabled:
            cache_path = self.perf_config.vector_cache_path or "./data/vector_cache.sqlite"
            self.persistent_cache = PersistentVectorCache(cache_path)
            logger.info(f"Persistent vector cache enabled at {cache_path}")

    def get_token_vector(self, token: str) -> BinaryHDV:
        """Get or create a deterministic vector for a token."""
        if token in self._token_cache:
            return self._token_cache[token]
            
        # Try persistent cache first
        if self.persistent_cache:
            cached = self.persistent_cache.get(f"token:{token}", self.dimension)
            if cached:
                self._token_cache[token] = cached
                return cached
        
        # Generate and cache
        vec = BinaryHDV.from_seed(token, self.dimension)
        self._token_cache[token] = vec
        
        if self.persistent_cache:
            self.persistent_cache.set(f"token:{token}", vec)
            
        return vec

    def encode(self, text: str) -> BinaryHDV:
        """
        Encode a text string to a binary HDV.

        Tokenization: simple whitespace split after normalization.
        Each token is bound with its position via XOR(token, permute(position_marker, i)).
        All position-bound tokens are bundled via majority vote.
        """
        # Try full-text cache first
        if self.persistent_cache:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cached = self.persistent_cache.get(f"text:{text_hash}", self.dimension)
            if cached:
                return cached

        # Improved Tokenization: consistent alphanumeric extraction
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens:
            return BinaryHDV.random(self.dimension)

        if len(tokens) == 1:
            vec = self.get_token_vector(tokens[0])
            # Cache the single-token result too
            if self.persistent_cache:
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                self.persistent_cache.set(f"text:{text_hash}", vec)
            return vec

        # Build position-bound token vectors (#27)
        # Optimized: Batch process data instead of multiple object instantiations
        token_hdvs = [self.get_token_vector(t) for t in tokens]
        packed_data = np.stack([v.data for v in token_hdvs], axis=0)
        all_bits = np.unpackbits(packed_data, axis=1)

        # Apply position-based permutations (roll)
        for i in range(len(tokens)):
            if i > 0:
                all_bits[i] = np.roll(all_bits[i], i)

        # Vectorized majority vote (equivalent to majority_bundle)
        sums = all_bits.sum(axis=0)
        threshold = len(tokens) / 2.0
        result_bits = np.zeros(self.dimension, dtype=np.uint8)
        result_bits[sums > threshold] = 1

        result_vec = BinaryHDV(data=np.packbits(result_bits), dimension=self.dimension)
        
        # Cache the result
        if self.persistent_cache:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            self.persistent_cache.set(f"text:{text_hash}", result_vec)
            
        return result_vec

    def encode_with_context(
        self, text: str, context_hdv: BinaryHDV
    ) -> BinaryHDV:
        """
        Encode text and bind it with a context vector.

        Result = encode(text) ⊕ context
        This creates an association between the content and its context.
        """
        content_hdv = self.encode(text)
        return content_hdv.xor_bind(context_hdv)
