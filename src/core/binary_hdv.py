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
from typing import List, Optional, Tuple

import numpy as np


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
        Uses SHA-256 iterative expansion to fill the vector.
        """
        n_bytes = dimension // 8
        result = bytearray()
        counter = 0
        while len(result) < n_bytes:
            h = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
            result.extend(h)
            counter += 1
        data = np.frombuffer(bytes(result[:n_bytes]), dtype=np.uint8).copy()
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

        # Convert to unpacked bits, roll, repack
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

        Uses np.unpackbits + sum for correctness.
        Range: [0, dimension].
        """
        assert self.dimension == other.dimension
        xor_result = np.bitwise_xor(self.data, other.data)
        return int(np.unpackbits(xor_result).sum())

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

    def to_bytes(self) -> bytes:
        """Serialize to raw bytes (for storage)."""
        return self.data.tobytes()

    @classmethod
    def from_bytes(cls, raw: bytes, dimension: int = 16384) -> "BinaryHDV":
        """Deserialize from raw bytes."""
        data = np.frombuffer(raw, dtype=np.uint8).copy()
        return cls(data=data, dimension=dimension)

    def __repr__(self) -> str:
        popcount = int(np.unpackbits(self.data).sum())
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

    Args:
        query: Single BinaryHDV query vector.
        database: 2D array of shape (N, D//8) with dtype uint8, where each row
                  is a packed binary vector.

    Returns:
        1D array of shape (N,) with Hamming distances (int).
    """
    # XOR query with all database vectors: (N, D//8)
    xor_result = np.bitwise_xor(database, query.data)

    # Popcount via lookup table — count bits set in each byte
    # This is the fastest pure-NumPy approach for packed binary vectors
    popcount_table = _build_popcount_table()
    bit_counts = popcount_table[xor_result]  # (N, D//8)

    # Sum across bytes to get total Hamming distance per vector
    return bit_counts.sum(axis=1)


def batch_hamming_distance_matrix(
    database: np.ndarray,
) -> np.ndarray:
    """
    Compute the full pairwise Hamming distance matrix for a database.

    Args:
        database: 2D array of shape (N, D//8) with dtype uint8.

    Returns:
        2D array of shape (N, N) with Hamming distances.
    """
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
    all_bits = np.stack([np.unpackbits(v.data) for v in vectors], axis=0)  # (K, D)

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

    def get_token_vector(self, token: str) -> BinaryHDV:
        """Get or create a deterministic vector for a token."""
        if token not in self._token_cache:
            self._token_cache[token] = BinaryHDV.from_seed(token, self.dimension)
        return self._token_cache[token]

    def encode(self, text: str) -> BinaryHDV:
        """
        Encode a text string to a binary HDV.

        Tokenization: simple whitespace split + lowercasing.
        Each token is bound with its position via XOR(token, permute(position_marker, i)).
        All position-bound tokens are bundled via majority vote.
        """
        tokens = text.lower().split()
        if not tokens:
            return BinaryHDV.random(self.dimension)

        if len(tokens) == 1:
            return self.get_token_vector(tokens[0])

        # Build position-bound token vectors
        bound_vectors = []
        for i, token in enumerate(tokens):
            token_hdv = self.get_token_vector(token)
            # Permute by position index for order encoding
            positioned = token_hdv.permute(shift=i)
            bound_vectors.append(positioned)

        return majority_bundle(bound_vectors)

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


# ======================================================================
# Internal helpers
# ======================================================================

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
