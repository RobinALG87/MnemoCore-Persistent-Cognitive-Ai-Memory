"""
HAIM Test Suite — Binary HDV Tests
===================================
Tests for the core BinaryHDV operations (Phase 3.0).
Validates mathematical properties of VSA operations.
"""

import numpy as np
import pytest

from mnemocore.core.binary_hdv import (
    BinaryHDV,
    TextEncoder,
    batch_hamming_distance,
    majority_bundle,
    top_k_nearest,
)

# Default test dimension (smaller for speed)
D = 1024


class TestBinaryHDVConstruction:
    def test_random_creates_valid_vector(self):
        v = BinaryHDV.random(D)
        assert v.dimension == D
        assert v.data.shape == (D // 8,)
        assert v.data.dtype == np.uint8

    def test_zeros(self):
        v = BinaryHDV.zeros(D)
        assert np.all(v.data == 0)

    def test_ones(self):
        v = BinaryHDV.ones(D)
        assert np.all(v.data == 0xFF)

    def test_from_seed_deterministic(self):
        v1 = BinaryHDV.from_seed("hello", D)
        v2 = BinaryHDV.from_seed("hello", D)
        assert v1 == v2

    def test_different_seeds_different_vectors(self):
        v1 = BinaryHDV.from_seed("hello", D)
        v2 = BinaryHDV.from_seed("world", D)
        assert v1 != v2

    def test_dimension_must_be_multiple_of_8(self):
        with pytest.raises(AssertionError):
            BinaryHDV.random(100)

    def test_serialization_roundtrip(self):
        v = BinaryHDV.random(D)
        raw = v.to_bytes()
        assert len(raw) == D // 8
        v2 = BinaryHDV.from_bytes(raw, D)
        assert v == v2


class TestXORBinding:
    def test_self_inverse(self):
        """a ⊕ a = 0 (zero vector)."""
        a = BinaryHDV.random(D)
        result = a.xor_bind(a)
        assert result == BinaryHDV.zeros(D)

    def test_commutative(self):
        """a ⊕ b = b ⊕ a."""
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        assert a.xor_bind(b) == b.xor_bind(a)

    def test_associative(self):
        """(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)."""
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        c = BinaryHDV.random(D)
        lhs = a.xor_bind(b).xor_bind(c)
        rhs = a.xor_bind(b.xor_bind(c))
        assert lhs == rhs

    def test_xor_with_zeros_is_identity(self):
        """a ⊕ 0 = a."""
        a = BinaryHDV.random(D)
        z = BinaryHDV.zeros(D)
        assert a.xor_bind(z) == a

    def test_unbinding(self):
        """If c = a ⊕ b, then a = c ⊕ b (self-inverse property enables unbinding)."""
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        c = a.xor_bind(b)
        recovered_a = c.xor_bind(b)
        assert recovered_a == a

    def test_binding_preserves_distance(self):
        """hamming(a⊕c, b⊕c) = hamming(a, b)."""
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        c = BinaryHDV.random(D)
        dist_ab = a.hamming_distance(b)
        dist_ac_bc = a.xor_bind(c).hamming_distance(b.xor_bind(c))
        assert dist_ab == dist_ac_bc


class TestHammingDistance:
    def test_self_distance_is_zero(self):
        a = BinaryHDV.random(D)
        assert a.hamming_distance(a) == 0

    def test_inverse_is_max_distance(self):
        """hamming(a, ~a) = dimension."""
        a = BinaryHDV.random(D)
        assert a.hamming_distance(a.invert()) == D

    def test_symmetry(self):
        """hamming(a, b) = hamming(b, a)."""
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        assert a.hamming_distance(b) == b.hamming_distance(a)

    def test_triangle_inequality(self):
        """hamming(a, c) <= hamming(a, b) + hamming(b, c)."""
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        c = BinaryHDV.random(D)
        assert a.hamming_distance(c) <= a.hamming_distance(b) + b.hamming_distance(c)

    def test_random_vectors_near_half_dimension(self):
        """Random vectors should have Hamming distance ≈ D/2."""
        np.random.seed(42)
        distances = []
        for _ in range(50):
            a = BinaryHDV.random(D)
            b = BinaryHDV.random(D)
            distances.append(a.hamming_distance(b))
        mean_dist = np.mean(distances)
        # Should be close to D/2 = 512 for D=1024
        assert abs(mean_dist - D / 2) < D * 0.05  # Within 5% of expected

    def test_similarity_score_range(self):
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        sim = a.similarity(b)
        assert 0.0 <= sim <= 1.0

    def test_normalized_distance_range(self):
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        nd = a.normalized_distance(b)
        assert 0.0 <= nd <= 1.0


class TestPermutation:
    def test_permute_zero_is_identity(self):
        a = BinaryHDV.random(D)
        assert a.permute(0) == a

    def test_permute_full_cycle(self):
        """Permuting by D should return the original vector."""
        a = BinaryHDV.random(D)
        assert a.permute(D) == a

    def test_permute_produces_different_vector(self):
        """Non-zero permutation should produce a (very likely) different vector."""
        a = BinaryHDV.random(D)
        b = a.permute(1)
        assert a != b

    def test_permute_is_invertible(self):
        """permute(k) followed by permute(-k) recovers original."""
        a = BinaryHDV.random(D)
        b = a.permute(7).permute(-7)
        assert a == b


class TestMajorityBundle:
    def test_single_vector_bundle(self):
        """Bundling a single vector returns that vector."""
        a = BinaryHDV.random(D)
        result = majority_bundle([a])
        assert result == a

    def test_bundled_vector_similar_to_inputs(self):
        """Bundle of {a, b, c} should be more similar to each input than random."""
        np.random.seed(42)
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        c = BinaryHDV.random(D)
        bundled = majority_bundle([a, b, c])

        # Each input should be closer to the bundle than to a random vector
        random_v = BinaryHDV.random(D)
        for v in [a, b, c]:
            sim_to_bundle = bundled.similarity(v)
            sim_to_random = bundled.similarity(random_v)
            assert sim_to_bundle > sim_to_random, (
                f"Bundle should be more similar to its inputs than to random vectors. "
                f"sim_to_bundle={sim_to_bundle:.3f}, sim_to_random={sim_to_random:.3f}"
            )

    def test_bundle_is_approximate(self):
        """Bundle is not exact — it's a lossy superposition."""
        a = BinaryHDV.random(D)
        b = BinaryHDV.random(D)
        bundled = majority_bundle([a, b])
        # Bundled vector should be similar but not identical to either input
        assert bundled != a
        assert bundled != b
        assert bundled.similarity(a) > 0.5
        assert bundled.similarity(b) > 0.5

    def test_empty_bundle_raises(self):
        with pytest.raises(AssertionError):
            majority_bundle([])


class TestBatchOperations:
    def test_batch_hamming_distance(self):
        """Batch Hamming should match individual computations."""
        np.random.seed(42)
        query = BinaryHDV.random(D)
        n = 100
        db = np.stack([BinaryHDV.random(D).data for _ in range(n)], axis=0)

        batch_distances = batch_hamming_distance(query, db)
        assert batch_distances.shape == (n,)

        # Verify against individual computations
        for i in range(n):
            individual = query.hamming_distance(BinaryHDV(data=db[i], dimension=D))
            assert batch_distances[i] == individual

    def test_top_k_nearest(self):
        """Top-K should return the K closest vectors."""
        np.random.seed(42)
        query = BinaryHDV.random(D)
        n = 50
        db_vectors = [BinaryHDV.random(D) for _ in range(n)]
        db = np.stack([v.data for v in db_vectors], axis=0)

        # Make one vector very close to the query
        close_vector = query.data.copy()
        # Flip just a few bits
        close_vector[0] ^= 0x03  # Flip 2 bits
        db[0] = close_vector

        results = top_k_nearest(query, db, k=5)
        assert len(results) == 5
        # First result should be index 0 (the close vector)
        assert results[0][0] == 0
        # Distances should be sorted ascending
        for i in range(len(results) - 1):
            assert results[i][1] <= results[i + 1][1]


class TestTextEncoder:
    def test_encode_deterministic(self):
        enc = TextEncoder(dimension=D)
        v1 = enc.encode("hello world")
        v2 = enc.encode("hello world")
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        enc = TextEncoder(dimension=D)
        v1 = enc.encode("hello world")
        v2 = enc.encode("goodbye moon")
        assert v1 != v2

    def test_similar_texts_more_similar(self):
        """Texts sharing words should be more similar than completely different texts."""
        np.random.seed(42)
        enc = TextEncoder(dimension=D)
        v_base = enc.encode("the quick brown fox")
        v_similar = enc.encode("the quick brown dog")
        v_different = enc.encode("quantum computing research paper")

        sim_similar = v_base.similarity(v_similar)
        sim_different = v_base.similarity(v_different)
        assert sim_similar > sim_different, (
            f"Similar text should have higher similarity. "
            f"sim_similar={sim_similar:.3f}, sim_different={sim_different:.3f}"
        )

    def test_encode_with_context(self):
        enc = TextEncoder(dimension=D)
        context = BinaryHDV.random(D)
        v = enc.encode_with_context("hello world", context)
        # Should be different from encoding without context
        v_no_ctx = enc.encode("hello world")
        assert v != v_no_ctx
        # XOR with context should recover the content encoding
        recovered = v.xor_bind(context)
        assert recovered == v_no_ctx

    def test_empty_text(self):
        """Empty text should still produce a valid vector."""
        enc = TextEncoder(dimension=D)
        v = enc.encode("")
        assert v.dimension == D
        assert v.data.shape == (D // 8,)

    def test_token_caching(self):
        enc = TextEncoder(dimension=D)
        enc.encode("hello world")
        assert "hello" in enc._token_cache
        assert "world" in enc._token_cache


class TestFullDimension:
    """Tests at full 16,384 dimensions to verify scaling."""

    def test_full_dim_roundtrip(self):
        v = BinaryHDV.random(16384)
        assert v.data.shape == (2048,)  # 16384 / 8
        raw = v.to_bytes()
        assert len(raw) == 2048
        v2 = BinaryHDV.from_bytes(raw, 16384)
        assert v == v2

    def test_full_dim_hamming(self):
        a = BinaryHDV.random(16384)
        b = BinaryHDV.random(16384)
        dist = a.hamming_distance(b)
        # Should be roughly D/2 = 8192
        assert 6000 < dist < 10000

    def test_full_dim_batch_search(self):
        np.random.seed(42)
        query = BinaryHDV.random(16384)
        n = 1000
        db = np.stack([BinaryHDV.random(16384).data for _ in range(n)], axis=0)
        results = top_k_nearest(query, db, k=10)
        assert len(results) == 10
        # Verify sorted
        for i in range(len(results) - 1):
            assert results[i][1] <= results[i + 1][1]
