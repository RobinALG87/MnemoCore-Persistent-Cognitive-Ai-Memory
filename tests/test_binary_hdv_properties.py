"""
Hypothesis Property-Based Tests for BinaryHDV
==============================================
Mathematical correctness tests using property-based testing.

These tests validate the algebraic properties of BinaryHDV operations
using Hypothesis for automatic test case generation.
"""

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck, assume
import hypothesis.strategies as st

from src.core.binary_hdv import BinaryHDV, majority_bundle


# Use smaller dimension for faster property tests
TEST_DIMENSION = 512


# Custom Hypothesis strategies for BinaryHDV
@st.composite
def binary_hdv_strategy(draw, dimension: int = TEST_DIMENSION):
    """Generate a random BinaryHDV vector."""
    # Generate random bytes
    n_bytes = dimension // 8
    byte_list = [draw(st.integers(min_value=0, max_value=255)) for _ in range(n_bytes)]
    data = np.array(byte_list, dtype=np.uint8)
    return BinaryHDV(data=data, dimension=dimension)


@st.composite
def binary_hdv_pair_strategy(draw, dimension: int = TEST_DIMENSION):
    """Generate a pair of BinaryHDV vectors with the same dimension."""
    v1 = draw(binary_hdv_strategy(dimension))
    v2 = draw(binary_hdv_strategy(dimension))
    return (v1, v2)


@st.composite
def binary_hdv_triple_strategy(draw, dimension: int = TEST_DIMENSION):
    """Generate a triple of BinaryHDV vectors with the same dimension."""
    v1 = draw(binary_hdv_strategy(dimension))
    v2 = draw(binary_hdv_strategy(dimension))
    v3 = draw(binary_hdv_strategy(dimension))
    return (v1, v2, v3)


@st.composite
def shift_strategy(draw, dimension: int = TEST_DIMENSION):
    """Generate a shift value for permute operations."""
    return draw(st.integers(min_value=-dimension * 2, max_value=dimension * 2))


class TestBindCommutativity:
    """Test commutativity property of bind(): a.bind(b) == b.bind(a)"""

    @given(vectors=binary_hdv_pair_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_bind_commutativity(self, vectors):
        """bind(a, b) == bind(b, a)"""
        a, b = vectors
        assert a.bind(b) == b.bind(a), "bind() must be commutative"


class TestBindUnbindInverse:
    """Test inverse property: unbind(bind(a, b), b) == a"""

    @given(vectors=binary_hdv_pair_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_bind_unbind_inverse(self, vectors):
        """unbind(bind(a, b), b) == a"""
        a, b = vectors
        # XOR bind is self-inverse, so unbind = bind
        bound = a.bind(b)
        recovered = bound.unbind(b)
        assert recovered == a, "unbind(bind(a, b), b) must equal a"


class TestPermuteSelfInverse:
    """Test self-inverse property of permute(): permute(permute(a, k), -k) == a"""

    @given(vectors=binary_hdv_strategy(), shift=shift_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_permute_self_inverse(self, vectors, shift):
        """permute(permute(a, k), -k) == a"""
        a = vectors
        assume(shift != 0)  # Skip trivial case
        permuted = a.permute(shift)
        recovered = permuted.permute(-shift)
        assert recovered == a, f"permute(permute(a, {shift}), {-shift}) must equal a"

    @given(vectors=binary_hdv_strategy(), shift=shift_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_permute_full_cycle(self, vectors, shift):
        """permute(a, dimension) == a (full cycle returns original)"""
        a = vectors
        assume(shift != 0)
        # Normalize shift to dimension
        normalized_shift = shift % TEST_DIMENSION
        # Permute by dimension returns original
        permuted_by_dim = a.permute(TEST_DIMENSION)
        assert permuted_by_dim == a, "permute(a, dimension) must equal a"


class TestHammingDistanceIdentity:
    """Test Hamming distance identity: hamming(a, a) == 0"""

    @given(vector=binary_hdv_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_hamming_distance_identity(self, vector):
        """hamming(a, a) == 0"""
        a = vector
        assert a.hamming_distance(a) == 0, "hamming(a, a) must equal 0"


class TestHammingDistanceSymmetry:
    """Test Hamming distance symmetry: hamming(a, b) == hamming(b, a)"""

    @given(vectors=binary_hdv_pair_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_hamming_distance_symmetry(self, vectors):
        """hamming(a, b) == hamming(b, a)"""
        a, b = vectors
        assert a.hamming_distance(b) == b.hamming_distance(a), \
            "hamming(a, b) must equal hamming(b, a)"


class TestHammingDistanceNormalization:
    """Test Hamming distance normalization: normalized_distance in [0, 1]"""

    @given(vectors=binary_hdv_pair_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_normalized_distance_range(self, vectors):
        """normalized_distance(a, b) in [0.0, 1.0]"""
        a, b = vectors
        nd = a.normalized_distance(b)
        assert 0.0 <= nd <= 1.0, f"normalized_distance must be in [0, 1], got {nd}"

    @given(vectors=binary_hdv_pair_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_similarity_range(self, vectors):
        """similarity(a, b) in [0.0, 1.0]"""
        a, b = vectors
        sim = a.similarity(b)
        assert 0.0 <= sim <= 1.0, f"similarity must be in [0, 1], got {sim}"

    @given(vectors=binary_hdv_pair_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_normalized_distance_consistency(self, vectors):
        """normalized_distance(a, b) == hamming_distance(a, b) / dimension"""
        a, b = vectors
        expected = a.hamming_distance(b) / a.dimension
        actual = a.normalized_distance(b)
        assert actual == expected, \
            f"normalized_distance must equal hamming_distance / dimension"


class TestDeterminism:
    """Test determinism: same input always produces same output"""

    @given(seed=st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_from_seed_determinism(self, seed):
        """from_seed(seed) always produces the same vector"""
        v1 = BinaryHDV.from_seed(seed, TEST_DIMENSION)
        v2 = BinaryHDV.from_seed(seed, TEST_DIMENSION)
        assert v1 == v2, f"from_seed('{seed}') must be deterministic"

    @given(vector=binary_hdv_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_xor_bind_determinism(self, vector):
        """bind(a, b) always produces the same result for same inputs"""
        a = vector
        # Create a second random vector
        b = BinaryHDV.random(TEST_DIMENSION)
        result1 = a.bind(b)
        result2 = a.bind(b)
        assert result1 == result2, "bind() must be deterministic"

    @given(vector=binary_hdv_strategy(), shift=shift_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_permute_determinism(self, vector, shift):
        """permute(a, shift) always produces the same result for same inputs"""
        a = vector
        result1 = a.permute(shift)
        result2 = a.permute(shift)
        assert result1 == result2, "permute() must be deterministic"


class TestAdditionalAlgebraicProperties:
    """Additional algebraic property tests"""

    @given(vectors=binary_hdv_triple_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_bind_associativity(self, vectors):
        """(a.bind(b)).bind(c) == a.bind(b.bind(c))"""
        a, b, c = vectors
        lhs = a.bind(b).bind(c)
        rhs = a.bind(b.bind(c))
        assert lhs == rhs, "bind() must be associative"

    @given(vector=binary_hdv_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_bind_self_inverse(self, vector):
        """a.bind(a) == zeros"""
        a = vector
        result = a.bind(a)
        zeros = BinaryHDV.zeros(TEST_DIMENSION)
        assert result == zeros, "a.bind(a) must equal zero vector"

    @given(vectors=binary_hdv_triple_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_hamming_triangle_inequality(self, vectors):
        """hamming(a, c) <= hamming(a, b) + hamming(b, c)"""
        a, b, c = vectors
        assert a.hamming_distance(c) <= a.hamming_distance(b) + b.hamming_distance(c), \
            "Hamming distance must satisfy triangle inequality"

    @given(vectors=binary_hdv_triple_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_bind_preserves_distance(self, vectors):
        """hamming(a.bind(c), b.bind(c)) == hamming(a, b)"""
        a, b, c = vectors
        dist_ab = a.hamming_distance(b)
        dist_ac_bc = a.bind(c).hamming_distance(b.bind(c))
        assert dist_ab == dist_ac_bc, \
            f"bind must preserve distance: {dist_ab} != {dist_ac_bc}"

    @given(vector=binary_hdv_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_invert_is_max_distance(self, vector):
        """hamming(a, a.invert()) == dimension"""
        a = vector
        assert a.hamming_distance(a.invert()) == TEST_DIMENSION, \
            "hamming(a, ~a) must equal dimension"

    @given(vector=binary_hdv_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_invert_is_self_inverse(self, vector):
        """a.invert().invert() == a"""
        a = vector
        recovered = a.invert().invert()
        assert recovered == a, "double invert must return original"
