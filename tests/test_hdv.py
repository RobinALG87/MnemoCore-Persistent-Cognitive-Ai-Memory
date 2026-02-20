"""
Tests for HDV module (deprecated) and BinaryHDV compatibility shims.

This test file verifies:
1. Legacy HDV class still works (with deprecation warnings)
2. BinaryHDV compatibility shims work correctly
3. Migration path is valid
"""

import warnings

import numpy as np
import pytest

# Test BinaryHDV with compatibility shims
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.exceptions import DimensionMismatchError

# Test legacy HDV (deprecated)
from mnemocore.core.hdv import HDV


class TestLegacyHDV:
    """Tests for the deprecated HDV class."""

    def test_initialization(self):
        """Test HDV initialization."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hdv = HDV(dimension=1000)
            assert hdv.vector.shape[0] == 1000
            assert hdv.dimension == 1000
            # Import warning is emitted
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_xor_binding(self):
        """Test existing XOR binding behavior."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            v1 = HDV(dimension=100)
            v2 = HDV(dimension=100)

            # Test that __xor__ works
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                v3 = v1 ^ v2
                assert isinstance(v3, HDV)
                assert v3.dimension == 100
                assert v3.vector is not None
                # Deprecation warning should be emitted
                assert any(issubclass(x.category, DeprecationWarning) for x in w)

            # Test commutative property (circular convolution is commutative)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                v4 = v2 ^ v1
            np.testing.assert_allclose(
                v3.vector, v4.vector, atol=1e-8, err_msg="Binding should be commutative"
            )

    def test_bind_method(self):
        """Test the bind method."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            v1 = HDV(dimension=100)
            v2 = HDV(dimension=100)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                v3 = v1.bind(v2)
                assert isinstance(v3, HDV)
                assert v3.dimension == 100
                # Deprecation warning should be emitted
                assert any(issubclass(x.category, DeprecationWarning) for x in w)

            # Should be equivalent to XOR (which is an alias)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                v_xor = v1 ^ v2
            np.testing.assert_allclose(v3.vector, v_xor.vector)

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            v1 = HDV(dimension=100)
            v2 = HDV(dimension=200)

            with pytest.raises(DimensionMismatchError, match="Dimension mismatch"):
                _ = v1 ^ v2


class TestBinaryHDVCompatibilityShims:
    """Tests for BinaryHDV compatibility shims that match HDV API."""

    def test_bind_shim(self):
        """Test that bind() works as alias for xor_bind()."""
        v1 = BinaryHDV.random(dimension=1000)
        v2 = BinaryHDV.random(dimension=1000)

        # bind() should be equivalent to xor_bind()
        v_bind = v1.bind(v2)
        v_xor = v1.xor_bind(v2)

        assert v_bind == v_xor

    def test_unbind_shim(self):
        """Test that unbind() works (XOR is self-inverse)."""
        v1 = BinaryHDV.random(dimension=1000)
        v2 = BinaryHDV.random(dimension=1000)

        # unbind() should be equivalent to xor_bind() for XOR
        v_unbind = v1.unbind(v2)
        v_xor = v1.xor_bind(v2)

        assert v_unbind == v_xor

        # Self-inverse property: (a XOR b) XOR b = a
        recovered = v_unbind.xor_bind(v2)
        assert recovered == v1

    def test_cosine_similarity_shim(self):
        """Test that cosine_similarity() is an alias for similarity()."""
        v1 = BinaryHDV.random(dimension=1000)
        v2 = BinaryHDV.random(dimension=1000)

        cosine_sim = v1.cosine_similarity(v2)
        sim = v1.similarity(v2)

        assert cosine_sim == sim
        assert 0.0 <= cosine_sim <= 1.0

    def test_normalize_shim(self):
        """Test that normalize() returns a copy."""
        v1 = BinaryHDV.random(dimension=1000)

        normalized = v1.normalize()

        # For binary vectors, normalize is a no-op (returns copy)
        assert normalized == v1
        assert normalized is not v1  # Should be a different object
        assert normalized.data is not v1.data  # Data should be copied

    def test_xor_operator(self):
        """Test that __xor__ operator works for binding."""
        v1 = BinaryHDV.random(dimension=1000)
        v2 = BinaryHDV.random(dimension=1000)

        # v1 ^ v2 should use xor_bind
        v_xor = v1 ^ v2
        v_bind = v1.xor_bind(v2)

        assert v_xor == v_bind

    def test_full_roundtrip(self):
        """Test a full roundtrip: bind, unbind, similarity."""
        v_a = BinaryHDV.random(dimension=1000)
        v_b = BinaryHDV.random(dimension=1000)

        # Bind A and B
        v_ab = v_a.bind(v_b)

        # Unbind to recover A
        v_recovered = v_ab.unbind(v_b)

        # Should be similar to original A
        similarity = v_a.similarity(v_recovered)
        assert similarity == 1.0  # XOR is exact, not approximate


class TestMigrationPath:
    """Tests verifying the migration path from HDV to BinaryHDV."""

    def test_api_equivalence(self):
        """
        Verify that BinaryHDV has all methods needed to replace HDV.
        This test documents the migration path.
        """
        # Create equivalent vectors
        # HDV: hdv = HDV(dimension=10000)
        # BinaryHDV: hdv = BinaryHDV.random(dimension=16384)
        binary_hdv = BinaryHDV.random(dimension=16384)

        # All these methods should exist on BinaryHDV:
        assert hasattr(binary_hdv, "bind")
        assert hasattr(binary_hdv, "unbind")
        assert hasattr(binary_hdv, "permute")
        assert hasattr(binary_hdv, "cosine_similarity")
        assert hasattr(binary_hdv, "normalize")
        assert hasattr(binary_hdv, "xor_bind")
        assert hasattr(binary_hdv, "similarity")
        assert hasattr(binary_hdv, "hamming_distance")

    def test_xor_binding_self_inverse(self):
        """
        Demonstrate that XOR binding is self-inverse (unlike HRR).
        This is a key difference but makes the API simpler.
        """
        v_a = BinaryHDV.random(dimension=1000)
        v_b = BinaryHDV.random(dimension=1000)

        # XOR binding
        v_ab = v_a.xor_bind(v_b)

        # XOR unbinding (same operation!)
        v_recovered = v_ab.xor_bind(v_b)

        # Exact recovery (not approximate like HRR)
        assert v_recovered == v_a

        # Same with bind/unbind shims
        v_ab_shim = v_a.bind(v_b)
        v_recovered_shim = v_ab_shim.unbind(v_b)
        assert v_recovered_shim == v_a
