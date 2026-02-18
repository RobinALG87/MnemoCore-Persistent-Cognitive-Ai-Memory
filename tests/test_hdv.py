import pytest
import numpy as np
from src.core.hdv import HDV
from src.core.exceptions import DimensionMismatchError

class TestHDV:
    def test_initialization(self):
        hdv = HDV(dimension=1000)
        assert hdv.vector.shape[0] == 1000
        assert hdv.dimension == 1000

    def test_xor_binding(self):
        """Test existing XOR binding behavior."""
        v1 = HDV(dimension=100)
        v2 = HDV(dimension=100)

        # Test that __xor__ works
        v3 = v1 ^ v2
        assert isinstance(v3, HDV)
        assert v3.dimension == 100
        assert v3.vector is not None

        # Test commutative property (approximate for convolution? No, circular convolution is commutative)
        # a * b = b * a
        v4 = v2 ^ v1
        # Use atol because of FFT floating point errors
        np.testing.assert_allclose(v3.vector, v4.vector, atol=1e-8, err_msg="Binding should be commutative")

    def test_bind_method(self):
        """Test the new bind method."""
        v1 = HDV(dimension=100)
        v2 = HDV(dimension=100)

        v3 = v1.bind(v2)
        assert isinstance(v3, HDV)
        assert v3.dimension == 100

        # Should be equivalent to XOR (which is now an alias)
        v_xor = v1 ^ v2
        np.testing.assert_allclose(v3.vector, v_xor.vector)

    def test_xor_alias(self):
        """Test that __xor__ works as an alias for bind."""
        v1 = HDV(dimension=100)
        v2 = HDV(dimension=100)

        v_bind = v1.bind(v2)
        v_xor = v1 ^ v2

        np.testing.assert_array_equal(v_bind.vector, v_xor.vector)

    def test_dimension_mismatch(self):
        v1 = HDV(dimension=100)
        v2 = HDV(dimension=200)

        with pytest.raises(DimensionMismatchError, match="Dimension mismatch"):
            _ = v1 ^ v2
