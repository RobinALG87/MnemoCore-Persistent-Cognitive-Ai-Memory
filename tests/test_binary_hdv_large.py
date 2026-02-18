
import numpy as np
import pytest
from src.core.binary_hdv import BinaryHDV

class TestLargeDimensionPermutation:
    """Tests for large-dimension permute correctness."""

    LARGE_DIM = 65536 # Well above 32768 threshold

    def test_permute_large_dim_correctness(self):
        """
        Compare permute implementation against the golden reference
        (unpackbits->roll->packbits).
        """
        D = self.LARGE_DIM
        v = BinaryHDV.random(D)

        # Golden reference implementation
        bits = np.unpackbits(v.data)

        test_shifts = [0, 1, 7, 8, 9, 100, D-1, D+1]

        for shift in test_shifts:
            # Expected
            shifted_bits = np.roll(bits, shift)
            expected_data = np.packbits(shifted_bits)

            # Actual
            permuted = v.permute(shift)

            assert permuted.dimension == D
            assert np.array_equal(permuted.data, expected_data), \
                f"Mismatch for shift {shift}"

    def test_permute_invertible(self):
        D = self.LARGE_DIM
        a = BinaryHDV.random(D)
        b = a.permute(123).permute(-123)
        assert a == b

    def test_permute_full_cycle(self):
        D = self.LARGE_DIM
        a = BinaryHDV.random(D)
        assert a.permute(D) == a
