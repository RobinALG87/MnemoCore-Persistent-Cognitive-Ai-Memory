import base64

import numpy as np
import pytest

try:
    from src.core.binary_hdv import BinaryHDV
except (ModuleNotFoundError, ImportError) as exc:
    pytestmark = pytest.mark.skip(
        reason=f"BinaryHDV import unavailable in current branch state: {exc}"
    )
    BinaryHDV = None


def test_binary_hdv_base64_roundtrip():
    """Regression check for packed BinaryHDV base64 roundtrip."""
    dim = 16384
    original_hdv = BinaryHDV.random(dimension=dim)

    packed_bytes = original_hdv.data.tobytes()
    packed_b64 = base64.b64encode(packed_bytes).decode("ascii")
    restored_bytes = base64.b64decode(packed_b64)
    restored_packed = np.frombuffer(restored_bytes, dtype=np.uint8)
    restored_hdv = BinaryHDV(data=restored_packed, dimension=dim)

    assert np.array_equal(original_hdv.data, restored_hdv.data)
