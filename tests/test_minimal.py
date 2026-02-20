import base64
import os
import sys
import unittest

import numpy as np

# Add root to checking path
sys.path.append(os.getcwd())

from mnemocore.core.binary_hdv import BinaryHDV


class TestMinimal(unittest.TestCase):
    def test_packed_payload_roundtrip(self):
        print("Starting test...")
        dim = 16384
        original_hdv = BinaryHDV.random(dimension=dim)

        packed_bytes = original_hdv.data.tobytes()
        packed_b64 = base64.b64encode(packed_bytes).decode("ascii")

        payload = {"hdv_packed_b64": packed_b64, "dimension": dim, "hdv_type": "binary"}

        restored_bytes = base64.b64decode(payload["hdv_packed_b64"])
        restored_packed = np.frombuffer(restored_bytes, dtype=np.uint8)
        restored_hdv = BinaryHDV(data=restored_packed, dimension=payload["dimension"])

        np.testing.assert_array_equal(original_hdv.data, restored_hdv.data)
        print("Test passed!")


if __name__ == "__main__":
    unittest.main()
