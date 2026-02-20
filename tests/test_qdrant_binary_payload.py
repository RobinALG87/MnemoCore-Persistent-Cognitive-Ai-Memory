import base64
import os

# Adjust path to import core modules
import sys
import unittest
from datetime import datetime

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.node import MemoryNode


class TestQdrantBinaryPayload(unittest.TestCase):
    def test_packed_payload_roundtrip(self):
        """Verify that BinaryHDV can be packed to base64 payload and restored exactly."""
        dim = 16384
        original_hdv = BinaryHDV.random(dimension=dim)

        # --- Simulate _save_to_warm logic ---
        packed_bytes = original_hdv.data.tobytes()
        packed_b64 = base64.b64encode(packed_bytes).decode("ascii")

        self.assertTrue(isinstance(packed_b64, str))
        # Expected size: 2048 bytes * 4/3 base64 expansion ~= 2732 chars
        self.assertAlmostEqual(len(packed_b64), 2732, delta=100)

        payload = {"hdv_packed_b64": packed_b64, "dimension": dim, "hdv_type": "binary"}

        # --- Simulate _load_from_warm logic ---
        # 1. Check if packed data exists
        self.assertIn("hdv_packed_b64", payload)

        # 2. Decode
        restored_bytes = base64.b64decode(payload["hdv_packed_b64"])
        self.assertEqual(len(restored_bytes), dim // 8)

        # 3. Restore Numpy array
        restored_packed = np.frombuffer(restored_bytes, dtype=np.uint8)
        restored_hdv = BinaryHDV(data=restored_packed, dimension=payload["dimension"])

        # --- Verification ---
        self.assertEqual(original_hdv.dimension, restored_hdv.dimension)
        np.testing.assert_array_equal(original_hdv.data, restored_hdv.data)
        self.assertEqual(original_hdv, restored_hdv)

        print("\n[SUCCESS] BinaryHDV roundtrip successful.")
        print(f"Original size (packed): {len(packed_bytes)} bytes")
        print(f"Payload size (Base64): {len(packed_b64)} bytes")

    def test_vector_conversion_performance(self):
        """Check verify vector conversion overhead is low (conceptually)."""
        dim = 16384
        hdv = BinaryHDV.random(dimension=dim)

        # Current optimization:
        bits = np.unpackbits(hdv.data)
        vector_np = bits.astype(np.float32)

        self.assertIsInstance(vector_np, np.ndarray)
        self.assertEqual(vector_np.dtype, np.float32)
        self.assertEqual(len(vector_np), dim)

        # Previous (Bad) way:
        # vector_list = vector_np.tolist()
        # We just assume this is heavier.


if __name__ == "__main__":
    unittest.main()
