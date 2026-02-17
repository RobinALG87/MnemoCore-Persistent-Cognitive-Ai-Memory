import numpy as np
import base64
import sys
import os

# Add root to checking path
sys.path.append(os.getcwd())

print("Importing BinaryHDV...")
from src.core.binary_hdv import BinaryHDV
print("Imported.")

print("Generating random HDV...")
dim = 16384
original_hdv = BinaryHDV.random(dimension=dim)
print("Generated.")

print("Packing bytes...")
packed_bytes = original_hdv.data.tobytes()
print(f"Packed bytes len: {len(packed_bytes)}")

print("Encoding base64...")
packed_b64 = base64.b64encode(packed_bytes).decode('ascii')
print(f"Base64 len: {len(packed_b64)}")

print("Decoding base64...")
restored_bytes = base64.b64decode(packed_b64)
print(f"Decoded bytes len: {len(restored_bytes)}")

print("Converting to numpy...")
restored_packed = np.frombuffer(restored_bytes, dtype=np.uint8)
print("Converted.")

print("Creating new HDV...")
restored_hdv = BinaryHDV(data=restored_packed, dimension=dim)
print("Created.")

print("Comparing...")
if np.array_equal(original_hdv.data, restored_hdv.data):
    print("SUCCESS: Arrays match.")
else:
    print("FAILURE: Arrays differ.")

print("Done.")
