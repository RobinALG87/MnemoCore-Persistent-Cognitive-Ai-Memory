import numpy as np
import sys

print("Python version:", sys.version)
print("Numpy version:", np.__version__)
print("Generating random array...")
try:
    arr = np.random.randint(0, 256, size=2048, dtype=np.uint8)
    print("Random array generated.")
    print("Shape:", arr.shape)
    print("First 5 bytes:", arr[:5])
except Exception as e:
    print("Error:", e)
