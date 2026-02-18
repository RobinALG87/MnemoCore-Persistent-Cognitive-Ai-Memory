
import time
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.binary_hdv import TextEncoder

def benchmark():
    print("Initializing Engine...")
    engine = HAIMEngine(dimension=10000) # Slightly smaller for quick bench if needed, but 10k is realistic
    
    encoded_text = "The quick brown fox jumps over the lazy dog " * 50 # 450 words
    print(f"Text length: {len(encoded_text.split())} tokens")

    # 1. Benchmark Legacy Encoding
    print("\n--- Legacy Encoding (Float/Numpy) ---")
    
    # Force legacy mode if possible or just call the method directly to be sure
    start_time = time.time()
    iterations = 50
    for _ in range(iterations):
        _ = engine._legacy_encode_content_numpy(encoded_text)
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f"Average time per document: {avg_time*1000:.2f} ms")
    print(f"Total time for {iterations} docs: {end_time - start_time:.4f} s")

    # 2. Benchmark Binary Encoding
    print("\n--- Binary Encoding (BinaryHDV) ---")
    encoder = TextEncoder(dimension=10000)
    
    start_time = time.time()
    iterations = 50
    for _ in range(iterations):
        # Clear cache to measure raw encoding speed? 
        # Or measure with cache to see benefit?
        # Real usage is with cache, but "first load" matters too.
        # Let's keep cache for now as it's the default behavior.
        _ = encoder.encode(encoded_text)
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f"Average time per document (with cache): {avg_time*1000:.2f} ms")

    # Measure without cache hit (force new tokens)
    print("\n--- Binary Encoding (Cold Cache equivalent) ---")
    start_time = time.time()
    iterations = 10
    for i in range(iterations):
        # Unique tokens every time
        unique_text = f"{encoded_text} {i}" 
        _ = encoder.encode(unique_text)
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f"Average time per document (unique suffix): {avg_time*1000:.2f} ms")

if __name__ == "__main__":
    benchmark()
