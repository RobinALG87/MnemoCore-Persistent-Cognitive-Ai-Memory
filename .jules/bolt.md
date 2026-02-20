## 2025-05-23 - Vectorized Popcount Optimization
**Learning:** `np.bitwise_count` (available in NumPy 2.0+) is ~5x faster than a precomputed lookup table for population count on packed uint8 arrays.
**Action:** Always prefer `np.bitwise_count` over lookup tables for Hamming distance calculations when NumPy 2.0+ is available. Use `hasattr(np, 'bitwise_count')` for backward compatibility.

## 2025-05-23 - Batch Rolling Overhead
**Learning:** For small batch sizes (e.g., K=150 rows), a Python loop calling `np.roll` is significantly faster (0.55s) than fully vectorized fancy indexing (6.1s) because constructing the large index array (O(K*D)) incurs heavy memory allocation overhead.
**Action:** Benchmark before replacing loops with fancy indexing for operations that require large auxiliary index arrays.
