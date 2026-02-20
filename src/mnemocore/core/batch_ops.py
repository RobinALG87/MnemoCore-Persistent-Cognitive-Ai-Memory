"""
Batch Processing and GPU Acceleration (Phase 3.5.5)
===================================================
Implementation of batch operations for HDVs, leveraging PyTorch for GPU acceleration
if available, with fallback to NumPy (CPU).

Designed to scale comfortably from Raspberry Pi (CPU) to dedicated AI rigs (CUDA).
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

# Try importing torch, handle failure gracefully (CPU only environment)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from .binary_hdv import BinaryHDV, TextEncoder, batch_hamming_distance


def _encode_single_worker(args: tuple) -> bytes:
    """Module-level worker function for ProcessPoolExecutor (must be picklable)."""
    text, dim = args
    encoder = TextEncoder(dimension=dim)
    hdv = encoder.encode(text)
    return hdv.to_bytes()


class BatchProcessor:
    """
    Handles batched operations for HDV encoding and search.
    Automatically selects the best available backend (CUDA > MPS > CPU).
    """

    def __init__(self, use_gpu: bool = True, num_workers: Optional[int] = None):
        """
        Args:
            use_gpu: Whether to attempt using GPU acceleration.
            num_workers: Number of CPU workers for encoding (defaults to CPU count).
        """
        self.device = self._detect_device(use_gpu)
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.popcount_table_gpu = None  # Lazy init

        logger.info(f"BatchProcessor initialized on device: {self.device}")

        # Initialize text encoder for workers (pickled)
        # Note: TextEncoder is lightweight, so re-init in workers is fine.

    def _detect_device(self, use_gpu: bool) -> str:
        """Detect the best available compute device."""
        if not use_gpu or not TORCH_AVAILABLE:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _ensure_gpu_table(self):
        """Initialize bits-set lookup table on GPU if needed."""
        if self.device == "cpu" or self.popcount_table_gpu is not None:
            return

        # Precompute table matching numpy's _build_popcount_table
        # 256 values (0-255), value is number of bits set
        table = torch.tensor(
            [bin(i).count("1") for i in range(256)],
            dtype=torch.int32,  # int32 suffices (max 8)
            device=self.device,
        )
        self.popcount_table_gpu = table

    def encode_batch(self, texts: List[str], dimension: int = 16384) -> List[BinaryHDV]:
        """
        Encode a batch of texts into BinaryHDVs using parallel CPU processing.

        Encoding logic is strictly CPU-bound (tokenization + python loops),
        so we use ProcessPoolExecutor to bypass the GIL.
        """
        if not texts:
            return []

        results = [None] * len(texts)

        # Parallel execution using module-level worker (picklable)
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {
                executor.submit(_encode_single_worker, (text, dimension)): i
                for i, text in enumerate(texts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    raw_bytes = future.result()
                    results[idx] = BinaryHDV.from_bytes(raw_bytes, dimension=dimension)
                except Exception as e:
                    logger.error(f"Encoding failed for item {idx}: {e}")
                    results[idx] = BinaryHDV.zeros(dimension)

        return results

    def search_batch(
        self, queries: List[BinaryHDV], targets: List[BinaryHDV]
    ) -> np.ndarray:
        """
        Compute Hamming distance matrix between queries and targets.

        Args:
            queries: List of M query vectors.
            targets: List of N target vectors.

        Returns:
            np.ndarray of shape (M, N) with Hamming distances.
        """
        if not queries or not targets:
            return np.array([[]])

        # Prepare data containers
        d_bytes = queries[0].dimension // 8

        # Convert to numpy arrays first (fast packing)
        # Shape: (M, D//8) and (N, D//8)
        query_arr = np.stack([q.data for q in queries])
        target_arr = np.stack([t.data for t in targets])

        if self.device == "cpu":
            return self._search_cpu(query_arr, target_arr)
        else:
            return self._search_gpu(query_arr, target_arr)

    def _search_cpu(self, query_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        """NumPy-based batch Hamming distance."""
        # query_arr: (M, B), target_arr: (N, B)
        # We need (M, N) distance matrix.
        # Broadcasting: (M, 1, B) x (1, N, B) -> (M, N, B)
        # Memory warning: M*N*B bytes could be large.

        M, B = query_arr.shape
        N = target_arr.shape[0]

        # Optimization: Process in chunks if M*N is large to avoid OOM
        # For now, simplistic implementation

        dists = np.zeros((M, N), dtype=np.int32)

        # Iterate over queries to save memory (broadcast takes M*N*B RAM)
        # Using the batch_hamming_distance from binary_hdv which is (1 vs N)
        # We can reuse the logic but call it M times?
        # Or reimplement broadcasting with chunks.

        # Reusing binary_hdv logic for safety but adapting to array inputs
        # batch_hamming_distance in binary_hdv takes (BinaryHDV, database)
        # We have raw arrays here.

        # Let's import the table builder
        from .binary_hdv import _build_popcount_table

        popcount_table = _build_popcount_table()

        for i in range(M):
            # (N, B) XOR (B,) -> (N, B)
            xor_result = np.bitwise_xor(target_arr, query_arr[i])
            # (N, B) -> (N,) sums
            dists[i] = popcount_table[xor_result].sum(axis=1)

        return dists

    def _search_gpu(self, query_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        """PyTorch-based batch Hamming distance."""
        self._ensure_gpu_table()

        # Transfer to GPU
        # uint8 in numpy -> uint8 in torch (ByteTensor)
        q_tensor = torch.from_numpy(query_arr).to(self.device)  # (M, B)
        t_tensor = torch.from_numpy(target_arr).to(self.device)  # (N, B)

        M = q_tensor.shape[0]
        N = t_tensor.shape[0]

        # Result matrix
        dists = torch.zeros((M, N), dtype=torch.int32, device=self.device)

        # Chunking to fit in VRAM?
        # A 16k-bit vector is 2KB. 1M vectors = 2GB.
        # (M, 1, B) x (1, N, B) -> (M, N, B) int8 is huge.
        # XOR is huge. We must loop queries or targets.

        # Loop over queries (like CPU implementation) is safest for VRAM
        for i in range(M):
            # (1, B) XOR (N, B) -> (N, B)
            # Bitwise XOR supported on ByteTensor? Yes.
            xor_result = torch.bitwise_xor(t_tensor, q_tensor[i])

            # Lookup popcount: (N, B) [0-255] -> int32 count
            # We treat xor_result as indices into the table
            # xor_result is uint8, needs to be long/int64 for indexing in torch usually
            counts = self.popcount_table_gpu[xor_result.long()]

            # Sum bits
            dists[i] = counts.sum(dim=1)

        return dists.cpu().numpy()
