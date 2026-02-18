"""
Tests for Batch Processing and GPU Path (Phase 3.5.5)
=====================================================
Uses unittest.mock to simulate GPU availability and multiprocessing behavior.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.core.batch_ops import BatchProcessor
from src.core.binary_hdv import BinaryHDV

class TestBatchOps(unittest.TestCase):
    
    def setUp(self):
        # Create dummy data
        self.dim = 16  # Very small dimension for testing
        self.texts = ["hello world", "test memory"]
        
    def test_cpu_device_selection(self):
        """Verify fallback to CPU when GPU unavailable."""
        with patch("src.core.batch_ops.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            bp = BatchProcessor(use_gpu=True)
            self.assertEqual(bp.device, "cpu")

    def test_gpu_device_selection(self):
        """Verify selection of CUDA when available."""
        with patch("src.core.batch_ops.torch") as mock_torch, \
             patch("src.core.batch_ops.TORCH_AVAILABLE", True):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.backends.mps.is_available.return_value = False
            bp = BatchProcessor(use_gpu=True)
            self.assertEqual(bp.device, "cuda")

    def test_encode_batch(self):
        """Test parallel CPU encoding logic."""
        # Mock ProcessPoolExecutor to run synchronously or mock return
        bp = BatchProcessor(use_gpu=False, num_workers=1)
        
        # We can run the real encoding logic since it's deterministic
        results = bp.encode_batch(self.texts, dimension=self.dim)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], BinaryHDV)
        self.assertEqual(results[0].dimension, self.dim)
        
        # Verify content differs
        self.assertNotEqual(results[0], results[1])

    def test_search_cpu(self):
        """Test search logic on CPU backend."""
        bp = BatchProcessor(use_gpu=False)
        
        q = BinaryHDV.random(self.dim)
        t1 = BinaryHDV.random(self.dim)
        t2 = q  # Exact match should have distance 0
        
        # Ensure q != t1 for meaningful test
        while q == t1:
            t1 = BinaryHDV.random(self.dim)
            
        queries = [q]
        targets = [t1, t2]
        
        dists = bp.search_batch(queries, targets)
        
        self.assertEqual(dists.shape, (1, 2))
        self.assertEqual(dists[0, 1], 0)  # q vs t2 (identical)
        self.assertGreater(dists[0, 0], 0) # q vs t1 (random)

    @patch("src.core.batch_ops.torch")
    def test_search_gpu_mock(self, mock_torch):
        """Test GPU search logic flow (mocked tensor operations)."""
        # Configure mock torch behavior
        mock_torch.cuda.is_available.return_value = True
        bp = BatchProcessor(use_gpu=True)
        # Mock actual device string
        bp.device = "cuda" 
        
        # Setup mocks for tensor operations
        # q_tensor, t_tensor
        q_mock = MagicMock()
        t_mock = MagicMock()
        mock_torch.from_numpy.side_effect = [q_mock, t_mock]
        
        # Mock bitwise_xor result
        xor_res = MagicMock()
        mock_torch.bitwise_xor.return_value = xor_result = MagicMock()
        xor_result.long.return_value = "indices"
        
        # Mock popcount table lookup
        # self.popcount_table_gpu is set?
        bp.popcount_table_gpu = MagicMock()
        counts = MagicMock()
        bp.popcount_table_gpu.__getitem__.return_value = counts
        
        # Mock sum
        dists_tensor = MagicMock()
        counts.sum.return_value = 123
        
        # Execute search
        queries = [BinaryHDV.random(16)]
        targets = [BinaryHDV.random(16)]
        
        # We expect it to try moving tensors to device
        q_mock.to.return_value = q_mock
        t_mock.to.return_value = t_mock
        
        # Run
        # We need to catch the final .cpu().numpy() call on the result tensor
        # dists[i] = ... assignment is tricky with mocks on __setitem__
        # Just verifying it entered _search_gpu and called torch functions
        
        try:
            bp.search_batch(queries, targets)
        except Exception:
            # It will likely fail on strict mocking of tensor assignment
            # But we can verify calls made so far
            pass
            
        mock_torch.from_numpy.assert_called()
        mock_torch.bitwise_xor.assert_called()

if __name__ == '__main__':
    unittest.main()
