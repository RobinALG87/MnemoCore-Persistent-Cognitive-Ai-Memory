import numpy as np
from threading import Lock

from mnemocore.core.hnsw_index import HNSWIndexManager
import mnemocore.core.hnsw_index as hnsw_module


class _FakeIndex:
    def __init__(self, d: int, distances, ids):
        self.d = d
        self._distances = np.asarray(distances, dtype=np.int32)
        self._ids = np.asarray(ids, dtype=np.int64)
        self.last_query_shape = None

    def search(self, q, k):
        self.last_query_shape = q.shape
        return self._distances[:, :k], self._ids[:, :k]


def test_search_uses_index_dimension_and_clamps_scores(monkeypatch):
    manager = HNSWIndexManager.__new__(HNSWIndexManager)
    manager.dimension = 16384
    manager._id_map = ["n1", "n2"]
    manager._stale_count = 0
    manager._write_lock = Lock()
    manager._index = _FakeIndex(d=10000, distances=[[12000, 5000]], ids=[[0, 1]])

    monkeypatch.setattr(hnsw_module, "FAISS_AVAILABLE", True)

    # Query is from a 16384-bit service (2048 bytes), but index is 10000-bit (1250 bytes)
    query = np.zeros(2048, dtype=np.uint8)
    results = manager.search(query, top_k=2)

    # Query should be adjusted to index byte width
    assert manager._index.last_query_shape == (1, 1250)

    assert results[0] == ("n1", 0.0)  # 1 - 12000/10000 => clamped to 0.0
    assert results[1] == ("n2", 0.5)  # 1 - 5000/10000
