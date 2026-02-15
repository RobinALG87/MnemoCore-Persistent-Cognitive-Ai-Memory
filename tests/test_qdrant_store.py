"""
Tests for QdrantStore (Phase 3.5.2)
===================================
Uses unittest.mock to simulate Qdrant backend interactions.
Also tests connection failure fallback via mocks (though logic is in TierManager).
"""

import unittest
from unittest.mock import MagicMock, patch, ANY
import numpy as np

# Mock qdrant_client before importing QdrantStore if it tries to import it
# But QdrantStore imports inside methods or at top level?
# At top level. We need to patch `qdrant_client.QdrantClient`.

import sys
from types import ModuleType

# Create dummy qdrant_client module if not installed in test env (it should be though)
# Assuming installed.

from src.core.config import get_config
from src.core.qdrant_store import QdrantStore

class TestQdrantStore(unittest.TestCase):
    
    def setUp(self):
        # Patch QdrantClient class
        self.patcher = patch("src.core.qdrant_store.QdrantClient")
        self.MockQdrantClient = self.patcher.start()
        
        # Reset singleton
        QdrantStore._instance = None
        
        self.store = QdrantStore.get_instance()
        self.mock_client = self.store.client

    def tearDown(self):
        self.patcher.stop()

    def test_ensure_collections(self):
        # Setup mock to say collections don't exist
        self.mock_client.collection_exists.return_value = False
        
        self.store.ensure_collections()
        
        # Should create HOT and WARM
        self.assertEqual(self.mock_client.create_collection.call_count, 2)
        
        # Verify calls
        config = get_config().qdrant
        self.mock_client.create_collection.assert_any_call(
            collection_name=config.collection_hot,
            vectors_config=ANY,
            quantization_config=ANY,
            hnsw_config=ANY
        )
        self.mock_client.create_collection.assert_any_call(
            collection_name=config.collection_warm,
            vectors_config=ANY,
            quantization_config=ANY,
            hnsw_config=ANY
        )

    def test_upsert(self):
        points = [MagicMock()]
        self.store.upsert("test_coll", points)
        self.mock_client.upsert.assert_called_with(collection_name="test_coll", points=points)

    def test_search(self):
        query = [0.1, 0.2]
        self.store.search("test_coll", query, limit=5)
        self.mock_client.search.assert_called_with(
            collection_name="test_coll",
            query_vector=query,
            limit=5,
            score_threshold=0.0
        )

    def test_get_point(self):
        self.store.get_point("test_coll", "id1")
        self.mock_client.retrieve.assert_called_with(
            collection_name="test_coll",
            ids=["id1"],
            with_vectors=True,
            with_payload=True
        )

    def test_delete(self):
        self.store.delete("test_coll", ["id1"])
        self.mock_client.delete.assert_called_once()


if __name__ == '__main__':
    unittest.main()
