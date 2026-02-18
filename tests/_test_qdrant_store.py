"""
Tests for QdrantStore (Phase 3.5.2)
===================================
Uses unittest.mock to simulate Qdrant backend interactions.
Rewritten to use pytest-asyncio for proper async support.
"""

import pytest
from unittest.mock import MagicMock, patch, ANY
from mnemocore.core.config import get_config, reset_config
from mnemocore.core.qdrant_store import QdrantStore

@pytest.fixture
def mock_qdrant_client():
    with patch("src.core.qdrant_store.AsyncQdrantClient") as MockClass:
        mock_instance = MockClass.return_value
        # Setup default behaviors
        mock_instance.collection_exists.return_value = False
        yield mock_instance

@pytest.fixture
def store(mock_qdrant_client):
    reset_config()
    # Bypass get_instance() patch from conftest.py by instantiating directly
    # We want to test the logic of the class, not the singleton mechanism
    return QdrantStore()

@pytest.mark.asyncio
async def test_ensure_collections(store, mock_qdrant_client):
    # Setup mock to say collections don't exist (already default, but explicit here)
    mock_qdrant_client.collection_exists.return_value = False
    
    await store.ensure_collections()
    
    # Should create HOT and WARM
    assert mock_qdrant_client.create_collection.call_count == 2
    
    # Verify calls
    config = get_config().qdrant
    mock_qdrant_client.create_collection.assert_any_call(
        collection_name=config.collection_hot,
        vectors_config=ANY,
        quantization_config=ANY,
        hnsw_config=ANY
    )
    mock_qdrant_client.create_collection.assert_any_call(
        collection_name=config.collection_warm,
        vectors_config=ANY,
        quantization_config=ANY,
        hnsw_config=ANY
    )

@pytest.mark.asyncio
async def test_upsert(store, mock_qdrant_client):
    points = [MagicMock()]
    await store.upsert("test_coll", points)
    mock_qdrant_client.upsert.assert_called_with(collection_name="test_coll", points=points)

@pytest.mark.asyncio
async def test_search(store, mock_qdrant_client):
    query = [0.1, 0.2]
    await store.search("test_coll", query, limit=5)
    mock_qdrant_client.search.assert_called_with(
        collection_name="test_coll",
        query_vector=query,
        limit=5,
        score_threshold=0.0
    )

@pytest.mark.asyncio
async def test_get_point(store, mock_qdrant_client):
    await store.get_point("test_coll", "id1")
    mock_qdrant_client.retrieve.assert_called_with(
        collection_name="test_coll",
        ids=["id1"],
        with_vectors=True,
        with_payload=True
    )

@pytest.mark.asyncio
async def test_delete(store, mock_qdrant_client):
    await store.delete("test_coll", ["id1"])
    mock_qdrant_client.delete.assert_called_once()
