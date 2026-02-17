import pytest
from unittest.mock import MagicMock, patch, AsyncMock

@pytest.fixture(scope="session", autouse=True)
def mock_hardware_dependencies():
    """Globally mock Qdrant and Redis to prevent hangs during testing."""
    
    # 1. Mock Redis
    mock_redis_instance = AsyncMock()
    mock_redis_instance.check_health.return_value = True
    mock_redis_instance.ping.return_value = True

    # Setup redis_client (synchronous access in middleware)
    mock_redis_client = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__.return_value = mock_pipeline
    mock_pipeline.__aexit__.return_value = None
    mock_pipeline.execute = AsyncMock(return_value=[1, True]) # Default success
    mock_redis_client.pipeline.return_value = mock_pipeline
    mock_redis_instance.redis_client = mock_redis_client
    
    with patch("src.core.async_storage.AsyncRedisStorage.get_instance", return_value=mock_redis_instance), \
         patch("src.core.qdrant_store.QdrantStore.get_instance") as mock_qdrant_get:
        
        mock_qdrant_instance = MagicMock()
        mock_qdrant_get.return_value = mock_qdrant_instance
        mock_qdrant_instance.ensure_collections.return_value = None
        
        yield (mock_qdrant_instance, mock_redis_instance)

@pytest.fixture(autouse=True)
def clean_config():
    """Reset config state between tests."""
    from src.core.config import reset_config
    reset_config()
    yield
    reset_config()
