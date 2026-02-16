import pytest
from unittest.mock import MagicMock, patch, AsyncMock

@pytest.fixture(scope="session", autouse=True)
def mock_hardware_dependencies():
    """Globally mock Qdrant and Redis to prevent hangs during testing."""
    
    # 1. Mock Redis
    mock_redis_instance = AsyncMock()
    mock_redis_instance.check_health.return_value = True
    mock_redis_instance.ping.return_value = True
    
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
