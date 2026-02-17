import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from src.api.main import app, get_engine, get_api_key

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def cleanup_overrides():
    yield
    app.dependency_overrides = {}

def test_health_public(client):
    """Health endpoint should be public."""
    # Mocking engine calls
    # app.state.engine is set in lifespan, but we can mock dependencies
    mock_engine = MagicMock()
    mock_engine.get_stats.return_value = {"status": "ok"}
    app.dependency_overrides[get_engine] = lambda: mock_engine
    
    # Also need to mock Redis health check if it runs
    with patch("src.api.main.AsyncRedisStorage.get_instance") as mock_redis:
        mock_redis.return_value.check_health = AsyncMock(return_value=True)

        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


def test_secure_endpoints(client, monkeypatch):
    """Verify endpoints require X-API-Key."""
    # We need to mock config to enforce API key check
    # But get_api_key dependency is what we are testing.
    
    # Mock engine for valid calls
    mock_engine = MagicMock()
    mock_engine.store.return_value = "mem_1"
    mock_node = MagicMock()
    mock_node.id = "mem_1"
    mock_node.content = "test"
    mock_node.metadata = {}
    mock_node.ltp_strength = 0.5
    mock_node.created_at.isoformat.return_value = "2023-01-01T00:00:00"
    mock_engine.get_memory.return_value = mock_node
    app.dependency_overrides[get_engine] = lambda: mock_engine
    
    # We need to patch get_config inside get_api_key or just rely on env vars if get_config reads them
    # The implementation uses get_config() inside get_api_key.
    
    # To test failure (403), we need get_api_key to raise HTTPException.
    # The default get_api_key checks header vs config.

    # 1. Store (No Key) - Should be 403
    # Note: TestClient doesn't send X-API-Key by default.
    # But if env var is not set, maybe it defaults to "mnemocore-beta-key"?
    # We should set a known key in config.

    with patch("src.api.main.get_config") as mock_conf:
        # Mock config with a specific key
        mock_conf.return_value.security.api_key = "secret-key"

        # Test without header
        response = client.post("/store", json={"content": "test"})
        # If no header, FastAPI returns 403 (auto_error=False) or passes None?
        # api_key_header(auto_error=False) passes None if missing.
        # But get_api_key raises if api_key != expected.
        assert response.status_code == 403

        # Test with wrong header
        response = client.post("/store", json={"content": "test"}, headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403

        # Test with correct header
        # We need to mock run_in_thread since store_memory calls it
        with patch("src.api.main.run_in_thread", side_effect=lambda f, *a, **k: "mem_1"):
            with patch("src.api.main.AsyncRedisStorage.get_instance"):
                 response = client.post(
                    "/store",
                    json={"content": "test"},
                    headers={"X-API-Key": "secret-key"}
                )
                 assert response.status_code == 200


def test_input_validation(client):
    """Verify input validation rules (max_length, min_length)."""
    # Override API key dependency to bypass auth for validation tests
    app.dependency_overrides[get_api_key] = lambda: "valid-key"

    # 1. StoreRequest
    # Content too short (empty)
    response = client.post("/store", json={"content": ""})
    assert response.status_code == 422

    # Content too long
    response = client.post("/store", json={"content": "A" * 100001})
    assert response.status_code == 422

    # 2. QueryRequest
    # Query too short (empty)
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422

    # Query too long
    response = client.post("/query", json={"query": "A" * 1001})
    assert response.status_code == 422

    # 3. ConceptRequest
    # Name too short
    response = client.post("/concept", json={"name": "", "attributes": {}})
    assert response.status_code == 422

    # Name too long
    response = client.post("/concept", json={"name": "A" * 101, "attributes": {}})
    assert response.status_code == 422

    # 4. AnalogyRequest
    # Fields too short/long
    response = client.post("/analogy", json={"source_concept": "", "source_value": "x", "target_concept": "y"})
    assert response.status_code == 422

    response = client.post("/analogy", json={"source_concept": "x", "source_value": "", "target_concept": "y"})
    assert response.status_code == 422

    response = client.post("/analogy", json={"source_concept": "A" * 101, "source_value": "x", "target_concept": "y"})
    assert response.status_code == 422
