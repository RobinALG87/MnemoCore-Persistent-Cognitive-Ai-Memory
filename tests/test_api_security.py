import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.api.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_public(client):
    """Health endpoint should be public."""
    # We need to mock get_stats on the engine in app.state
    # The lifespan puts a HAIMEngine there.
    # Since we mocked HAIMEngine global in conftest (or we should), it should be fine.
    # Wait, I didn't mock HAIMEngine in conftest, only Qdrant/Redis inside it.
    
    # Let's mock the engine's stats call specifically for this test
    with patch.object(app.state.engine, 'get_stats', return_value={"status": "ok"}):
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

def test_secure_endpoints(client, monkeypatch):
    """Verify endpoints require X-API-Key."""
    monkeypatch.setenv("HAIM_API_KEY", "test-key")
    from src.core.config import reset_config
    reset_config()
    
    # 1. Store
    response = client.post("/store", json={"content": "test"})
    assert response.status_code == 403
    
    # 2. Query
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 403
    
    # 3. Valid key
    with patch.object(app.state.engine, 'store', return_value="mem_1"), \
         patch.object(app.state.engine, 'get_memory', return_value=MagicMock()):
        response = client.post(
            "/store", 
            json={"content": "test"},
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
