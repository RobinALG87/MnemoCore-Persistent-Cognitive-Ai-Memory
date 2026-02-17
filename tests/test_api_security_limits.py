
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

def test_store_content_too_large():
    """Verify that content larger than 100,000 chars is rejected."""
    with TestClient(app) as client:
        large_content = "a" * 100001
        response = client.post(
            "/store",
            json={"content": large_content},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 422
        assert "String should have at most 100000 characters" in response.text

def test_store_content_valid():
    """Verify that content within limit is accepted."""
    with TestClient(app) as client:
        valid_content = "a" * 1000
        response = client.post(
            "/store",
            json={"content": valid_content},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 200

def test_query_too_large():
    """Verify query string limits."""
    with TestClient(app) as client:
        large_query = "q" * 20000
        response = client.post(
            "/query",
            json={"query": large_query},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 422
        assert "String should have at most 10000 characters" in response.text

def test_query_valid():
    """Verify query within limit is accepted."""
    with TestClient(app) as client:
        valid_query = "hello world"
        response = client.post(
            "/query",
            json={"query": valid_query},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        # It might return 200 or 500 depending on engine state, but NOT 422
        assert response.status_code != 422

def test_concept_name_too_large():
    with TestClient(app) as client:
        large_name = "n" * 10000
        response = client.post(
            "/concept",
            json={"name": large_name, "attributes": {}},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 422
        assert "String should have at most 256 characters" in response.text

def test_concept_attributes_too_many():
    with TestClient(app) as client:
        many_attributes = {f"k{i}": "v" for i in range(51)}
        response = client.post(
            "/concept",
            json={"name": "test", "attributes": many_attributes},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 422
        assert "Too many attributes" in response.text

def test_concept_attribute_key_too_long():
    with TestClient(app) as client:
        long_key = "k" * 65
        response = client.post(
            "/concept",
            json={"name": "test", "attributes": {long_key: "val"}},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 422
        assert f"Attribute key {long_key} too long" in response.text

def test_analogy_too_large():
    with TestClient(app) as client:
        large_str = "a" * 10000
        response = client.post(
            "/analogy",
            json={
                "source_concept": large_str,
                "source_value": "val",
                "target_concept": "target"
            },
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 422
        assert "String should have at most 256 characters" in response.text

def test_store_metadata_too_large():
    with TestClient(app) as client:
        many_metadata = {f"k{i}": "v" for i in range(51)}
        response = client.post(
            "/store",
            json={"content": "foo", "metadata": many_metadata},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 422
        assert "Too many metadata keys" in response.text

def test_store_metadata_key_too_long():
    with TestClient(app) as client:
        long_key = "k" * 65
        response = client.post(
            "/store",
            json={"content": "foo", "metadata": {long_key: "val"}},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 422
        assert f"Metadata key {long_key} too long" in response.text
