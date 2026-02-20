"""
API Security Limits Tests
========================
Comprehensive tests for input validation and rate limiting.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure path is set
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemocore.core.config import reset_config

API_KEY = "test-key"

# Setup mocks before importing app
mock_engine_cls = MagicMock()
mock_engine_instance = MagicMock()
mock_engine_instance.get_stats = AsyncMock(return_value={"status": "ok"})
mock_engine_instance.get_memory = AsyncMock(return_value=None)
mock_engine_instance.delete_memory = AsyncMock(return_value=True)
mock_engine_instance.store = AsyncMock(return_value="mem_id_123")
mock_engine_instance.query = AsyncMock(return_value=[("mem_id_123", 0.9)])
mock_engine_instance.initialize = AsyncMock(return_value=None)
mock_engine_instance.close = AsyncMock(return_value=None)
mock_engine_instance.define_concept = AsyncMock(return_value=None)
mock_engine_instance.reason_by_analogy = AsyncMock(return_value=[("result1", 0.8)])
mock_engine_cls.return_value = mock_engine_instance

# Mock container
mock_container = MagicMock()
mock_container.redis_storage = AsyncMock()
mock_container.redis_storage.check_health = AsyncMock(return_value=True)
mock_container.redis_storage.store_memory = AsyncMock()
mock_container.redis_storage.publish_event = AsyncMock()
mock_container.redis_storage.retrieve_memory = AsyncMock(return_value=None)
mock_container.redis_storage.delete_memory = AsyncMock()
mock_container.redis_storage.close = AsyncMock()
mock_container.qdrant_store = MagicMock()

# Setup pipeline mock
mock_pipeline = MagicMock()
mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
mock_pipeline.__aexit__ = AsyncMock(return_value=None)
mock_pipeline.incr = MagicMock()
mock_pipeline.expire = MagicMock()
mock_pipeline.execute = AsyncMock(return_value=[1, True])

mock_redis_client = MagicMock()
mock_redis_client.pipeline.return_value = mock_pipeline
mock_container.redis_storage.redis_client = mock_redis_client

# Patch before import
patcher1 = patch("mnemocore.api.main.HAIMEngine", mock_engine_cls)
patcher2 = patch("mnemocore.api.main.build_container", return_value=mock_container)
patcher1.start()
patcher2.start()

from mnemocore.api.main import app


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("HAIM_API_KEY", API_KEY)
    reset_config()
    # Mock app state
    app.state.engine = mock_engine_instance
    app.state.container = mock_container
    yield
    reset_config()


# ============================================================================
# INPUT VALIDATION TESTS - Store Endpoint
# ============================================================================


def test_store_content_too_large():
    """Verify that content larger than 100,000 chars is rejected."""
    with TestClient(app) as client:
        large_content = "a" * 100001
        response = client.post(
            "/store", json={"content": large_content}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 422
        assert "String should have at most 100000 characters" in response.text


def test_store_content_valid():
    """Verify that content within limit is accepted."""
    mock_memory = MagicMock(
        id="mem_1",
        content="a" * 1000,
        metadata={},
        ltp_strength=0.5,
        created_at=MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:00:00")),
    )
    mock_engine_instance.get_memory.return_value = mock_memory
    mock_engine_instance.store.return_value = "mem_1"

    with TestClient(app) as client:
        valid_content = "a" * 1000
        response = client.post(
            "/store", json={"content": valid_content}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 200


def test_store_content_empty():
    """Verify that empty content is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/store", json={"content": ""}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 422


def test_store_content_whitespace_only():
    """Verify that whitespace-only content is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/store", json={"content": "   \n\t   "}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 422


def test_store_metadata_too_many_keys():
    """Verify that metadata with too many keys is rejected."""
    with TestClient(app) as client:
        many_metadata = {f"k{i}": "v" for i in range(51)}
        response = client.post(
            "/store",
            json={"content": "foo", "metadata": many_metadata},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422
        assert "Too many metadata keys" in response.text


def test_store_metadata_key_too_long():
    """Verify that metadata key longer than 64 chars is rejected."""
    with TestClient(app) as client:
        long_key = "k" * 65
        response = client.post(
            "/store",
            json={"content": "foo", "metadata": {long_key: "val"}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422
        assert "too long" in response.text


def test_store_metadata_value_too_long():
    """Verify that metadata value longer than 1000 chars is rejected."""
    with TestClient(app) as client:
        long_value = "v" * 1001
        response = client.post(
            "/store",
            json={"content": "foo", "metadata": {"key": long_value}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422
        assert "too long" in response.text


def test_store_metadata_invalid_key_characters():
    """Verify that metadata key with invalid characters is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/store",
            json={"content": "foo", "metadata": {"key$invalid": "val"}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422
        assert "invalid characters" in response.text


def test_store_metadata_nested_structure():
    """Verify that nested metadata values are rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/store",
            json={"content": "foo", "metadata": {"nested": {"key": "value"}}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422
        assert "primitive type" in response.text


def test_store_agent_id_too_long():
    """Verify that agent_id longer than 256 chars is rejected."""
    with TestClient(app) as client:
        long_agent_id = "a" * 257
        response = client.post(
            "/store",
            json={"content": "foo", "agent_id": long_agent_id},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


def test_store_agent_id_invalid_characters():
    """Verify that agent_id with invalid characters is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/store",
            json={"content": "foo", "agent_id": "agent$invalid"},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


def test_store_ttl_out_of_range():
    """Verify that TTL outside valid range is rejected."""
    with TestClient(app) as client:
        # TTL too small
        response = client.post(
            "/store", json={"content": "foo", "ttl": 0}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 422

        # TTL too large (> 1 year)
        response = client.post(
            "/store",
            json={"content": "foo", "ttl": 86400 * 365 + 1},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


# ============================================================================
# INPUT VALIDATION TESTS - Query Endpoint
# ============================================================================


def test_query_too_large():
    """Verify query string limits."""
    with TestClient(app) as client:
        large_query = "q" * 20000
        response = client.post(
            "/query", json={"query": large_query}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 422
        assert "String should have at most 10000 characters" in response.text


def test_query_valid():
    """Verify query within limit is accepted."""
    # Setup mock memory with tier attribute
    mock_memory = MagicMock(
        id="mem_1",
        content="test result",
        metadata={},
        ltp_strength=0.5,
        created_at=MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:00:00")),
        tier="hot",
    )
    mock_engine_instance.get_memory.return_value = mock_memory
    mock_engine_instance.query.return_value = [("mem_1", 0.9)]

    with TestClient(app) as client:
        valid_query = "hello world"
        response = client.post(
            "/query", json={"query": valid_query}, headers={"X-API-Key": API_KEY}
        )
        # It might return 200 or 500 depending on engine state, but NOT 422
        assert response.status_code != 422


def test_query_empty():
    """Verify that empty query is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/query", json={"query": ""}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 422


def test_query_whitespace_only():
    """Verify that whitespace-only query is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/query", json={"query": "   \n\t   "}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 422


def test_query_top_k_out_of_range():
    """Verify that top_k outside valid range is rejected."""
    with TestClient(app) as client:
        # top_k too small
        response = client.post(
            "/query", json={"query": "test", "top_k": 0}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 422

        # top_k too large
        response = client.post(
            "/query",
            json={"query": "test", "top_k": 101},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


# ============================================================================
# INPUT VALIDATION TESTS - Concept Endpoint
# ============================================================================


def test_concept_name_too_large():
    """Verify concept name limit."""
    with TestClient(app) as client:
        large_name = "n" * 10000
        response = client.post(
            "/concept",
            json={"name": large_name, "attributes": {"key": "value"}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422
        assert "String should have at most 256 characters" in response.text


def test_concept_name_empty():
    """Verify that empty concept name is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/concept",
            json={"name": "", "attributes": {"key": "value"}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


def test_concept_name_invalid_characters():
    """Verify that concept name with invalid characters is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/concept",
            json={"name": "concept$invalid", "attributes": {"key": "value"}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


def test_concept_attributes_too_many():
    """Verify that too many attributes are rejected."""
    with TestClient(app) as client:
        many_attributes = {f"k{i}": "v" for i in range(51)}
        response = client.post(
            "/concept",
            json={"name": "test", "attributes": many_attributes},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422
        assert "Too many attributes" in response.text


def test_concept_attributes_empty():
    """Verify that empty attributes are rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/concept",
            json={"name": "test", "attributes": {}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


def test_concept_attribute_key_too_long():
    """Verify that attribute key longer than 64 chars is rejected."""
    with TestClient(app) as client:
        long_key = "k" * 65
        response = client.post(
            "/concept",
            json={"name": "test", "attributes": {long_key: "val"}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422
        assert "too long" in response.text


def test_concept_attribute_key_invalid_characters():
    """Verify that attribute key with invalid characters is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/concept",
            json={"name": "test", "attributes": {"key$invalid": "val"}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


def test_concept_attribute_value_too_long():
    """Verify that attribute value longer than 1000 chars is rejected."""
    with TestClient(app) as client:
        long_value = "v" * 1001
        response = client.post(
            "/concept",
            json={"name": "test", "attributes": {"key": long_value}},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


# ============================================================================
# INPUT VALIDATION TESTS - Analogy Endpoint
# ============================================================================


def test_analogy_source_concept_too_large():
    """Verify analogy source concept limit."""
    with TestClient(app) as client:
        large_str = "a" * 10000
        response = client.post(
            "/analogy",
            json={
                "source_concept": large_str,
                "source_value": "val",
                "target_concept": "target",
            },
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422
        assert "String should have at most 256 characters" in response.text


def test_analogy_empty_concept():
    """Verify that empty concept is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/analogy",
            json={
                "source_concept": "",
                "source_value": "val",
                "target_concept": "target",
            },
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


def test_analogy_empty_value():
    """Verify that empty value is rejected."""
    with TestClient(app) as client:
        response = client.post(
            "/analogy",
            json={
                "source_concept": "source",
                "source_value": "",
                "target_concept": "target",
            },
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


def test_analogy_target_concept_too_large():
    """Verify analogy target concept limit."""
    with TestClient(app) as client:
        large_str = "a" * 10000
        response = client.post(
            "/analogy",
            json={
                "source_concept": "source",
                "source_value": "val",
                "target_concept": large_str,
            },
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


# ============================================================================
# RATE LIMITING TESTS - Store (100/minute)
# ============================================================================


def test_store_rate_limiter_within_limit():
    """Verify store requests within limit succeed."""
    # Ensure pipeline execute returns count < limit (100 for store)
    mock_pipeline.execute.return_value = [1, True]

    mock_memory = MagicMock(
        id="mem_1",
        content="test",
        metadata={},
        ltp_strength=0.5,
        created_at=MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:00:00")),
    )
    mock_engine_instance.get_memory.return_value = mock_memory
    mock_engine_instance.store.return_value = "mem_1"

    with TestClient(app) as client:
        response = client.post(
            "/store", json={"content": "test"}, headers={"X-API-Key": API_KEY}
        )

        assert response.status_code == 200
        assert response.json()["ok"] is True


def test_store_rate_limiter_exceeded():
    """Verify store rate limit returns 429 with Retry-After header."""
    # Simulate return value [count=101, expire_result=True] (Limit is 100 for store)
    mock_pipeline.execute.return_value = [101, True]

    with TestClient(app) as client:
        response = client.post(
            "/store", json={"content": "test"}, headers={"X-API-Key": API_KEY}
        )

        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert "Retry-After" in response.headers


def test_store_rate_limiter_retry_after_value():
    """Verify Retry-After header contains valid seconds."""
    mock_pipeline.execute.return_value = [101, True]

    with TestClient(app) as client:
        response = client.post(
            "/store", json={"content": "test"}, headers={"X-API-Key": API_KEY}
        )

        assert response.status_code == 429
        retry_after = response.headers.get("Retry-After")
        assert retry_after is not None
        # Should be a positive integer
        assert int(retry_after) > 0
        assert int(retry_after) <= 60  # Max window size


# ============================================================================
# RATE LIMITING TESTS - Query (500/minute)
# ============================================================================


def test_query_rate_limiter_within_limit():
    """Verify query requests within limit succeed."""
    # Ensure pipeline execute returns count < limit (500 for query)
    mock_pipeline.execute.return_value = [100, True]

    # Setup mock memory with tier attribute
    mock_memory = MagicMock(
        id="mem_1",
        content="test result",
        metadata={},
        ltp_strength=0.5,
        created_at=MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:00:00")),
        tier="hot",
    )
    mock_engine_instance.get_memory.return_value = mock_memory
    mock_engine_instance.query.return_value = [("mem_1", 0.9)]

    with TestClient(app) as client:
        response = client.post(
            "/query", json={"query": "test"}, headers={"X-API-Key": API_KEY}
        )

        # Should not be 429
        assert response.status_code != 429


def test_query_rate_limiter_exceeded():
    """Verify query rate limit returns 429 with Retry-After header."""
    # Simulate return value [count=501, expire_result=True] (Limit is 500 for query)
    mock_pipeline.execute.return_value = [501, True]

    with TestClient(app) as client:
        response = client.post(
            "/query", json={"query": "test"}, headers={"X-API-Key": API_KEY}
        )

        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert "Retry-After" in response.headers


# ============================================================================
# RATE LIMITING TESTS - Concept (100/minute)
# ============================================================================


def test_concept_rate_limiter_within_limit():
    """Verify concept requests within limit succeed."""
    mock_pipeline.execute.return_value = [50, True]

    with TestClient(app) as client:
        response = client.post(
            "/concept",
            json={"name": "test", "attributes": {"key": "value"}},
            headers={"X-API-Key": API_KEY},
        )

        assert response.status_code != 429


def test_concept_rate_limiter_exceeded():
    """Verify concept rate limit returns 429."""
    mock_pipeline.execute.return_value = [101, True]

    with TestClient(app) as client:
        response = client.post(
            "/concept",
            json={"name": "test", "attributes": {"key": "value"}},
            headers={"X-API-Key": API_KEY},
        )

        assert response.status_code == 429


# ============================================================================
# RATE LIMITING TESTS - Analogy (100/minute)
# ============================================================================


def test_analogy_rate_limiter_within_limit():
    """Verify analogy requests within limit succeed."""
    mock_pipeline.execute.return_value = [50, True]

    with TestClient(app) as client:
        response = client.post(
            "/analogy",
            json={
                "source_concept": "source",
                "source_value": "val",
                "target_concept": "target",
            },
            headers={"X-API-Key": API_KEY},
        )

        assert response.status_code != 429


def test_analogy_rate_limiter_exceeded():
    """Verify analogy rate limit returns 429."""
    mock_pipeline.execute.return_value = [101, True]

    with TestClient(app) as client:
        response = client.post(
            "/analogy",
            json={
                "source_concept": "source",
                "source_value": "val",
                "target_concept": "target",
            },
            headers={"X-API-Key": API_KEY},
        )

        assert response.status_code == 429


# ============================================================================
# RATE LIMITING - Differentiated Limits Per Category
# ============================================================================


def test_rate_limit_different_categories():
    """Verify that different endpoints have different rate limits."""
    with TestClient(app) as client:
        # Get rate limit configuration
        response = client.get("/rate-limits")
        assert response.status_code == 200

        limits = response.json()["limits"]

        # Store: 100/min
        assert limits["store"]["requests"] == 100
        assert limits["store"]["window_seconds"] == 60

        # Query: 500/min
        assert limits["query"]["requests"] == 500
        assert limits["query"]["window_seconds"] == 60

        # Concept: 100/min
        assert limits["concept"]["requests"] == 100
        assert limits["concept"]["window_seconds"] == 60

        # Analogy: 100/min
        assert limits["analogy"]["requests"] == 100
        assert limits["analogy"]["window_seconds"] == 60


def test_rate_limit_x_forwarded_for():
    """Verify that X-Forwarded-For header is used for client IP."""
    mock_pipeline.execute.return_value = [1, True]

    mock_memory = MagicMock(
        id="mem_1",
        content="test",
        metadata={},
        ltp_strength=0.5,
        created_at=MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:00:00")),
    )
    mock_engine_instance.get_memory.return_value = mock_memory
    mock_engine_instance.store.return_value = "mem_1"

    with TestClient(app) as client:
        response = client.post(
            "/store",
            json={"content": "test"},
            headers={"X-API-Key": API_KEY, "X-Forwarded-For": "10.0.0.1, 192.168.1.1"},
        )

        # Should succeed (rate limit check should pass)
        assert response.status_code == 200


# ============================================================================
# EDGE CASES - Memory ID Validation
# ============================================================================


def test_get_memory_invalid_id_empty():
    """Verify that empty memory_id is rejected."""
    with TestClient(app) as client:
        response = client.get("/memory/", headers={"X-API-Key": API_KEY})
        # Should return 404 or 405, not 500
        assert response.status_code in [404, 405]


def test_get_memory_invalid_id_too_long():
    """Verify that memory_id longer than 256 chars is rejected."""
    with TestClient(app) as client:
        long_id = "a" * 300
        response = client.get(f"/memory/{long_id}", headers={"X-API-Key": API_KEY})
        assert response.status_code == 400


def test_delete_memory_invalid_id_too_long():
    """Verify that memory_id longer than 256 chars is rejected for delete."""
    with TestClient(app) as client:
        long_id = "a" * 300
        response = client.delete(f"/memory/{long_id}", headers={"X-API-Key": API_KEY})
        assert response.status_code == 400
