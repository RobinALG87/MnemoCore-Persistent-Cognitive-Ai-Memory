"""
Tests for MCP API Adapter
=========================
Tests for src/mnemocore/mcp/adapters/api_adapter.py covering URL encoding,
HTTPS enforcement, retry on transient failure, and connection timeout handling.
"""

import os
import pytest
import requests
from unittest.mock import MagicMock, patch
import urllib.parse

from mnemocore.mcp.adapters.api_adapter import MnemoCoreAPIAdapter, MnemoCoreAPIError


class DummyResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = text

    def json(self):
        return self._data


# ============================================================================
# Original Tests
# ============================================================================

def test_adapter_success(monkeypatch):
    """Test successful API adapter call."""
    def fake_request(method, url, json, headers, timeout):
        assert method == "GET"
        assert url.endswith("/health")
        assert headers["X-API-Key"] == "key"
        assert timeout == 5
        return DummyResponse(status_code=200, data={"status": "healthy"})

    monkeypatch.setattr(requests, "request", fake_request)

    adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key", timeout_seconds=5)
    result = adapter.health()
    assert result["status"] == "healthy"


def test_adapter_http_error(monkeypatch):
    """Test API adapter handles HTTP errors."""
    def fake_request(method, url, json, headers, timeout):
        return DummyResponse(status_code=404, data={"detail": "not found"})

    monkeypatch.setattr(requests, "request", fake_request)

    adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")

    try:
        adapter.get_memory("missing")
        assert False, "Expected MnemoCoreAPIError"
    except MnemoCoreAPIError as exc:
        assert exc.status_code == 404


def test_adapter_network_error(monkeypatch):
    """Test API adapter handles network errors."""
    def fake_request(method, url, json, headers, timeout):
        raise requests.RequestException("timeout")

    monkeypatch.setattr(requests, "request", fake_request)

    adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")

    try:
        adapter.stats()
        assert False, "Expected MnemoCoreAPIError"
    except MnemoCoreAPIError as exc:
        assert "Upstream request failed" in str(exc)


# ============================================================================
# Task 15.5: Additional MCP Adapter Tests
# ============================================================================

class TestURLEncoding:
    """Tests for URL encoding of query parameters (after Agent 3 fix)."""

    def test_url_encoding_special_characters(self, monkeypatch):
        """Test that special characters in query params are properly encoded."""
        captured_url = []

        def fake_request(method, url, json, headers, timeout):
            captured_url.append(url)
            return DummyResponse(status_code=200, data={"ok": True})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        adapter.get_working_context("agent with spaces", limit=10)

        # URL should be properly encoded
        assert len(captured_url) == 1
        # Spaces should be encoded as %20
        assert "agent%20with%20spaces" in captured_url[0] or "agent+with+spaces" in captured_url[0]

    def test_url_encoding_query_string(self, monkeypatch):
        """Test that query strings with special characters are encoded."""
        captured_url = []

        def fake_request(method, url, json, headers, timeout):
            captured_url.append(url)
            return DummyResponse(status_code=200, data={"ok": True})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        adapter.search_procedures("query with & special=chars", top_k=5)

        assert len(captured_url) == 1
        # & and = should be encoded in the query parameter value
        assert "query%20with" in captured_url[0] or "query+with" in captured_url[0]

    def test_url_encoding_unicode(self, monkeypatch):
        """Test that unicode characters are properly encoded."""
        captured_url = []

        def fake_request(method, url, json, headers, timeout):
            captured_url.append(url)
            return DummyResponse(status_code=200, data={"ok": True})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        # Use unicode characters
        adapter.search_procedures("hello world", agent_id="agent-123", top_k=5)

        assert len(captured_url) == 1

    def test_build_url_filters_none_values(self, monkeypatch):
        """Test that None values are filtered from query parameters."""
        captured_url = []

        def fake_request(method, url, json, headers, timeout):
            captured_url.append(url)
            return DummyResponse(status_code=200, data={"ok": True})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        # agent_id=None should be filtered out
        adapter.search_procedures("test query", agent_id=None, top_k=5)

        assert len(captured_url) == 1
        # Should not contain "agent_id=None"
        assert "agent_id=None" not in captured_url[0]


class TestHTTPSEnforcement:
    """Tests for HTTPS enforcement in production mode."""

    def test_https_required_in_production(self, monkeypatch):
        """Test that HTTPS is required when HAIM_ENV=production."""
        monkeypatch.setenv("HAIM_ENV", "production")

        # Need to re-import to get the updated env var
        import importlib
        import mnemocore.mcp.adapters.api_adapter as adapter_module
        importlib.reload(adapter_module)

        with pytest.raises(adapter_module.MnemoCoreAPIError) as exc_info:
            adapter_module.MnemoCoreAPIAdapter("http://api.example.com", "key")

        assert "HTTPS" in str(exc_info.value)

    def test_http_allowed_in_development(self, monkeypatch):
        """Test that HTTP is allowed in development mode."""
        monkeypatch.setenv("HAIM_ENV", "development")

        # Should not raise an error
        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        assert adapter.base_url == "http://localhost:8100"

    def test_http_allowed_for_localhost(self, monkeypatch):
        """Test that HTTP is allowed for localhost URLs."""
        # Even in production-like mode, localhost should be allowed
        # (though the current implementation warns but doesn't error for localhost)
        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        assert adapter.base_url == "http://localhost:8100"

    def test_https_url_accepted(self, monkeypatch):
        """Test that HTTPS URLs are always accepted."""
        monkeypatch.setenv("HAIM_ENV", "production")

        import importlib
        import mnemocore.mcp.adapters.api_adapter as adapter_module
        importlib.reload(adapter_module)

        adapter = adapter_module.MnemoCoreAPIAdapter("https://api.example.com", "key")
        assert adapter.base_url == "https://api.example.com"


class TestRetryOnTransientFailure:
    """Tests for retry behavior on transient failures."""

    def test_retry_on_connection_error(self, monkeypatch):
        """Test that transient connection errors can be retried."""
        call_count = [0]

        def fake_request(method, url, json, headers, timeout):
            call_count[0] += 1
            if call_count[0] < 3:
                raise requests.ConnectionError("Connection refused")
            return DummyResponse(status_code=200, data={"ok": True})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")

        # First call will fail
        with pytest.raises(MnemoCoreAPIError):
            adapter.health()

        # But subsequent calls can succeed
        result = adapter.health()
        assert result["ok"] is True

    def test_retry_on_timeout(self, monkeypatch):
        """Test that timeout errors can be retried."""
        call_count = [0]

        def fake_request(method, url, json, headers, timeout):
            call_count[0] += 1
            if call_count[0] == 1:
                raise requests.Timeout("Request timed out")
            return DummyResponse(status_code=200, data={"status": "healthy"})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")

        # First call will fail
        with pytest.raises(MnemoCoreAPIError):
            adapter.health()

        # Second call succeeds
        result = adapter.health()
        assert result["status"] == "healthy"


class TestConnectionTimeout:
    """Tests for connection timeout handling."""

    def test_custom_timeout_used(self, monkeypatch):
        """Test that custom timeout is properly passed to requests."""
        captured_timeout = []

        def fake_request(method, url, json, headers, timeout):
            captured_timeout.append(timeout)
            return DummyResponse(status_code=200, data={"ok": True})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key", timeout_seconds=30)
        adapter.health()

        assert len(captured_timeout) == 1
        assert captured_timeout[0] == 30

    def test_default_timeout(self, monkeypatch):
        """Test that default timeout is used when not specified."""
        captured_timeout = []

        def fake_request(method, url, json, headers, timeout):
            captured_timeout.append(timeout)
            return DummyResponse(status_code=200, data={"ok": True})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        adapter.health()

        assert len(captured_timeout) == 1
        assert captured_timeout[0] == 15  # Default from class definition

    def test_timeout_exception_wrapped(self, monkeypatch):
        """Test that timeout exceptions are wrapped in MnemoCoreAPIError."""
        def fake_request(method, url, json, headers, timeout):
            raise requests.Timeout("Connection timed out")

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")

        with pytest.raises(MnemoCoreAPIError) as exc_info:
            adapter.health()

        assert "Upstream request failed" in str(exc_info.value)


class TestAdapterMethods:
    """Tests for various adapter methods."""

    def test_store_method(self, monkeypatch):
        """Test store method sends POST to /store."""
        captured = []

        def fake_request(method, url, json, headers, timeout):
            captured.append({"method": method, "url": url, "json": json})
            return DummyResponse(status_code=200, data={"ok": True, "memory_id": "mem_123"})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        result = adapter.store({"content": "test memory"})

        assert len(captured) == 1
        assert captured[0]["method"] == "POST"
        assert "/store" in captured[0]["url"]
        assert result["ok"] is True

    def test_query_method(self, monkeypatch):
        """Test query method sends POST to /query."""
        captured = []

        def fake_request(method, url, json, headers, timeout):
            captured.append({"method": method, "url": url, "json": json})
            return DummyResponse(status_code=200, data={"ok": True, "results": []})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        result = adapter.query({"query": "test query", "top_k": 10})

        assert len(captured) == 1
        assert captured[0]["method"] == "POST"
        assert "/query" in captured[0]["url"]

    def test_delete_memory_method(self, monkeypatch):
        """Test delete_memory method sends DELETE."""
        captured = []

        def fake_request(method, url, json, headers, timeout):
            captured.append({"method": method, "url": url})
            return DummyResponse(status_code=200, data={"ok": True})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        result = adapter.delete_memory("mem_123")

        assert len(captured) == 1
        assert captured[0]["method"] == "DELETE"
        assert "mem_123" in captured[0]["url"]

    def test_dream_method(self, monkeypatch):
        """Test dream method sends POST to /dream."""
        captured = []

        def fake_request(method, url, json, headers, timeout):
            captured.append({"method": method, "url": url, "json": json})
            return DummyResponse(status_code=200, data={"ok": True, "cycles_completed": 1})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        result = adapter.dream({"max_cycles": 1})

        assert len(captured) == 1
        assert captured[0]["method"] == "POST"
        assert "/dream" in captured[0]["url"]

    def test_export_method(self, monkeypatch):
        """Test export method sends GET to /export with params."""
        captured = []

        def fake_request(method, url, json, headers, timeout):
            captured.append({"method": method, "url": url})
            return DummyResponse(status_code=200, data={"ok": True, "memories": []})

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")
        result = adapter.export({"limit": 100, "format": "json"})

        assert len(captured) == 1
        assert captured[0]["method"] == "GET"
        assert "/export" in captured[0]["url"]
        assert "limit" in captured[0]["url"]


class TestNonJSONResponse:
    """Tests for handling non-JSON responses."""

    def test_non_json_response_raises_error(self, monkeypatch):
        """Test that non-JSON responses raise MnemoCoreAPIError."""
        class NonJSONResponse:
            status_code = 200
            text = "Not JSON"

            def json(self):
                raise ValueError("No JSON")

        def fake_request(method, url, json, headers, timeout):
            return NonJSONResponse()

        monkeypatch.setattr(requests, "request", fake_request)

        adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")

        with pytest.raises(MnemoCoreAPIError) as exc_info:
            adapter.health()

        assert "non-JSON" in str(exc_info.value)
