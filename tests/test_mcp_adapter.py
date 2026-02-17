import requests

from src.mcp.adapters.api_adapter import MnemoCoreAPIAdapter, MnemoCoreAPIError


class DummyResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = text

    def json(self):
        return self._data


def test_adapter_success(monkeypatch):
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
    def fake_request(method, url, json, headers, timeout):
        raise requests.RequestException("timeout")

    monkeypatch.setattr(requests, "request", fake_request)

    adapter = MnemoCoreAPIAdapter("http://localhost:8100", "key")

    try:
        adapter.stats()
        assert False, "Expected MnemoCoreAPIError"
    except MnemoCoreAPIError as exc:
        assert "Upstream request failed" in str(exc)
