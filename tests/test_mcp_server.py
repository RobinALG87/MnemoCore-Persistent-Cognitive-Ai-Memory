import sys
import types

import pytest

from mnemocore.core.config import HAIMConfig, MCPConfig, SecurityConfig
from mnemocore.mcp import server as mcp_server
from mnemocore.mcp.adapters.api_adapter import MnemoCoreAPIError


class FakeFastMCP:
    def __init__(self, name: str):
        self.name = name
        self.tools = {}
        self.run_calls = []

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator

    def run(self, **kwargs):
        self.run_calls.append(kwargs)


class FakeAdapter:
    def store(self, payload):
        return {"memory_id": "mem_1", "payload": payload}

    def query(self, payload):
        return {"results": [{"id": "mem_1", "score": 1.0}], "payload": payload}

    def get_memory(self, memory_id: str):
        return {"id": memory_id, "content": "hello"}

    def delete_memory(self, memory_id: str):
        return {"deleted": memory_id}

    def stats(self):
        return {"engine_version": "3.5.1"}

    def health(self):
        return {"status": "healthy"}


def _install_fake_mcp_modules(monkeypatch):
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = FakeFastMCP

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.server", server_mod)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)


def test_build_server_registers_only_allowlisted_tools(monkeypatch):
    _install_fake_mcp_modules(monkeypatch)
    monkeypatch.setattr(
        mcp_server, "MnemoCoreAPIAdapter", lambda *args, **kwargs: FakeAdapter()
    )

    config = HAIMConfig(
        security=SecurityConfig(api_key="test-key"),
        mcp=MCPConfig(
            enabled=True,
            allow_tools=["memory_health", "memory_stats"],
            api_key="test-key",
        ),
    )

    server = mcp_server.build_server(config)
    assert sorted(server.tools.keys()) == ["memory_health", "memory_stats"]

    health_result = server.tools["memory_health"]()
    assert health_result["ok"] is True
    assert health_result["data"]["status"] == "healthy"


def test_tool_error_handling(monkeypatch):
    class ErrorAdapter(FakeAdapter):
        def health(self):
            raise MnemoCoreAPIError("boom", status_code=503)

    _install_fake_mcp_modules(monkeypatch)
    monkeypatch.setattr(
        mcp_server, "MnemoCoreAPIAdapter", lambda *args, **kwargs: ErrorAdapter()
    )

    config = HAIMConfig(
        security=SecurityConfig(api_key="test-key"),
        mcp=MCPConfig(enabled=True, allow_tools=["memory_health"], api_key="test-key"),
    )

    server = mcp_server.build_server(config)
    result = server.tools["memory_health"]()

    assert result["ok"] is False
    assert "boom" in result["error"]


def test_main_runs_with_stdio_transport(monkeypatch):
    fake_server = FakeFastMCP("x")

    monkeypatch.setattr(
        mcp_server,
        "get_config",
        lambda: HAIMConfig(
            security=SecurityConfig(api_key="k"),
            mcp=MCPConfig(enabled=True, transport="stdio", api_key="k"),
        ),
    )
    monkeypatch.setattr(mcp_server, "build_server", lambda cfg: fake_server)

    mcp_server.main()
    assert fake_server.run_calls == [{"transport": "stdio"}]


def test_main_runs_with_sse_transport(monkeypatch):
    fake_server = FakeFastMCP("x")

    monkeypatch.setattr(
        mcp_server,
        "get_config",
        lambda: HAIMConfig(
            security=SecurityConfig(api_key="k"),
            mcp=MCPConfig(
                enabled=True,
                transport="sse",
                host="127.0.0.1",
                port=8222,
                api_key="k",
            ),
        ),
    )
    monkeypatch.setattr(mcp_server, "build_server", lambda cfg: fake_server)

    mcp_server.main()
    assert fake_server.run_calls == [
        {"transport": "sse", "host": "127.0.0.1", "port": 8222}
    ]


def test_main_rejects_unknown_transport(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "get_config",
        lambda: HAIMConfig(
            security=SecurityConfig(api_key="k"),
            mcp=MCPConfig(enabled=True, transport="unknown", api_key="k"),
        ),
    )
    monkeypatch.setattr(mcp_server, "build_server", lambda cfg: FakeFastMCP("x"))

    with pytest.raises((ValueError, Exception), match="Unsupported transport"):
        mcp_server.main()
