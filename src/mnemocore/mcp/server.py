"""
MnemoCore MCP Server
====================
MCP bridge exposing MnemoCore API tools for agent clients.
"""

from typing import Any, Callable, Dict

from loguru import logger

from mnemocore.core.config import HAIMConfig, get_config
from mnemocore.core.exceptions import (
    DependencyMissingError,
    UnsupportedTransportError,
)
from mnemocore.mcp.adapters.api_adapter import MnemoCoreAPIAdapter, MnemoCoreAPIError
from mnemocore.mcp.schemas import MemoryIdInput, QueryToolInput, StoreToolInput


def _result_ok(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "data": data}


def _result_error(message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message}


def build_server(config: HAIMConfig | None = None):
    cfg = config or get_config()

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise DependencyMissingError(
            dependency="mcp", message="Install package 'mcp' to run the MCP server."
        ) from exc

    adapter = MnemoCoreAPIAdapter(
        base_url=cfg.mcp.api_base_url,
        api_key=cfg.mcp.api_key or cfg.security.api_key,
        timeout_seconds=cfg.mcp.timeout_seconds,
    )

    server = FastMCP("MnemoCore MCP")
    allow_tools = set(cfg.mcp.allow_tools)

    def register_tool(name: str, fn: Callable[[], None]) -> None:
        if name in allow_tools:
            fn()
        else:
            logger.info("Skipping disabled MCP tool: %s", name)

    def with_error_handling(call: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return _result_ok(call())
        except MnemoCoreAPIError as exc:
            return _result_error(str(exc))
        except Exception as exc:
            return _result_error(f"Unexpected error: {exc}")

    def register_memory_store() -> None:
        @server.tool()
        def memory_store(
            content: str,
            metadata: Dict[str, Any] | None = None,
            agent_id: str | None = None,
            ttl: int | None = None,
        ) -> Dict[str, Any]:
            payload = StoreToolInput(
                content=content,
                metadata=metadata,
                agent_id=agent_id,
                ttl=ttl,
            ).model_dump(exclude_none=True)
            return with_error_handling(lambda: adapter.store(payload))

    def register_memory_query() -> None:
        @server.tool()
        def memory_query(
            query: str,
            top_k: int = 5,
            agent_id: str | None = None,
        ) -> Dict[str, Any]:
            payload = QueryToolInput(
                query=query,
                top_k=top_k,
                agent_id=agent_id,
            ).model_dump(exclude_none=True)
            return with_error_handling(lambda: adapter.query(payload))

    def register_memory_get() -> None:
        @server.tool()
        def memory_get(memory_id: str) -> Dict[str, Any]:
            data = MemoryIdInput(memory_id=memory_id)
            return with_error_handling(lambda: adapter.get_memory(data.memory_id))

    def register_memory_delete() -> None:
        @server.tool()
        def memory_delete(memory_id: str) -> Dict[str, Any]:
            data = MemoryIdInput(memory_id=memory_id)
            return with_error_handling(lambda: adapter.delete_memory(data.memory_id))

    def register_memory_stats() -> None:
        @server.tool()
        def memory_stats() -> Dict[str, Any]:
            return with_error_handling(adapter.stats)

    def register_memory_health() -> None:
        @server.tool()
        def memory_health() -> Dict[str, Any]:
            return with_error_handling(adapter.health)

    register_tool("memory_store", register_memory_store)
    register_tool("memory_query", register_memory_query)
    register_tool("memory_get", register_memory_get)
    register_tool("memory_delete", register_memory_delete)
    register_tool("memory_stats", register_memory_stats)
    register_tool("memory_health", register_memory_health)

    return server


def main() -> None:
    cfg = get_config()
    if not cfg.mcp.enabled:
        logger.warning("MCP is disabled in config (haim.mcp.enabled=false)")

    server = build_server(cfg)

    if cfg.mcp.transport == "stdio":
        server.run(transport="stdio")
        return

    if cfg.mcp.transport == "sse":
        server.run(transport="sse", host=cfg.mcp.host, port=cfg.mcp.port)
        return

    raise UnsupportedTransportError(
        transport=cfg.mcp.transport, supported_transports=["stdio", "sse"]
    )


if __name__ == "__main__":
    main()
