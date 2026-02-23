"""
MnemoCore MCP Server
====================
MCP bridge exposing MnemoCore API tools for agent clients.
"""

from typing import Any, Callable, Dict
from loguru import logger

from mnemocore.core.config import get_config, HAIMConfig
from mnemocore.mcp.adapters.api_adapter import MnemoCoreAPIAdapter, MnemoCoreAPIError
from mnemocore.mcp.schemas import (
    StoreToolInput, QueryToolInput, MemoryIdInput,
    ObserveToolInput, ContextToolInput, EpisodeToolInput
)
from mnemocore.core.exceptions import (
    DependencyMissingError,
    UnsupportedTransportError,
)


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
            dependency="mcp",
            message="Install package 'mcp' to run the MCP server."
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

    # --- Phase 5: Cognitive Client Tools ---

    def register_store_observation() -> None:
        @server.tool()
        def store_observation(
            agent_id: str,
            content: str,
            kind: str = "observation",
            importance: float = 0.5,
            tags: list[str] | None = None
        ) -> Dict[str, Any]:
            payload = ObserveToolInput(
                agent_id=agent_id, content=content, kind=kind, importance=importance, tags=tags
            ).model_dump(exclude_none=True)
            return with_error_handling(lambda: adapter.observe_context(payload))

    def register_recall_context() -> None:
        @server.tool()
        def recall_context(agent_id: str, limit: int = 16) -> Dict[str, Any]:
            data = ContextToolInput(agent_id=agent_id, limit=limit)
            return with_error_handling(lambda: adapter.get_working_context(data.agent_id, data.limit))

    def register_start_episode() -> None:
        @server.tool()
        def start_episode(agent_id: str, goal: str, context: str | None = None) -> Dict[str, Any]:
            payload = EpisodeToolInput(
                agent_id=agent_id, goal=goal, context=context
            ).model_dump(exclude_none=True)
            return with_error_handling(lambda: adapter.start_episode(payload))

    def register_get_knowledge_gaps() -> None:
        @server.tool()
        def get_knowledge_gaps() -> Dict[str, Any]:
            return with_error_handling(adapter.get_knowledge_gaps)

    def register_get_subtle_thoughts() -> None:
        @server.tool()
        def get_subtle_thoughts(agent_id: str, limit: int = 5) -> Dict[str, Any]:
            return with_error_handling(lambda: adapter.get_subtle_thoughts(agent_id, limit))

    def register_search_procedures() -> None:
        @server.tool()
        def search_procedures(query: str, agent_id: str | None = None, top_k: int = 5) -> Dict[str, Any]:
            return with_error_handling(lambda: adapter.search_procedures(query, agent_id, top_k))

    def register_procedure_feedback() -> None:
        @server.tool()
        def procedure_feedback(proc_id: str, success: bool) -> Dict[str, Any]:
            return with_error_handling(lambda: adapter.procedure_feedback(proc_id, success))

    def register_get_self_improvement_proposals() -> None:
        @server.tool()
        def get_self_improvement_proposals() -> Dict[str, Any]:
            return with_error_handling(adapter.get_self_improvement_proposals)

    register_tool("memory_store", register_memory_store)
    register_tool("memory_query", register_memory_query)
    register_tool("memory_get", register_memory_get)
    register_tool("memory_delete", register_memory_delete)
    register_tool("memory_stats", register_memory_stats)
    register_tool("memory_health", register_memory_health)
    register_tool("store_observation", register_store_observation)
    register_tool("recall_context", register_recall_context)
    register_tool("start_episode", register_start_episode)
    register_tool("get_knowledge_gaps", register_get_knowledge_gaps)
    register_tool("get_subtle_thoughts", register_get_subtle_thoughts)
    register_tool("search_procedures", register_search_procedures)
    register_tool("procedure_feedback", register_procedure_feedback)
    register_tool("get_self_improvement_proposals", register_get_self_improvement_proposals)

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
        transport=cfg.mcp.transport,
        supported_transports=["stdio", "sse"]
    )


if __name__ == "__main__":
    main()
