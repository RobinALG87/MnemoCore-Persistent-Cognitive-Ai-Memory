"""
MnemoCore MCP Server
====================
MCP bridge exposing MnemoCore API tools for agent clients.
"""

from typing import Any, Callable, Dict, List, Optional
from loguru import logger
import asyncio

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

    # --- Phase 4.5 & 5.0: Advanced Synthesis & Export Tools ---

    def register_memory_synthesize() -> None:
        @server.tool()
        def memory_synthesize(
            query: str,
            top_k: int = 10,
            max_depth: int = 3,
            context_text: str | None = None,
            project_id: str | None = None,
        ) -> Dict[str, Any]:
            """
            Phase 4.5: Recursive Synthesis Engine.

            Decomposes complex queries into sub-questions, searches MnemoCore
            in parallel, and synthesizes a comprehensive answer using the
            Recursive Language Model (RLM) paradigm.

            Args:
                query: The complex query to synthesize (can be multi-topic)
                top_k: Number of final results to return (1-50)
                max_depth: Maximum recursion depth (0-5)
                context_text: Optional external text for RippleContext search
                project_id: Optional project scope for isolation masking
            """
            # Input validation
            if not query or len(query.strip()) < 3:
                return _result_error("Query must be at least 3 characters")
            if not 1 <= top_k <= 50:
                return _result_error("top_k must be between 1 and 50")
            if not 0 <= max_depth <= 5:
                return _result_error("max_depth must be between 0 and 5")

            payload = {
                "query": query,
                "top_k": top_k,
                "max_depth": max_depth,
            }
            if context_text:
                payload["context_text"] = context_text
            if project_id:
                payload["project_id"] = project_id

            return with_error_handling(lambda: adapter.synthesize(payload))

    def register_memory_dream() -> None:
        @server.tool()
        def memory_dream(
            max_cycles: int = 1,
            force_insight: bool = False,
        ) -> Dict[str, Any]:
            """
            Manually trigger a dream session (SubconsciousDaemon cycle).

            The dream loop performs:
            - Concept extraction from recent memories
            - Parallel drawing (finding unexpected connections)
            - Memory re-evaluation and valuation
            - Meta-insight generation

            Args:
                max_cycles: Number of dream cycles to run (1-10)
                force_insight: Force generation of a meta-insight

            Returns:
                Dict with cycle results including insights generated.
            """
            # Input validation
            if not 1 <= max_cycles <= 10:
                return _result_error("max_cycles must be between 1 and 10")

            payload = {
                "max_cycles": max_cycles,
                "force_insight": force_insight,
            }
            return with_error_handling(lambda: adapter.dream(payload))

    def register_memory_export() -> None:
        @server.tool()
        def memory_export(
            agent_id: str | None = None,
            tier: str | None = None,
            limit: int = 100,
            include_metadata: bool = True,
            format: str = "json",
        ) -> Dict[str, Any]:
            """
            Export memories as JSON for backup, analysis, or migration.

            Args:
                agent_id: Optional filter by agent_id
                tier: Optional filter by tier ("hot", "warm", "cold")
                limit: Maximum number of memories to export (1-1000)
                include_metadata: Include full metadata in export
                format: Export format ("json" or "jsonl")

            Returns:
                Dict with exported memories count and data.
            """
            # Input validation
            if not 1 <= limit <= 1000:
                return _result_error("limit must be between 1 and 1000")
            if tier and tier not in ("hot", "warm", "cold", "soul"):
                return _result_error("tier must be one of: hot, warm, cold, soul")
            if format not in ("json", "jsonl"):
                return _result_error("format must be 'json' or 'jsonl'")

            payload = {
                "agent_id": agent_id,
                "tier": tier,
                "limit": limit,
                "include_metadata": include_metadata,
                "format": format,
            }
            return with_error_handling(lambda: adapter.export(payload))

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
    register_tool("memory_synthesize", register_memory_synthesize)
    register_tool("memory_dream", register_memory_dream)
    register_tool("memory_export", register_memory_export)

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
