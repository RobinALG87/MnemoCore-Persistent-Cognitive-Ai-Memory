"""Small allowlisted MCP surface for the vNext agent-memory API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from .bridge import AgentMemoryBridge, IntegrationPolicy


class MCPMemoryTools:
    """MCP-callable operations; no delete, export, or unrestricted tools by default."""

    allowed_tools = frozenset({"memory_recall", "memory_remember"})

    def __init__(self, bridge: AgentMemoryBridge) -> None:
        self.bridge = bridge

    async def memory_recall(self, query: str, *, token_budget: int | None = None) -> Any:
        return _json_safe(
            await self.bridge.context(query, token_budget=token_budget)
        )

    async def memory_remember(self, content: str, *, kind: str = "observation") -> Any:
        return _json_safe(await self.bridge.remember(content, kind=kind))


def create_mcp_server(
    memory: Any,
    *,
    name: str = "MnemoCore Memory",
    policy: IntegrationPolicy | None = None,
) -> Any:
    """Create a FastMCP server lazily; importing MnemoCore never requires MCP."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError(
            "MCP integration requires the optional 'mcp' package"
        ) from error

    tools = MCPMemoryTools(AgentMemoryBridge(memory, policy=policy))
    server = FastMCP(name)

    @server.tool(name="memory_recall")
    async def memory_recall(query: str, token_budget: int | None = None) -> Any:
        return await tools.memory_recall(query, token_budget=token_budget)

    @server.tool(name="memory_remember")
    async def memory_remember(content: str, kind: str = "observation") -> Any:
        return await tools.memory_remember(content, kind=kind)

    return server


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return value.value
    return value
