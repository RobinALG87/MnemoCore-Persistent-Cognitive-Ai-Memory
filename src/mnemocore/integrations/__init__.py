"""Optional framework adapters for MnemoCore's agent-memory API."""

from .bridge import AgentMemoryBridge, IntegrationError, IntegrationPolicy
from .crewai import CrewAIMemoryTools
from .langgraph import LangGraphMemory
from .mcp import MCPMemoryTools, create_mcp_server
from .openclaw import OpenClawMemory

__all__ = [
    "AgentMemoryBridge",
    "CrewAIMemoryTools",
    "IntegrationError",
    "IntegrationPolicy",
    "LangGraphMemory",
    "MCPMemoryTools",
    "OpenClawMemory",
    "create_mcp_server",
]
