"""
MnemoCore MCP (Model Context Protocol) Module
==============================================
Legacy v2 MCP server implementation for AI agent integration.

This package proxies the legacy HAIM REST surface, so it is not the v3
AgentMemory persistence boundary. It remains for existing v2 clients. New
scope-bound integrations should use ``mnemocore.integrations.mcp`` with an
explicit AgentMemory client, or compose HybridMemoryRuntime directly.

The Model Context Protocol (MCP) provides a standardized way for AI agents
to interact with MnemoCore's memory system.

Features:
    - Full MCP protocol compliance
    - Tool definitions for memory operations
    - STDIO and TCP transport support
    - Configurable tool permissions

Available Tools:
    - memory_store: Store new memories
    - memory_query: Search and retrieve memories
    - memory_get: Retrieve specific memory by ID
    - memory_delete: Remove memories
    - memory_stats: Get system statistics
    - memory_health: Health check endpoint

Configuration:
    MCP settings are configured in config.yaml under the 'mcp' section:
    - enabled: Enable/disable MCP server
    - transport: "stdio" or "tcp"
    - host/port: TCP binding (if using TCP transport)
    - allow_tools: List of permitted tools

Usage:
    The MCP server is typically started automatically when the API starts.
    For direct integration:

    from mnemocore.mcp import MCPServer

    server = MCPServer(engine)
    await server.start()
"""

