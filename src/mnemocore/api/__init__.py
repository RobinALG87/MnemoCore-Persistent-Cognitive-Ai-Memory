"""
MnemoCore API Package
=====================
Legacy v2 FastAPI REST API for MnemoCore memory system.

This package's ``main:app`` owns the HAIM lifecycle and legacy global
JSONL/tiering semantics. It is retained for existing v2 deployments and is
not AgentMemory-backed v3 persistence. New v3 HTTP deployments must compose
``create_v3_app(sqlite_path, scope_authorizer=...)`` and authenticate the full
requested MemoryScope before each operation.

Provides HTTP endpoints for all memory operations:

Core Endpoints:
    - POST /store: Store a new memory
    - POST /query: Search memories by content
    - GET /memory/{id}: Retrieve specific memory
    - DELETE /memory/{id}: Delete a memory
    - GET /stats: System statistics
    - GET /health: Health check

Advanced Endpoints:
    - POST /feedback: Provide feedback on retrieval quality
    - GET /insights/gaps: List detected knowledge gaps
    - POST /dream/trigger: Manually trigger dream session
    - GET /export: Export memories in various formats

Features:
    - Rate limiting (configurable)
    - API key authentication
    - Prometheus metrics at /metrics
    - CORS support
    - Request/response validation

Configuration:
    API settings are configured in config.yaml under 'api' section.
    Security settings (API keys) are loaded from environment variables.

Example:
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8100/store",
            json={"content": "Remember this", "metadata": {"source": "api"}},
            headers={"X-API-Key": "your-key"}
        )
"""
