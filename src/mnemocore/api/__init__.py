"""
MnemoCore API Package
=====================
FastAPI-based REST API for MnemoCore memory system.

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

