"""
MnemoCore - Infrastructure for Persistent Cognitive Memory
===========================================================

A cognitive memory system built on Binary Hyperdimensional Computing (HDC/VSA)
that provides AI agents with persistent, self-organizing memory.

Key Features:
    - Binary HDV vectors (16,384 dimensions) for efficient storage and computation
    - Three-tier memory architecture (HOT/WARM/COLD) with automatic consolidation
    - Bayesian LTP (Long-Term Potentiation) for memory strength tracking
    - Subconscious AI for background processing and dream synthesis
    - MCP (Model Context Protocol) server for AI agent integration

Main Packages:
    - core: Engine, HDV operations, tier management, cognitive services
    - storage: Backup, import/export, vector compression
    - cognitive: Reconstructive recall, context optimization, associations
    - events: Event bus and webhook system
    - subconscious: Background daemon, dream pipeline, self-improvement
    - meta: Goal tracking, learning journal
    - mcp: Model Context Protocol server and adapters
    - api: FastAPI REST endpoints
    - cli: Command-line interface
    - llm: Multi-provider LLM integration (OpenAI, Anthropic, Ollama, etc.)

Quick Start:
    from mnemocore.core import HAIMEngine, BinaryHDV

    engine = HAIMEngine()
    await engine.store("Important fact to remember")
    results = await engine.query("fact")

Version: 2.0.0
"""

__version__ = "2.0.0"
