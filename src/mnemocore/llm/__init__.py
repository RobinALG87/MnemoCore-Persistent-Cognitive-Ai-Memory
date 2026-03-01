"""
LLM Integration Package – Multi-provider LLM support
=====================================================
Multi-provider LLM support: OpenAI, OpenRouter, Anthropic, Google Gemini, and Local AI models.

This package provides modular LLM integration with separate modules for each concern:

Configuration:
    - LLMProvider: Enum of supported providers
    - LLMConfig: Configuration dataclass with provider-specific factory methods

Client Management:
    - LLMClientFactory: Factory for creating LLM clients

Providers:
    - OllamaClient: Fallback HTTP client for Ollama

Integration:
    - HAIMLLMIntegrator: Bridge between HAIM and LLM reasoning
    - ContextAwareLLMIntegrator: Token-aware context optimization
    - RLMIntegrator: Recursive language model queries
    - MultiAgentHAIM: Multi-agent system with shared memory

Usage:
    from mnemocore.llm import LLMConfig, HAIMLLMIntegrator

    config = LLMConfig.openai(model="gpt-4", api_key="...")
    integrator = HAIMLLMIntegrator.from_config(engine, config)
    result = integrator.reconstructive_recall("my query")
"""

from .config import LLMProvider, LLMConfig
from .factory import LLMClientFactory
from .ollama import OllamaClient
from .integrator import HAIMLLMIntegrator
from .context_aware import ContextAwareLLMIntegrator
from .multi_agent import MultiAgentHAIM
from .rlm import RLMIntegrator


__all__ = [
    # Configuration
    "LLMProvider",
    "LLMConfig",
    # Factory
    "LLMClientFactory",
    # Providers
    "OllamaClient",
    # Integration
    "HAIMLLMIntegrator",
    "ContextAwareLLMIntegrator",
    "MultiAgentHAIM",
    "RLMIntegrator",
]
