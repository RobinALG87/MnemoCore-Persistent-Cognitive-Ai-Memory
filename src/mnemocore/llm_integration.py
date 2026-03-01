"""
LLM Integration for HAIM
Multi-provider LLM support: OpenAI, OpenRouter, Anthropic, Google Gemini, and Local AI models

This module provides backward-compatible imports from the new llm package.
All implementation has been moved to the llm/ package.

For direct access to components:
    from mnemocore.llm import (
        LLMProvider,
        LLMConfig,
        LLMClientFactory,
        OllamaClient,
        HAIMLLMIntegrator,
        ContextAwareLLMIntegrator,
        MultiAgentHAIM,
        RLMIntegrator,
    )
"""

import json

# Re-export all components from the llm package for backward compatibility
from .llm import (
    # Configuration
    LLMProvider,
    LLMConfig,
    # Factory
    LLMClientFactory,
    # Providers
    OllamaClient,
    # Integration
    HAIMLLMIntegrator,
    ContextAwareLLMIntegrator,
    MultiAgentHAIM,
    RLMIntegrator,
)


def create_demo():
    """Create HAIM demo with multi-agent system"""
    print("Creating HAIM Multi-Agent Demo...")

    # Create multi-agent system
    multi_agent_haim = MultiAgentHAIM(num_agents=3)

    # Demonstrate collective consciousness
    result = multi_agent_haim.demonstrate_collective_consciousness()

    print("\n=== DEMO RESULT ===")
    print(json.dumps(result, indent=2))

    return result


# Export for backward compatibility
__all__ = [
    "LLMProvider",
    "LLMConfig",
    "LLMClientFactory",
    "OllamaClient",
    "HAIMLLMIntegrator",
    "ContextAwareLLMIntegrator",
    "MultiAgentHAIM",
    "RLMIntegrator",
]


if __name__ == "__main__":
    create_demo()
