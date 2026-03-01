"""
LLM Client Factory – Creates clients for different providers
=============================================================
Factory for creating LLM clients for various providers.
"""

from typing import Any, Optional

from loguru import logger

from .config import LLMConfig, LLMProvider
from ..core.exceptions import UnsupportedProviderError


class LLMClientFactory:
    """Factory for creating LLM clients"""

    @staticmethod
    def create_client(config: LLMConfig) -> Any:
        """Create an LLM client based on configuration"""
        provider = config.provider

        if provider == LLMProvider.MOCK:
            return None

        if provider == LLMProvider.OPENAI:
            return LLMClientFactory._create_openai_client(config)

        if provider == LLMProvider.OPENROUTER:
            return LLMClientFactory._create_openrouter_client(config)

        if provider == LLMProvider.ANTHROPIC:
            return LLMClientFactory._create_anthropic_client(config)

        if provider == LLMProvider.GOOGLE_GEMINI:
            return LLMClientFactory._create_gemini_client(config)

        if provider == LLMProvider.OLLAMA:
            return LLMClientFactory._create_ollama_client(config)

        if provider == LLMProvider.LM_STUDIO:
            return LLMClientFactory._create_lm_studio_client(config)

        if provider == LLMProvider.CUSTOM:
            return LLMClientFactory._create_custom_client(config)

        supported = [p.value for p in LLMProvider]
        raise UnsupportedProviderError(str(provider.value), supported_providers=supported)

    @staticmethod
    def _create_openai_client(config: LLMConfig) -> Any:
        """Create OpenAI client"""
        import os
        try:
            from openai import OpenAI
            api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
            return OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None

    @staticmethod
    def _create_openrouter_client(config: LLMConfig) -> Any:
        """Create OpenRouter client (OpenAI-compatible)"""
        import os
        try:
            from openai import OpenAI
            api_key = config.api_key or os.environ.get("OPENROUTER_API_KEY")
            return OpenAI(
                base_url=config.base_url,
                api_key=api_key,
                default_headers=config.extra_headers
            )
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None

    @staticmethod
    def _create_anthropic_client(config: LLMConfig) -> Any:
        """Create Anthropic client"""
        import os
        try:
            import anthropic
            api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.warning("anthropic package not installed. Install with: pip install anthropic")
            return None

    @staticmethod
    def _create_gemini_client(config: LLMConfig) -> Any:
        """Create Google Gemini client"""
        import os
        try:
            import google.generativeai as genai
            api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(config.model)
        except ImportError:
            logger.warning("google-generativeai package not installed. Install with: pip install google-generativeai")
            return None

    @staticmethod
    def _create_ollama_client(config: LLMConfig) -> Any:
        """Create Ollama client for local models"""
        try:
            from openai import OpenAI
            return OpenAI(base_url=config.base_url, api_key="ollama")
        except ImportError:
            # Fallback to direct HTTP calls
            from .ollama import OllamaClient
            return OllamaClient(base_url=config.base_url, model=config.model)

    @staticmethod
    def _create_lm_studio_client(config: LLMConfig) -> Any:
        """Create LM Studio client (OpenAI-compatible)"""
        try:
            from openai import OpenAI
            return OpenAI(base_url=config.base_url, api_key="lm-studio")
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None

    @staticmethod
    def _create_custom_client(config: LLMConfig) -> Any:
        """Create custom OpenAI-compatible client"""
        try:
            from openai import OpenAI
            return OpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "custom"
            )
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None


__all__ = ["LLMClientFactory"]
