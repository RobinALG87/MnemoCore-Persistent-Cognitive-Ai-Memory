"""
LLM Configuration – Shared types and configuration
===================================================
Provides LLMProvider enum, LLMConfig dataclass, and factory methods.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from loguru import logger


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    GOOGLE_GEMINI = "google_gemini"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    CUSTOM = "custom"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider = LLMProvider.MOCK
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Provider-specific defaults
    @classmethod
    def openai(cls, model: str = "gpt-4", api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.OPENAI, model=model, api_key=api_key, **kwargs)

    @classmethod
    def openrouter(cls, model: str = "anthropic/claude-3.5-sonnet", api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(
            provider=LLMProvider.OPENROUTER,
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": "https://mnemocore.ai", "X-Title": "MnemoCore"},
            **kwargs
        )

    @classmethod
    def anthropic(cls, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.ANTHROPIC, model=model, api_key=api_key, **kwargs)

    @classmethod
    def google_gemini(cls, model: str = "gemini-1.5-pro", api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.GOOGLE_GEMINI, model=model, api_key=api_key, **kwargs)

    @classmethod
    def ollama(cls, model: str = "llama3.1", base_url: str = "http://localhost:11434", **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.OLLAMA, model=model, base_url=base_url, **kwargs)

    @classmethod
    def lm_studio(cls, model: str = "local-model", base_url: str = "http://localhost:1234/v1", **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.LM_STUDIO, model=model, base_url=base_url, **kwargs)

    @classmethod
    def custom(cls, model: str, base_url: str, api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.CUSTOM, model=model, base_url=base_url, api_key=api_key, **kwargs)

    @classmethod
    def mock(cls, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.MOCK, **kwargs)


__all__ = ["LLMProvider", "LLMConfig"]
