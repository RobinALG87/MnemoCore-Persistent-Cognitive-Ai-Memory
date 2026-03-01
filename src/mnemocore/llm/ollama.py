"""
Ollama Client – Fallback HTTP client for Ollama
================================================
Fallback Ollama client using direct HTTP calls when OpenAI SDK is not available.
"""

import json
import socket
from typing import Any

from loguru import logger


class OllamaClient:
    """Fallback Ollama client using direct HTTP calls"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response using Ollama API"""
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "")
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            return f"[Ollama Error: {str(e)}]"

    async def agenerate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Async generate response using Ollama API"""
        import aiohttp

        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=120) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        return f"[Ollama Error: HTTP {response.status}]"
        except Exception as e:
            return f"[Ollama Error: {str(e)}]"


__all__ = ["OllamaClient"]
