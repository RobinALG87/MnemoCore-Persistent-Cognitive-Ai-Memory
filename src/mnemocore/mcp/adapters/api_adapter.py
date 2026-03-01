"""
MnemoCore API Adapter
=====================
HTTP client adapter for communicating with MnemoCore API server.
"""

from typing import Any, Dict, Optional
import urllib.parse
import os

import requests

from mnemocore.core.exceptions import MnemoCoreError


def _is_production_mode() -> bool:
    """Check if running in production mode."""
    return os.environ.get("HAIM_ENV", "development").lower() in ("production", "prod", "staging")


class MnemoCoreAPIError(MnemoCoreError):
    """
    Exception raised when API communication fails.

    Attributes:
        status_code: HTTP status code if available (None for network errors).
    """

    def __init__(self, message: str, status_code: Optional[int] = None, context: Optional[dict] = None):
        ctx = context or {}
        if status_code is not None:
            ctx["status_code"] = status_code
        super().__init__(message, ctx)
        self.status_code = status_code


class MnemoCoreAPIAdapter:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int = 15):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

        # SECURITY: Enforce HTTPS in production mode
        if _is_production_mode() and not self.base_url.startswith("https://"):
            raise MnemoCoreAPIError(
                f"SECURITY ERROR: API URL must use HTTPS in production mode. "
                f"Got: {self.base_url}. Set HAIM_ENV=development for local testing."
            )

        # Warn if using HTTP in any non-development environment
        if not self.base_url.startswith("https://") and not self.base_url.startswith("http://localhost"):
            import warnings
            warnings.warn(
                f"API URL '{self.base_url}' is not using HTTPS. "
                "API key will be sent over an unencrypted connection. "
                "This is strongly discouraged for production.",
                UserWarning
            )

    def _build_url(self, path: str, query_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Build URL with properly encoded query parameters.

        SECURITY: Uses urllib.parse.urlencode() to prevent URL injection attacks.
        """
        url = f"{self.base_url}{path}"
        if query_params:
            # Filter out None values and encode properly
            filtered_params = {k: v for k, v in query_params.items() if v is not None}
            if filtered_params:
                encoded_params = urllib.parse.urlencode(filtered_params, safe='')
                url = f"{url}?{encoded_params}"
        return url

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.request(
                method=method,
                url=url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise MnemoCoreAPIError(f"Upstream request failed: {exc}") from exc

        if response.status_code >= 400:
            try:
                details = response.json()
            except ValueError:
                details = {"detail": response.text}
            raise MnemoCoreAPIError(
                f"Upstream error ({response.status_code}): {details}",
                status_code=response.status_code,
            )

        try:
            return response.json()
        except ValueError as exc:
            raise MnemoCoreAPIError("Upstream returned non-JSON response") from exc

    def store(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/store", payload)

    def query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/query", payload)

    def get_memory(self, memory_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/memory/{memory_id}")

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/memory/{memory_id}")

    def stats(self) -> Dict[str, Any]:
        return self._request("GET", "/stats")

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    # --- Phase 5: Cognitive Client Adapters ---

    def observe_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/wm/observe", payload)

    def get_working_context(self, agent_id: str, limit: int = 16) -> Dict[str, Any]:
        encoded_agent_id = urllib.parse.quote(agent_id, safe='')
        url = self._build_url(f"/wm/context/{encoded_agent_id}", {"limit": limit})
        return self._request("GET", url.replace(self.base_url, ""))

    def start_episode(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/episodes/start", payload)
        
    def get_knowledge_gaps(self) -> Dict[str, Any]:
        return self._request("GET", "/gaps")
        
    def get_subtle_thoughts(self, agent_id: str, limit: int = 5) -> Dict[str, Any]:
        encoded_agent_id = urllib.parse.quote(agent_id, safe='')
        url = self._build_url(f"/agents/{encoded_agent_id}/subtle-thoughts", {"limit": limit})
        return self._request("GET", url.replace(self.base_url, ""))

    def search_procedures(self, query: str, agent_id: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        params = {"query": query, "top_k": top_k}
        if agent_id:
            params["agent_id"] = agent_id
        url = self._build_url("/procedures/search", params)
        return self._request("GET", url.replace(self.base_url, ""))
        
    def procedure_feedback(self, proc_id: str, success: bool) -> Dict[str, Any]:
        return self._request("POST", f"/procedures/{proc_id}/feedback", {"success": success})

    def get_self_improvement_proposals(self) -> Dict[str, Any]:
        return self._request("GET", "/meta/proposals")

    # --- Phase 4.5 & 5.0: Advanced Synthesis & Export ---

    def synthesize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4.5: Recursive synthesis query."""
        return self._request("POST", "/rlm/query", payload)

    def dream(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a dream session (SubconsciousDaemon cycle)."""
        return self._request("POST", "/dream", payload)

    def export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Export memories as JSON."""
        # Build query string for export parameters using safe encoding
        params = {
            "limit": payload.get("limit", 100),
            "include_metadata": str(payload.get("include_metadata", True)).lower(),
            "format": payload.get("format", "json"),
        }
        if payload.get("agent_id"):
            params["agent_id"] = payload["agent_id"]
        if payload.get("tier"):
            params["tier"] = payload["tier"]

        url = self._build_url("/export", params)
        return self._request("GET", url.replace(self.base_url, ""))

