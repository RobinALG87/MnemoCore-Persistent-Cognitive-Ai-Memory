"""
MnemoCore API Adapter
=====================
HTTP client adapter for communicating with MnemoCore API server.
"""

from typing import Any, Dict, Optional
import requests

from mnemocore.core.exceptions import MnemoCoreError


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
