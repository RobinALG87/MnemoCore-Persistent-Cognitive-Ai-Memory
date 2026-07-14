"""Authorization boundary for v3 memory scopes.

Authentication integrations implement this port.  They must verify the
request's authenticated credentials authorize the complete supplied scope.
"""

from __future__ import annotations

from typing import Protocol

from fastapi import Request

from mnemocore.agent_memory import MemoryScope


class ScopeAuthorizer(Protocol):
    """Authorize an authenticated request for one exact memory scope."""

    async def authorize(self, request: Request, scope: MemoryScope) -> bool:
        """Return whether credentials on ``request`` grant ``scope``."""
