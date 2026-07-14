"""Removed v2 ``Memory`` facade kept only to provide a safe migration error."""


_MEMORY_FACADE_REMOVAL_MESSAGE = (
    "v3 migration: Memory is no longer available because it cannot bind an explicit "
    "MemoryScope. Migrate to AgentMemory with an explicit MemoryScope (or "
    "HybridMemoryRuntime for hybrid retrieval)."
)


class Memory:
    """Removed v2 facade.

    ``Memory`` accepted unscoped calls and could construct the legacy HAIM
    persistence stack. That is unsafe under v3's exact-scope contract, so all
    invocations fail closed. Direct ``HAIMEngine`` users retain the documented
    v2 compatibility surface; v3 users must compose AgentMemory explicitly.
    """

    def __init__(self, profile: str = "lite", **engine_kwargs) -> None:
        del profile, engine_kwargs
        raise RuntimeError(_MEMORY_FACADE_REMOVAL_MESSAGE)
