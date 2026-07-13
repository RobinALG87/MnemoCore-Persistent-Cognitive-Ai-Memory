"""Backward-compatible high-level Memory facade."""

from .core.config import load_config


_LITE_REMOVAL_MESSAGE = (
    'v3 migration: the "lite" profile was removed. Migrate to AgentMemory with an '
    "explicit MemoryScope (or HybridMemoryRuntime for hybrid retrieval)."
)


def _run_sync(coro):
    """Run an async coroutine synchronously for the non-lite engine path."""
    import asyncio
    import concurrent.futures

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


def _get_haim_engine():
    from .core.engine import HAIMEngine

    return HAIMEngine


class Memory:
    """Simple high-level memory client for agents."""

    def __init__(self, profile: str = "lite", **engine_kwargs):
        if profile == "lite":
            raise RuntimeError(_LITE_REMOVAL_MESSAGE)
        if "config" not in engine_kwargs:
            import os
            from dataclasses import replace

            previous_profile = os.environ.get("HAIM_PROFILE")
            os.environ["HAIM_PROFILE"] = profile
            try:
                config = load_config()
            finally:
                if previous_profile is None:
                    os.environ.pop("HAIM_PROFILE", None)
                else:
                    os.environ["HAIM_PROFILE"] = previous_profile
            if getattr(config, "profile", None) != profile:
                config = replace(config, profile=profile) if hasattr(config, "profile") else config
            engine_kwargs["config"] = config
        engine_class = _get_haim_engine()
        self.engine = engine_class(**engine_kwargs)
        self._initialized = False

    def add(self, content: str, **meta):
        self._ensure_init()
        call_kwargs = {}
        metadata = dict(meta.get("metadata") or {})
        for key, value in list(meta.items()):
            if key in ("metadata", "goal_id", "project_id"):
                call_kwargs[key] = value
            elif key in ("user_id", "agent_id", "run_id", "context"):
                metadata[key] = value
        if metadata:
            call_kwargs["metadata"] = metadata
        coroutine = self.engine.store(content, **call_kwargs)
        return _run_sync(coroutine)

    def search(self, query: str, top_k: int = 5, **kwargs):
        self._ensure_init()
        coroutine = self.engine.query(query, top_k=top_k, **kwargs)
        results = _run_sync(coroutine)
        if results:
            return results
        try:
            normalized_query = (query or "").lower()
            hot = getattr(getattr(self.engine, "tier_manager", None), "hot", {}) or {}
            matches = []
            for node_id, node in list(hot.items())[:100]:
                if normalized_query and normalized_query in (
                    getattr(node, "content", "") or ""
                ).lower():
                    matches.append((node_id, 0.85))
                    if len(matches) >= top_k:
                        break
            return matches
        except Exception:
            return results or []

    def _ensure_init(self):
        if getattr(self, "_initialized", False):
            return
        try:
            if hasattr(self, "engine") and hasattr(self.engine, "initialize"):
                _run_sync(self.engine.initialize())
            self._initialized = True
        except Exception:
            self._initialized = True

    def __repr__(self):
        return "<Memory>"
