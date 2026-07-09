"""
MnemoCore - Infrastructure for Persistent Cognitive Memory
===========================================================

Lightweight agentic memory for AI (default: pure HDV/VSA lite profile).
Binary 16384-dim vectors + Hamming sim, zero heavy deps/workers by default.

Primary usage (lite - recommended, only Python needed):
    from mnemocore import Memory
    m = Memory()  # profile="lite"
    m.add("User prefers concise answers")
    results = m.search("concise answers")

Non-lite uses full HAIM engine (lazy loaded).

Key: real TextEncoder + BinaryHDV (encode + similarity) preserved in lite.
Heavy workers (subconscious, pulse, anticipatory) disabled for lite.

Version: 2.0.0
"""

__version__ = "2.0.0"

# Lightweight high-level facade (Phase 0+)
# Usage (very light by default):
#   from mnemocore import Memory
#   m = Memory()                    # uses lite profile, in-memory
#   m.add("User prefers concise answers")
#   results = m.search("preferences")

from .core.config import load_config

def _run_sync(coro):
    """Helper to run an async coroutine synchronously (used only for non-lite engine path)."""
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
    else:
        return asyncio.run(coro)

# Lazy import for non-lite to keep 'from mnemocore import Memory' light
def _get_haim_engine():
    from .core.engine import HAIMEngine
    return HAIMEngine

from .core.lite_engine import LiteEngine


class Memory:
    """
    Simple high-level memory client for agents.
    For profile='lite' (default): uses pure LiteEngine (sync HDV + Hamming, in-memory, no heavy deps/workers).
    Non-lite: delegates to HAIMEngine (preserves back-compat) with lazy import.
    """

    def __init__(self, profile: str = "lite", **engine_kwargs):
        self._is_lite = profile == "lite"
        if self._is_lite:
            self._backend = LiteEngine()
            return
        # non-lite path - lazy import to keep top-level import light
        if "config" not in engine_kwargs:
            import os
            from dataclasses import replace
            prev = os.environ.get("HAIM_PROFILE")
            os.environ["HAIM_PROFILE"] = profile
            try:
                cfg = load_config()
            finally:
                if prev is None:
                    os.environ.pop("HAIM_PROFILE", None)
                else:
                    os.environ["HAIM_PROFILE"] = prev
            if getattr(cfg, "profile", None) != profile:
                cfg = replace(cfg, profile=profile) if hasattr(cfg, "profile") else cfg
            engine_kwargs["config"] = cfg
        HaimEngine = _get_haim_engine()
        self.engine = HaimEngine(**engine_kwargs)
        self._initialized = False

    def add(self, content: str, **meta):
        if self._is_lite:
            # Normalize meta (user_id etc) into metadata dict for consistency with non-lite
            md = dict(meta.get("metadata") or {})
            for k, v in list(meta.items()):
                if k != "metadata":
                    md[k] = v
            return self._backend.store(content, metadata=md if md else None)
        self._ensure_init()
        call_kwargs = {}
        md = dict(meta.get("metadata") or {})
        for k, v in list(meta.items()):
            if k in ("metadata", "goal_id", "project_id"):
                call_kwargs[k] = v
            elif k in ("user_id", "agent_id", "run_id", "context"):
                md[k] = v
        if md:
            call_kwargs["metadata"] = md
        coro = self.engine.store(content, **call_kwargs)
        return _run_sync(coro)

    def search(self, query: str, top_k: int = 5, **kwargs):
        if self._is_lite:
            return self._backend.query(query, top_k=top_k, **kwargs)
        self._ensure_init()
        coro = self.engine.query(query, top_k=top_k, **kwargs)
        results = _run_sync(coro)
        if results:
            return results
        try:
            q = (query or "").lower()
            hot = getattr(getattr(self.engine, "tier_manager", None), "hot", {}) or {}
            matches = []
            for nid, node in list(hot.items())[:100]:
                if q and q in (getattr(node, "content", "") or "").lower():
                    matches.append((nid, 0.85))
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
        return "<Memory (lite)>" if getattr(self, "_is_lite", False) else "<Memory>"
