"""
Structural tests for lite Memory (drives shipped code, no data/ dep).
"""
import sys
import importlib.util
from pathlib import Path

# ensure src in path for test
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnemocore import Memory


def test_light_memory_add_search_reliable():
    """Core: lite add/search with natural phrase returns positive count."""
    m = Memory(profile="lite")
    m.add("User prefers concise answers")
    results = m.search("concise answers", top_k=3)
    assert isinstance(results, (list, tuple))
    assert len(results) > 0, "reliable add+search must return positive count per verification plan"


def test_lite_metadata_preserved():
    """Lite supports user_id/agent_id etc via metadata (normalized)."""
    m = Memory(profile="lite")
    mid = m.add("Project prefers concise answers", user_id="robin", agent_id="cli")
    node = m._backend._memories[mid]
    assert node.metadata.get("user_id") == "robin"
    assert node.metadata.get("agent_id") == "cli"
    # also works with explicit metadata=
    mid2 = m.add("Another fact", metadata={"topic": "test"})
    assert m._backend._memories[mid2].metadata.get("topic") == "test"


def test_lite_backend_type():
    """Structural: lite uses LiteEngine backend."""
    m = Memory(profile="lite")
    assert type(m._backend).__name__ == "LiteEngine"


def test_lite_import_is_light():
    """Structural: importing/using lite Memory does not load the heavy engine module."""
    # fresh
    if 'mnemocore.core.engine' in sys.modules:
        del sys.modules['mnemocore.core.engine']
    m = Memory(profile="lite")
    m.add("test fact")
    # engine should not be loaded
    assert importlib.util.find_spec("mnemocore.core.engine") is None or 'mnemocore.core.engine' not in sys.modules


def test_lite_config_disables_heavy():
    """Lite profile disables heavy workers."""
    import os
    from mnemocore.core.config import load_config
    os.environ['HAIM_PROFILE'] = 'lite'
    try:
        cfg = load_config()
    finally:
        os.environ.pop('HAIM_PROFILE', None)
    assert cfg.subconscious_ai.enabled == False
    assert cfg.pulse.enabled == False
    # anticipatory also forced off
    assert getattr(cfg, 'anticipatory', None) is None or cfg.anticipatory.enabled == False


def test_lite_exact_plan_verification():
    """Exact verification command from plan: 'test fact' -> positive len."""
    m = Memory(profile="lite")
    m.add("test fact")
    res_len = len(m.search("fact"))
    assert res_len > 0, f"plan verification expects len>0 for search('fact') after add('test fact'), got {res_len}"
