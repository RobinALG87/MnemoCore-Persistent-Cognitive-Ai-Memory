# MnemoCore Core

## Beta Notice

This core implementation is in beta.
Behavior, interfaces, and internals may change without backward compatibility guarantees.

## Purpose

`src/core` contains the runtime memory engine and foundational primitives:
- vector encoding,
- memory node lifecycle,
- tier placement,
- synaptic associations,
- configuration and storage adapters.

## Main Modules

- `engine.py` â€“ Core orchestration for store/query and conceptual proxy operations.
- `binary_hdv.py` â€“ Binary vector operations and text encoding utilities.
- `tier_manager.py` â€“ HOT/WARM/COLD movement and persistence strategy.
- `node.py` â€“ Memory node data model and access/LTP-related behavior.
- `synapse.py` â€“ Synaptic edge model and reinforcement dynamics.
- `config.py` â€“ Typed config loading (`config.yaml` + `HAIM_*` overrides).
- `async_storage.py` â€“ Async Redis metadata and stream support.

## Example

```python
from mnemocore.core.engine import HAIMEngine

engine = HAIMEngine()
memory_id = engine.store("The quick brown fox")
results = engine.query("quick fox", top_k=3)
print(memory_id, results)
```

## Testing

Run from repo root:

```bash
python -m pytest tests -v
```

## More Docs

- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`

