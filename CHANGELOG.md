# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Deprecated

#### Float HDV deprecation (src/core/hdv.py)
- **HDV class**: All public methods now emit `DeprecationWarning` when called
- **Migration path**: Use `BinaryHDV` from `src.core.binary_hdv` instead
- **API mappings**:
  - `HDV(dimension=N)` -> `BinaryHDV.random(dimension=N)`
  - `hdv.bind(other)` -> `hdv.xor_bind(other)`
  - `hdv.unbind(other)` -> `hdv.xor_bind(other)` (XOR is self-inverse)
  - `hdv.cosine_similarity(other)` -> `hdv.similarity(other)`
  - `hdv.permute(shift)` -> `hdv.permute(shift)`
  - `hdv.normalize()` -> No-op (binary vectors are already normalized)
- **Removal timeline**: Float HDV will be removed in a future version

#### BinaryHDV compatibility shims added
- **bind()**: Alias for `xor_bind()` - for legacy API compatibility
- **unbind()**: Alias for `xor_bind()` - XOR is self-inverse
- **cosine_similarity()**: Alias for `similarity()` - returns Hamming-based similarity
- **normalize()**: No-op for binary vectors
- **__xor__()**: Enables `v1 ^ v2` syntax for binding

### Fixed

#### llm_integration.py (6 fixes)
- **Import paths**: Fixed incorrect import paths from `haim.src.core.engine` to `src.core.engine` and `haim.src.core.node` to `src.core.node`
- **Missing import**: Added `from datetime import datetime` for dynamic timestamps
- **Memory access API**: Changed `self.haim.memory_nodes.get()` to `self.haim.tier_manager.get_memory()` at lines 34, 114, 182, 244, 272 - using the correct API for memory access
- **Superposition query**: Replaced non-existent `superposition_query()` method with workaround using combined hypotheses query (includes TODO comment for future improvement)
- **Concept binding**: Replaced non-existent `bind_concepts()` with placeholder - engine has `bind_memories()` available
- **OR orchestration**: Replaced non-existent `orchestrate_orch_or()` with workaround that sorts by LTP strength

#### api/main.py (1 fix)
- **Delete endpoint**: Fixed attribute reference from `engine.memory_nodes` to `engine.tier_manager.hot` at line 229 - correct attribute for hot memory tier

#### engine.py (1 fix)
- **Synapse persistence**: Implemented `_save_synapses()` method (lines 369-390) that was previously an empty stub
  - Creates parent directory if it doesn't exist
  - Writes all synapses to disk in JSONL format
  - Includes all synapse attributes: `neuron_a_id`, `neuron_b_id`, `strength`, `fire_count`, `success_count`, `last_fired`
  - Handles errors gracefully with logging

### Changed

- **Dynamic timestamps**: LLM integration now uses `datetime.now().isoformat()` instead of hardcoded timestamp `"2026-02-04"` for accurate temporal tracking
