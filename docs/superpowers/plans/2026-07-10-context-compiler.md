# Context Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic local `compile_context()` API that returns a token-bounded mission brief with receipts.

**Architecture:** The client reuses scoped `recall()` and a new pure compiler maps kinds into five levels. The compiler deduplicates by memory id, visits levels in a fixed priority, and never exceeds the user-provided budget. It estimates tokens with `max(1, ceil(len(content) / 4))`, avoiding any new dependency.

**Tech Stack:** Python 3.10+, frozen dataclasses, existing client API, pytest.

## Global Constraints

- No persistence schema, provider, network call, LLM, or dependency is added.
- Scope remains enforced only by existing recall policy; base clients default exact scope and sessions retain controlled ancestor recall.
- Every output retains content, source scope, score components, reason, and evidence ids.
- `token_budget` is a positive integer; returned estimated tokens never exceed it.

---

### Task 1: Context models

**Files:**
- Modify: `src/mnemocore/agent_memory/models.py`
- Modify: `src/mnemocore/agent_memory/__init__.py`
- Test: `tests/agent_memory/test_models.py`

**Produces:** frozen `MemoryReceipt` and `ContextItem`.

- [ ] **Step 1: Add the failing receipt test**

```python
def test_context_item_preserves_receipt(scope):
    receipt = MemoryReceipt("memory-1", scope, MemoryKind.PREFERENCE, 1.0,
        {"bm25_rank": 1.0}, "Matched lexical terms", ("event-1",), 4)
    assert ContextItem("Prefer concise updates", receipt).receipt.memory_id == "memory-1"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/agent_memory/test_models.py::test_context_item_preserves_receipt -q`  
Expected: import failure because both types are absent.

- [ ] **Step 3: Implement and export the minimal immutable types**

```python
@dataclass(frozen=True, slots=True)
class MemoryReceipt:
    memory_id: str; scope: MemoryScope; kind: MemoryKind; score: float
    score_components: Mapping[str, Any]; reason: str
    evidence_ids: tuple[str, ...]; estimated_tokens: int

@dataclass(frozen=True, slots=True)
class ContextItem:
    content: str; receipt: MemoryReceipt
```

Validate nonblank ids/content/reason, finite score, string evidence ids, and positive estimated tokens with the existing validation helpers.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/agent_memory/test_models.py -q`  
Commit: `git commit -m "feat(memory): add context receipt models"`

### Task 2: Pure compiler and public APIs

**Files:**
- Create: `src/mnemocore/agent_memory/context.py`
- Modify: `src/mnemocore/agent_memory/models.py`
- Modify: `src/mnemocore/agent_memory/client.py`
- Modify: `src/mnemocore/agent_memory/__init__.py`
- Test: `tests/agent_memory/test_client.py`

**Produces:** frozen `ContextPack(query, token_budget, estimated_tokens, core, working, episodic, semantic, procedural)`; async/sync `compile_context` on base clients and sessions.

- [ ] **Step 1: Add the failing public behavior test**

```python
@pytest.mark.asyncio
async def test_compile_context_is_bounded_and_receipted(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="core")
    async with await AgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        await memory.remember("Prefer concise retrieval updates", kind=MemoryKind.PREFERENCE)
        await memory.remember("Use FTS retrieval before reranking", kind=MemoryKind.PROCEDURE)
        pack = await memory.compile_context("retrieval", token_budget=20)
    assert pack.estimated_tokens <= 20
    assert pack.core[0].receipt.kind is MemoryKind.PREFERENCE
    assert pack.procedural[0].receipt.evidence_ids
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/agent_memory/test_client.py::test_compile_context_is_bounded_and_receipted -q`  
Expected: `AttributeError` for `compile_context`.

- [ ] **Step 3: Implement the compiler and thin client wiring**

```python
LEVELS = (("core", (MemoryKind.PREFERENCE,)),
          ("working", (MemoryKind.OBSERVATION, MemoryKind.DECISION)),
          ("procedural", (MemoryKind.PROCEDURE,)),
          ("episodic", (MemoryKind.EPISODE,)),
          ("semantic", (MemoryKind.FACT, MemoryKind.SUMMARY)))

def estimate_tokens(content: str) -> int:
    return max(1, math.ceil(len(content) / 4))
```

For each level, invoke the existing recall with its kinds and `limit=10`, transform each result to `ContextItem`, skip duplicate memory ids, and skip any item that cannot fit the remaining budget. Base API defaults `include_ancestors=False`; session API defaults `True`; sync wrappers delegate unchanged.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/agent_memory/test_models.py tests/agent_memory/test_client.py -q --tb=short`  
Commit: `git commit -m "feat(memory): compile bounded mission context"`

### Task 3: Document and prove session provenance

**Files:**
- Modify: `docs/AGENT_MEMORY_QUICKSTART.md`
- Modify: `examples/agent_memory_wow.py`
- Test: `tests/agent_memory/test_client.py`

- [ ] **Step 1: Add the failing session-provenance test**

```python
@pytest.mark.asyncio
async def test_session_context_receipts_project_preference(tmp_path):
    # Store a project preference, start a child session, compile context,
    # and assert the receipt retains project scope and an evidence id.
```

- [ ] **Step 2: Implement only the verified usage documentation**

```python
brief = await sess.compile_context("retrieval fusion", token_budget=800)
for item in brief.procedural:
    print(item.receipt.memory_id, item.receipt.reason)
```

- [ ] **Step 3: Verify and commit**

Run: `python -m pytest tests/agent_memory/test_client.py -q --tb=short`  
Run: `$env:PYTHONPATH='src'; python examples/agent_memory_wow.py`  
Commit: `git commit -m "docs(memory): demonstrate mission brief receipts"`

## Self-Review

This plan implements only design section 4.7: bounded, leveled context and memory receipts. Feedback, compaction, semantic candidate providers, graph retrieval, and benchmarks remain separate slices. All proposed public types exist before client methods use them; all collection remains behind the current scope-safe recall boundary.
