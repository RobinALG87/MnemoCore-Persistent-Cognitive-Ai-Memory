"""
MnemoCore Agent Memory "Wow" Demo - First Version
==================================================
Demonstrates the magic core loop for an optimal agent memory:

Recall before task → observe during work → evaluate result → learn for next session.

The scenario:
- Day 1: A code agent (codex) tries to fix a bug in the repo. It takes a bad path,
  gets feedback ("that broke tests"), records the failure + decision.
- "Next day": New session. Agent wakes up, recalls the repo + previous failure,
  avoids the bad approach, follows Robin's preference, and can explain *which*
  memory influenced the decision.

Run:
  PYTHONPATH=src python examples/agent_memory_wow.py

Uses only the new agent_memory (SQLite local, scope isolation, explainable recall).
No LLM required for this demo.
"""

import asyncio
import tempfile
from pathlib import Path

from mnemocore.agent_memory import AgentMemory, MemoryKind, MemoryScope


async def simulate_day1(memory: AgentMemory) -> None:
    """Simulate first session: agent attempts fix, fails, learns."""
    print("\n=== DAY 1: Code agent wakes up, attacks bug ===")
    sess = await memory.start_session(goal="Fix failing retrieval fusion in agent_memory")

    # Agent "recalls" (empty first time) then starts working
    print("[codex] No prior memories for this exact session. Proceeding...")

    # Agent tries a bad approach (simulated decision)
    bad_decision = await sess.remember(
        "Tried pure HDV rerank as primary signal for retrieval (skipped FTS union).",
        kind=MemoryKind.OBSERVATION,
        metadata={"repo": "MnemoCore", "file": "sqlite_store.py", "attempt": 1},
        idempotency_key="day1-bad-path",
    )
    print(f"[codex] Stored attempt: {bad_decision.id[:8]}...")

    # Simulate running and getting outcome feedback
    print("[codex] Ran the change... tests exploded, precision dropped.")
    await sess.remember(
        "Pure HDV alone caused bad recall and missed lexical matches. Failed.",
        kind=MemoryKind.OBSERVATION,
        metadata={"outcome": "failure", "error": "tests broken, low precision"},
        idempotency_key="day1-failure-feedback",
    )

    # Record a hard lesson / preference from feedback - at *project* level (no session_id) so it survives across sessions
    # (session-scoped for detailed episodes; project for reusable rules/preferences)
    lesson = await memory.remember(  # use base memory for cross-session reusable knowledge
        "For retrieval: ALWAYS start with FTS/BM25 candidate union. HDV only as cheap reranker or novelty signal, never sole semantic source.",
        kind=MemoryKind.PREFERENCE,
        metadata={"source": "failure_feedback", "repo": "MnemoCore", "author": "robin"},
        idempotency_key="retrieval-rule-1",
    )
    print(f"[codex] Learned project-level preference: {lesson.id[:8]}")

    await sess.finish(
        outcome="failure",
        reward=0.1,
        notes="Bad path taken; documented to avoid repeat. Will use union next time.",
    )
    print("[codex] Session finished with failure. Memory ledger updated.")


async def simulate_day2(memory: AgentMemory) -> None:
    """Simulate next day / new session: agent is smarter."""
    print("\n=== DAY 2 (new session): Agent wakes smarter ===")
    sess = await memory.start_session(
        goal="Fix failing retrieval fusion in agent_memory (retry)"
    )

    print("[codex] Starting with recall before any code change...")

    # Recall previous failures + preferences.
    # With the new include_ancestors (default True on sessions), a session can see
    # project-level reusable knowledge while keeping detailed episodes scoped to the session.
    warnings = await sess.recall(
        "failed retrieval OR pure HDV OR bad approach OR union",
        kinds=(MemoryKind.OBSERVATION, MemoryKind.PREFERENCE),
        limit=5,
    )
    print(f"[codex] Recalled {len(warnings)} relevant memories (session + ancestor/project scopes).")

    for i, r in enumerate(warnings):
        m = r.memory
        print(f"  [{i+1}] {m.kind.value}: {m.content[:70]}...")
        print(f"       why: {r.reason}")
        print(f"       score_components: {dict(r.score_components)}")
        # The magic: explicit explanation of influence
        if "HDV" in m.content or "union" in m.content.lower():
            print("       >>> This memory will BLOCK the bad path and GUIDE the fix.")

    # Agent now decides correctly thanks to memory
    decision = await memory.remember(
        "Decision: implement candidate union (FTS first) + optional HDV rerank. Follows documented preference.",
        kind=MemoryKind.OBSERVATION,
        metadata={"based_on": [w.memory.id for w in warnings if w.memory.kind in (MemoryKind.PREFERENCE, MemoryKind.OBSERVATION)]},
        idempotency_key="day2-good-decision",
    )
    print(f"\n[codex] Made informed decision: {decision.id[:8]}")

    # Simulate success (in session for episode detail)
    await sess.remember(
        "Implemented union + HDV rerank. All tests green, recall quality improved.",
        kind=MemoryKind.OBSERVATION,
        metadata={"outcome": "success", "repo": "MnemoCore"},
        idempotency_key="day2-success",
    )

    await sess.finish(outcome="success", reward=1.0, notes="Avoided documented failure. Used preference.")
    print("[codex] Session finished successfully. Agent measurably smarter.")


async def main() -> None:
    print("MnemoCore Agent Memory OS - First Wow Demo")
    print("Core loop: Recall → Observe → Evaluate → Learn (cross-session)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "agent-memory-wow.db"

        # Use explicit scope: user=robin, agent=codex, project=mnemocore
        scope = MemoryScope(
            tenant_id="local",
            user_id="robin",
            agent_id="codex",
            project_id="mnemocore",
        )

        async with await AgentMemory.open(db_path, scope=scope) as memory:
            # Day 1
            await simulate_day1(memory)

            # "Next day" - new session, same project/user/agent scope (different session_id auto)
            await simulate_day2(memory)

            # Bonus: show cross-session retrieval at project level (no session_id)
            print("\n=== BONUS: Project-level recall (no active session) ===")
            project_ctx = await memory.recall("retrieval rule OR preference OR union", limit=3)
            print(f"Project-level found {len(project_ctx)} items (from any session).")
            for r in project_ctx[:1]:
                print(f"  - {r.memory.content[:60]}... (reason: {r.reason})")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE.")
    print("Agent woke smarter: recognized context, avoided failure, followed preference,")
    print("and every recalled item explains *why* it was chosen + its source.")


if __name__ == "__main__":
    asyncio.run(main())
