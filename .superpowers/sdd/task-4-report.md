# Task 4 Report: Scoped FTS5 Recall and Temporal Validity

## Status

Implemented scope-safe lexical recall with point-in-time validity filtering.

## Changes

- Added `SQLiteMemoryStore.recall(...)` with the protocol's existing signature.
- Tokenized queries with Unicode `\w+`, quoted every token, and joined tokens with
  `OR` so raw FTS syntax never reaches `MATCH`.
- Restricted candidates to the exact scope, active status, optional memory kinds,
  and inclusive-start/exclusive-end validity at a normalized ISO-Z `as_of` value.
- Ordered candidates by SQLite FTS5 BM25, over-fetched by `limit * 3`, and
  deduplicated by memory ID.
- Returned reciprocal lexical rank as both the foundation score and `bm25_rank`,
  retained SQLite's unmodified value as `bm25_raw`, and used the required reason
  string.
- Preserved schema version 1 and Task 3 transactional persistence invariants.

## TDD Evidence

- RED: `python -m pytest tests/agent_memory/test_sqlite_store.py -k recall -q`
  produced 2 expected failures because `SQLiteMemoryStore.recall` was absent.
- GREEN: the same focused command passed 2 tests after implementation.
- Refactor verification: the focused recall command still passed 2 tests.

## Final Verification

- `python -m pytest tests/agent_memory/test_sqlite_store.py -q` — 37 passed.
- `python -m flake8 src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_sqlite_store.py`
  — passed with no findings.
- `python -m mypy --follow-imports=skip src/mnemocore/agent_memory/sqlite_store.py`
  — success, no issues in 1 source file.
- `git diff --check` — passed before report creation; final check repeated before
  commit.

## Concerns

- SQLite BM25 values are intentionally exposed raw and may be negative; callers
  should use `score` or `bm25_rank` for the normalized foundation score.
- A normal package-following mypy invocation is blocked by pre-existing missing
  optional dependency stubs and unrelated legacy import errors. The scoped check
  above disables import following and validates the changed module itself.
- The Task 3 first-write metadata representation minor was not touched because
  recall hydrates stored rows and no metadata write path change was required.

## Temporal Correctness Follow-up

- Corrected lexical timestamp ordering by making the shared SQLite timestamp
  canonicalizer always emit UTC ISO-Z with exactly six fractional digits.
- Confirmed all event, memory, and history writes plus recall's `as_of` parameter
  use that same canonicalizer.
- Added a public `recall` regression covering inclusive `valid_from` and exclusive
  `valid_to` boundaries at a whole-second `as_of`, with records exactly at and one
  microsecond after the boundary.
- RED evidence: the regression incorrectly returned the record starting one
  microsecond in the future and omitted the record valid for one more microsecond.
- GREEN evidence: the focused regression passed after the canonicalizer change.
- Final follow-up verification: the full Task 4 store file passed 38 tests;
  scoped Flake8 passed with no findings; scoped mypy reported no issues in the
  changed source file.
