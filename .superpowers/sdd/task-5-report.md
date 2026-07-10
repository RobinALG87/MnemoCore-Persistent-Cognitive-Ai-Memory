# Task 5 Report: Logical Forget and Audit History

## Status

Implemented the exact-scope, atomic `SQLiteMemoryStore.forget(...)` contract.

## Behavior

- An active memory is tombstoned with `status="forgotten"` and a strictly newer UTC `updated_at`.
- The same `BEGIN IMMEDIATE` transaction appends a `forgotten` ledger event, appends history containing the supplied reason, and removes the FTS row.
- A repeated forget returns the existing forgotten record unchanged and does not duplicate its event or history entry.
- A foreign scope receives `MemoryNotFoundError` without ledger or projection side effects.
- A projection failure rolls back the ledger event, status/history changes, and preserves the searchable FTS projection.
- A non-active status other than `forgotten` is not eligible for forgetting and is reported as an active-memory miss.

## TDD Evidence

- RED: `python -m pytest tests/agent_memory/test_sqlite_store.py::test_forget_tombstones_and_removes_from_recall -q`
  - Failed with `AttributeError: 'SQLiteMemoryStore' object has no attribute 'forget'`.
- GREEN (forget cases): `python -m pytest tests/agent_memory/test_sqlite_store.py -k "forget" -q`
  - `4 passed, 39 deselected`.
- SQLite store suite: `python -m pytest tests/agent_memory/test_sqlite_store.py -q`
  - `43 passed`.
- Full agent-memory suite: `python -m pytest tests/agent_memory -q`
  - `83 passed`.

## Scoped Checks

- `git diff --check`: passed.
- `python -m flake8 src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_sqlite_store.py --select=E9,F63,F7,F82`: passed.
- `python -m mypy --follow-imports=skip --ignore-missing-imports --python-version 3.12 src/mnemocore/agent_memory/sqlite_store.py`: passed with no issues.
- A direct mypy run without import isolation is blocked by unrelated missing optional dependencies/stubs in the wider package.
- A whole-file Black check reports pre-existing formatting drift as well as formatting preferences in the added tests; applying Black globally would create unrelated edits, so it was not used as a mutation step.

## Files Changed

- `src/mnemocore/agent_memory/sqlite_store.py`
- `tests/agent_memory/test_sqlite_store.py`
- `.superpowers/sdd/task-5-report.md`
