# Task 3 Report: Transactional Remember, History, and Persistence

## RED

Command:

```text
python -m pytest tests/agent_memory/test_sqlite_store.py::test_remember_is_persistent_and_idempotent -q
```

Result: `1 failed in 58.30s`.

Expected failure:

```text
AttributeError: 'SQLiteMemoryStore' object has no attribute 'remember'
```

## GREEN

After the minimal transactional CRUD implementation, the same focused command
reported `1 passed in 57.95s`.

The follow-up Task 3 edge-case selection reported:

```text
9 passed, 17 deselected in 57.72s
```

It covered exact-scope access and scope-local idempotency, recursive immutable
JSON serialization, UTC timestamp persistence, list filtering/order/limits, and
transaction rollback with the original `sqlite3.Error` preserved as the cause.

## Final verification

```text
python -m pytest tests/agent_memory/test_sqlite_store.py -q
26 passed in 59.26s

python -m mypy --python-version 3.12 --follow-imports=skip --ignore-missing-imports src/mnemocore/agent_memory/sqlite_store.py
Success: no issues found in 1 source file

python -m flake8 src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_sqlite_store.py
exit 0

python -m compileall -q src/mnemocore/agent_memory
exit 0
```

The repository-wide import-following mypy invocation remains unsuitable for a
scoped check because pre-existing legacy modules and third-party stubs fail
before this file is checked. The scoped invocation above isolates Task 3.

## Review fixes

### RED

Command:

```text
python -m pytest tests/agent_memory/test_sqlite_store.py -k "noncanonical or precedes_new_payload or corrupt_stored" -q
```

Result: `9 failed, 26 deselected in 59.34s`.

The failures confirmed all three findings: non-string mapping keys and
non-finite floats were persisted, invalid retry payloads were validated before
the idempotency lookup, and corrupt stored rows leaked or omitted the database
path from hydration errors.

### GREEN and final verification

```text
python -m pytest tests/agent_memory/test_sqlite_store.py -q
35 passed in 60.26s

python -m flake8 src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_sqlite_store.py
exit 0

python -m mypy --python-version 3.12 --follow-imports=skip --ignore-missing-imports src/mnemocore/agent_memory/sqlite_store.py
Success: no issues found in 1 source file
```

Canonical JSON now recursively requires string keys and rejects NaN/infinity
before writes. `BEGIN IMMEDIATE` and the scope-local idempotency lookup now
precede construction or validation of a new payload. Record, event, and history
hydration failures are converted to path-bearing `StorageError` instances while
preserving the hydration exception as `__cause__`.
