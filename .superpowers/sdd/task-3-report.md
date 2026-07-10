# Task 3 Report: Pure Timeline Policy and Atomic Supersession

## Status

DONE. Canonical timeline query policy, complete immutable supersession snapshots,
strict replay validation, and atomic exact-scope fact supersession are implemented.

## RED / GREEN evidence

### Initial RED

Command:

```text
python -m pytest tests/agent_memory/test_timeline.py tests/agent_memory/test_sqlite_timeline.py -k supersede -q
```

Result: collection failed as expected with
`ModuleNotFoundError: No module named 'mnemocore.agent_memory.timeline'` (`28 items / 1 error`).
No production implementation existed before this run.

### Pure-policy GREEN

Command:

```text
python -m pytest tests/agent_memory/test_timeline.py -q
```

Result: `22 passed in 57.54s`.

### Transactional GREEN and rollback-harness correction

Command:

```text
python -m pytest tests/agent_memory/test_sqlite_timeline.py -q
```

Initial result: `27 passed, 1 failed`. The failure was in the test injection mechanism:
SQLite does not allow triggers on FTS virtual tables. The FTS boundary test was corrected to
use a temporary check-constrained stand-in table so the replacement FTS insert itself fails.

Repeated result: `28 passed in 60.79s`.

### Review RED / GREEN

Two self-review regressions were written and observed failing:

```text
python -m pytest tests/agent_memory/test_timeline.py tests/agent_memory/test_sqlite_timeline.py -q -k "denormalized or later_superseded"
```

Result: `2 failed, 9 passed, 41 deselected`. The failures proved that a whitespace-denormalized
scope field could normalize back to the same key, and an original retry failed after its
replacement was later superseded.

Fixes:

- Replay now requires each raw scope field to already equal its normalized value.
- Retry still validates the immutable original event, then returns the replacement's current
  exact-scope projection without assuming it remains equal to its event-time active snapshot.

Final focused command:

```text
python -m pytest tests/agent_memory/test_timeline.py tests/agent_memory/test_sqlite_timeline.py -q
```

Fresh final result: `52 passed in 59.51s`.

### Full agent-memory verification

Command:

```text
python -m pytest tests/agent_memory -q
```

Fresh final result: `222 passed in 77.73s`.

Additional final checks:

```text
python -m flake8 --extend-ignore=C901 src/mnemocore/agent_memory/timeline.py src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_timeline.py tests/agent_memory/test_sqlite_timeline.py
python -m black --check --line-length 100 src/mnemocore/agent_memory/timeline.py tests/agent_memory/test_timeline.py tests/agent_memory/test_sqlite_timeline.py
python -m py_compile src/mnemocore/agent_memory/timeline.py src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_timeline.py tests/agent_memory/test_sqlite_timeline.py
git diff --check
```

Result: all exited 0. C901 was excluded because the existing store already contains deliberately
transactional `_forget` and `_rebuild` functions above the configured McCabe threshold; no other
Flake8 finding remained.

## Policy and payload invariants

- `normalize_timeline_query` rejects simultaneous `as_of` and `valid_at`, treats `as_of` as the
  legacy valid-time alias, defaults missing valid/known boundaries independently to the supplied
  aware `now`, and emits UTC `YYYY-MM-DDTHH:MM:SS.ffffffZ` values.
- Half-open interval selection is `[start, end)`, including exact microsecond exclusion of the old
  fact and inclusion of its replacement at the effective boundary.
- `build_superseded_payload` includes complete source and replacement snapshots: record id, every
  normalized scope field and canonical scope key, kind, content, detached metadata, status,
  confidence, all temporal fields, effective boundary, normalized reason, relation id/type and
  endpoints, and evidence endpoints.
- `parse_superseded_payload` requires exact field sets, canonical timestamps, normalized scope
  fields, recomputed scope keys, fact/status/boundary consistency, event/source ownership, relation
  endpoints, and evidence endpoints. It returns detached frozen `MemoryRecord` values and wraps
  corruption in path-bearing `StorageError` with the original cause.

## Transaction boundary and write order

`SQLiteMemoryStore.supersede` runs one worker under the store's existing cancellation-shielded
lifecycle lock. The worker opens one connection and executes:

1. `BEGIN IMMEDIATE`.
2. Exact-scope idempotency lookup before validating any retry payload/source input.
3. Exact-scope source lookup and complete new-input/model/payload serialization validation before
   the first mutation.
4. Insert the one immutable `SUPERSEDED` event.
5. Conditionally update the active fact source to `superseded` and close its valid interval.
6. Insert the active replacement fact.
7. Close the source active knowledge interval.
8. Insert the open source-superseded and replacement-active lifecycle intervals.
9. Insert deterministic `{event}:history:source` and `{event}:history:replacement` rows.
10. Insert the replacement-to-source evidence row.
11. Insert deterministic `{event}:relation:supersedes` relation.
12. Insert the replacement FTS row while preserving the source FTS row.
13. Commit once. Every exception rolls the full transaction back.

The rollback suite injects failure after each of the eleven projection boundaries and compares the
complete ledger, memory, lifecycle, history, evidence, relation, and FTS row snapshots. All eleven
cases restore the exact pre-operation snapshot.

## Concurrency results

- Two independently opened stores racing with the same exact-scope idempotency key serialize under
  `BEGIN IMMEDIATE`; both return the same replacement and the database contains one supersession
  event and two memory projections.
- Two stores racing with different keys serialize to one successful transition and one
  `MemoryConflictError`; the loser observes the now-non-active source without adding any row.
- A prior key remains idempotent even if its replacement is later superseded: retry validates the
  original immutable event and returns that replacement's current projection.

## Files

- `src/mnemocore/agent_memory/timeline.py`: pure standard-library timeline normalization,
  half-open interval policy, complete payload builder, and strict replay parser.
- `src/mnemocore/agent_memory/sqlite_store.py`: cancellation-safe public `supersede` method and one-
  transaction exact-scope worker.
- `tests/agent_memory/test_timeline.py`: canonical time, alias, boundary, reason, snapshot,
  detachment, and adversarial replay tests.
- `tests/agent_memory/test_sqlite_timeline.py`: projection shape, input/conflict/idempotency,
  rollback, chained retry, and two-store concurrency tests.

## Self-review and concerns

- Checked every Task 3 brief item against the final diff and kept `_rebuild` unchanged; full
  supersession replay/rebuild remains intentionally reserved for Task 5.
- Existing source FTS retention is deliberate and covered; later temporal recall can select using
  lifecycle/validity policy instead of destroying historical lexical content.
- No known blocking concern. The store's single transaction is intentionally explicit and exceeds
  the repository McCabe threshold, as do its pre-existing foundation transaction workers.
