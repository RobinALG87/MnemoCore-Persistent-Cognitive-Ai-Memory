"""Cross-process coordination and physical SQLite memory erasure."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from uuid import uuid4

from .errors import MemoryConflictError, MemoryNotFoundError, StorageError, ValidationError


@dataclass(frozen=True, slots=True)
class ErasureReceipt:
    """Content-free evidence that a physical erasure completed."""

    scope_key: str
    memory_ids: tuple[str, ...]


@contextmanager
def database_operation_lock(path: Path) -> Iterator[None]:
    """Hold the database's cooperative, cross-process exclusive operation lock."""

    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
    finally:
        handle.close()


def _component(connection: sqlite3.Connection, scope_key: str, requested: set[str]) -> set[str]:
    component = set(requested)
    changed = True
    while changed:
        changed = False
        for source_id, target_id in connection.execute(
            """
            SELECT source_id, target_id FROM memory_relations
            WHERE scope_key = ? AND relation_type = 'supersedes'
            """,
            (scope_key,),
        ):
            if source_id in component or target_id in component:
                before = len(component)
                component.update((source_id, target_id))
                changed = changed or len(component) != before
    return component


def physically_erase(
    path: Path,
    scope_key: str,
    memory_ids: Sequence[str],
    *,
    cascade: bool = False,
) -> ErasureReceipt:
    """Rewrite the database without selected exact-scope memory streams."""

    requested = tuple(dict.fromkeys(memory_ids))
    if not requested or any(not isinstance(value, str) or not value.strip() for value in requested):
        raise ValidationError("memory_ids must contain at least one non-empty id")
    temporary = path.with_name(f".{path.name}.erase-{uuid4().hex}.tmp")
    try:
        with closing(sqlite3.connect(path, timeout=10)) as source:
            source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            placeholders = ",".join("?" for _ in requested)
            owned = {
                row[0]
                for row in source.execute(
                    f"SELECT id FROM memories WHERE scope_key = ? AND id IN ({placeholders})",
                    (scope_key, *requested),
                )
            }
            if owned != set(requested):
                raise MemoryNotFoundError("One or more memories were not found in this scope")
            connected = _component(source, scope_key, owned)
            if connected != owned and not cascade:
                raise MemoryConflictError(
                    "Erasure would split a supersession component; pass cascade=True"
                )
            erased = connected if cascade else owned
            target = sqlite3.connect(temporary)
            try:
                source.backup(target)
            finally:
                target.close()

        with closing(sqlite3.connect(temporary)) as rewritten:
            rewritten.execute("PRAGMA foreign_keys=OFF")
            rewritten.execute("BEGIN IMMEDIATE")
            immutable_triggers = tuple(
                row[0]
                for row in rewritten.execute(
                    """
                    SELECT sql FROM sqlite_master
                    WHERE type = 'trigger'
                      AND name IN (
                        'trg_memory_events_immutable_delete',
                        'trg_memory_events_immutable_update'
                      )
                    ORDER BY name
                    """
                )
                if row[0] is not None
            )
            rewritten.execute("DROP TRIGGER IF EXISTS trg_memory_events_immutable_delete")
            rewritten.execute("DROP TRIGGER IF EXISTS trg_memory_events_immutable_update")
            marks = ",".join("?" for _ in erased)
            params = tuple(sorted(erased))
            event_ids = tuple(
                row[0]
                for row in rewritten.execute(
                    f"SELECT id FROM memory_events WHERE memory_id IN ({marks})", params
                )
            )
            rewritten.execute(f"DELETE FROM memory_fts WHERE memory_id IN ({marks})", params)
            rewritten.execute(f"DELETE FROM memory_evidence WHERE memory_id IN ({marks}) OR source_memory_id IN ({marks})", (*params, *params))
            rewritten.execute(f"DELETE FROM memory_relations WHERE source_id IN ({marks}) OR target_id IN ({marks})", (*params, *params))
            rewritten.execute(f"DELETE FROM memory_history WHERE memory_id IN ({marks})", params)
            rewritten.execute(f"DELETE FROM memory_lifecycle WHERE memory_id IN ({marks})", params)
            rewritten.execute(f"DELETE FROM memories WHERE id IN ({marks})", params)
            if event_ids:
                event_marks = ",".join("?" for _ in event_ids)
                rewritten.execute(f"DELETE FROM memory_events WHERE id IN ({event_marks})", event_ids)
            for trigger_sql in immutable_triggers:
                rewritten.execute(trigger_sql)
            rewritten.commit()
            rewritten.execute("VACUUM")
            rewritten.execute("PRAGMA journal_mode=DELETE")
        # Windows requires a writable descriptor for FlushFileBuffers, which
        # backs os.fsync there.
        with temporary.open("r+b") as file_handle:
            os.fsync(file_handle.fileno())
        os.replace(temporary, path)
        for suffix in ("-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        return ErasureReceipt(scope_key=scope_key, memory_ids=tuple(sorted(erased)))
    except (MemoryConflictError, MemoryNotFoundError, ValidationError):
        raise
    except (OSError, sqlite3.Error) as error:
        raise StorageError(f"Failed to physically erase memory from {path}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
