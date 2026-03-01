"""
Backup Manager for MnemoCore
=============================

Provides automated Qdrant snapshots with Write-Ahead Log (WAL) support
for incremental backups. Supports scheduled snapshots, retention policies,
and disaster recovery.

Features:
    - Automated Qdrant snapshots (full collection snapshots)
    - Incremental WAL-based backups
    - Scheduled snapshots with configurable intervals
    - Retention policy management
    - Disaster recovery with snapshot restoration
    - Backup verification and integrity checking

Usage:
    ```python
    from mnemocore.storage import BackupManager
    from mnemocore.core.qdrant_store import QdrantStore

    qdrant = QdrantStore(url="http://localhost:6333", ...)
    backup_mgr = BackupManager(qdrant, backup_dir="./backups")

    # Create a full snapshot
    snapshot = await backup_mgr.create_snapshot("haim_hot")

    # List available snapshots
    snapshots = await backup_mgr.list_snapshots()

    # Restore from snapshot
    await backup_mgr.restore_snapshot(snapshot_id)
    ```
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import aiosqlite
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import hashlib

import numpy as np
from loguru import logger

from mnemocore.core.exceptions import (
    StorageConnectionError,
    StorageError,
    ValidationError,
    wrap_storage_exception,
)
from mnemocore.utils.json_compat import dumps, loads


# =============================================================================
# Configuration Data Classes
# =============================================================================


class SnapshotStatus(Enum):
    """Status of a snapshot operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"


@dataclass
class SnapshotInfo:
    """Metadata about a snapshot."""
    snapshot_id: str
    collection_name: str
    created_at: datetime
    status: SnapshotStatus
    size_bytes: int
    point_count: int
    is_incremental: bool
    parent_snapshot_id: Optional[str] = None
    checksum: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "collection_name": self.collection_name,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "size_bytes": self.size_bytes,
            "point_count": self.point_count,
            "is_incremental": self.is_incremental,
            "parent_snapshot_id": self.parent_snapshot_id,
            "checksum": self.checksum,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotInfo":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            collection_name=data["collection_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            status=SnapshotStatus(data["status"]),
            size_bytes=data["size_bytes"],
            point_count=data["point_count"],
            is_incremental=data["is_incremental"],
            parent_snapshot_id=data.get("parent_snapshot_id"),
            checksum=data.get("checksum"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BackupConfig:
    """Configuration for backup operations."""
    # Snapshot settings
    auto_snapshot_enabled: bool = True
    snapshot_interval_hours: int = 24
    max_snapshots: int = 7
    compression_enabled: bool = True

    # WAL settings
    wal_enabled: bool = True
    wal_flush_interval_seconds: int = 300  # 5 minutes
    wal_max_size_mb: int = 100

    # Storage settings
    backup_dir: str = "./backups"
    verify_checksums: bool = True
    parallel_snapshot_threads: int = 1

    # Retention policy
    retention_days: int = 30
    keep_daily: int = 7
    keep_weekly: int = 4
    keep_monthly: int = 12

    # Recovery settings
    restore_timeout_seconds: int = 300
    verify_after_restore: bool = True


@dataclass
class RecoverableBackup:
    """A backup that can be restored."""
    snapshot_id: str
    snapshot_info: SnapshotInfo
    file_path: Path
    checksum: str
    is_valid: bool = True


# =============================================================================
# Write-Ahead Log (WAL)
# =============================================================================


class WriteAheadLog:
    """
    Write-Ahead Log for incremental backup of point operations.

    Tracks all modifications (inserts, updates, deletes) to enable
    incremental backups and point-in-time recovery.
    """

    def __init__(
        self,
        wal_path: Union[str, Path],
        collection_name: str,
        max_size_mb: int = 100,
    ):
        self.wal_path = Path(wal_path)
        self.collection_name = collection_name
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _init_db(self):
        """Initialize WAL database asynchronously."""
        if self._initialized:
            return

        self.wal_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(
            str(self.wal_path),
            timeout=30.0,
        )
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS wal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                operation TEXT NOT NULL,
                point_id TEXT NOT NULL,
                point_data BLOB,
                metadata TEXT
            )
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON wal_entries(timestamp)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_operation
            ON wal_entries(operation)
        """)

        await self._conn.commit()
        self._initialized = True

    async def log_insert(
        self,
        point_id: str,
        point_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log an insert operation."""
        return await self._log_operation(
            operation="insert",
            point_id=point_id,
            point_data=point_data,
            metadata=metadata,
        )

    async def log_update(
        self,
        point_id: str,
        point_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log an update operation."""
        return await self._log_operation(
            operation="update",
            point_id=point_id,
            point_data=point_data,
            metadata=metadata,
        )

    async def log_delete(
        self,
        point_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log a delete operation."""
        return await self._log_operation(
            operation="delete",
            point_id=point_id,
            point_data=None,
            metadata=metadata,
        )

    async def _log_operation(
        self,
        operation: str,
        point_id: str,
        point_data: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Internal method to log an operation."""
        async with self._lock:
            if not self._initialized:
                await self._init_db()

            if not self._conn:
                logger.warning("WAL connection not initialized")
                return False

            # Check size limit
            if await self._get_size() >= self.max_size_bytes:
                await self._rotate_wal()

            try:
                data_blob = None
                if point_data:
                    data_blob = dumps(point_data).encode('utf-8')

                metadata_json = dumps(metadata or {})

                await self._conn.execute(
                    """
                    INSERT INTO wal_entries
                    (timestamp, operation, point_id, point_data, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        int(datetime.now(timezone.utc).timestamp()),
                        operation,
                        point_id,
                        data_blob,
                        metadata_json,
                    ),
                )
                await self._conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to log WAL operation: {e}")
                return False

    async def _get_size(self) -> int:
        """Get current WAL size in bytes."""
        if not self._conn:
            return 0
        try:
            cursor = await self._conn.execute(
                "SELECT SUM(LENGTH(point_data)) FROM wal_entries"
            )
            result = await cursor.fetchone()
            return result[0] or 0
        except Exception as e:
            logger.warning(f"Failed to get WAL size: {e}")
            return 0

    async def _rotate_wal(self):
        """Rotate WAL when size limit is reached."""
        if not self._conn:
            return

        logger.info(f"Rotating WAL for {self.collection_name}")

        # Archive current WAL
        archive_path = self.wal_path.with_suffix(
            f".{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.bak"
        )
        shutil.copy2(self.wal_path, archive_path)

        # Clear current WAL
        await self._conn.execute("DELETE FROM wal_entries")
        await self._conn.commit()

    async def get_entries_since(
        self,
        since_timestamp: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get WAL entries since a timestamp."""
        if not self._initialized:
            await self._init_db()

        if not self._conn:
            return []

        try:
            params: list = [since_timestamp]
            query = """
                SELECT timestamp, operation, point_id, point_data, metadata
                FROM wal_entries
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            cursor = await self._conn.execute(query, tuple(params))
            entries = []

            async for row in cursor:
                point_data = None
                if row[3]:
                    point_data = loads(row[3])

                metadata = loads(row[4]) if row[4] else {}

                entries.append({
                    "timestamp": row[0],
                    "operation": row[1],
                    "point_id": row[2],
                    "point_data": point_data,
                    "metadata": metadata,
                })

            return entries
        except Exception as e:
            logger.error(f"Failed to get WAL entries: {e}")
            return []

    async def truncate(self, before_timestamp: int) -> int:
        """Remove WAL entries before a timestamp."""
        if not self._initialized:
            await self._init_db()

        if not self._conn:
            return 0

        try:
            cursor = await self._conn.execute(
                "DELETE FROM wal_entries WHERE timestamp < ?",
                (before_timestamp,),
            )
            await self._conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Failed to truncate WAL: {e}")
            return 0

    async def close(self):
        """Close WAL connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False


# =============================================================================
# Backup Manager
# =============================================================================


class BackupManager:
    """
    Manages Qdrant snapshots with WAL support.

    Features:
        - Automated full snapshots of Qdrant collections
        - Incremental backups using WAL
        - Scheduled snapshots with configurable intervals
        - Retention policy enforcement
        - Snapshot verification and restoration
    """

    def __init__(
        self,
        qdrant_store: Any,  # QdrantStore (avoid circular import)
        config: Optional[BackupConfig] = None,
    ):
        """
        Initialize BackupManager.

        Args:
            qdrant_store: QdrantStore instance
            config: Backup configuration (uses defaults if None)
        """
        self.qdrant = qdrant_store
        self.config = config or BackupConfig()

        # Setup backup directory
        self.backup_dir = Path(self.config.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Snapshots database
        self.snapshots_db_path = self.backup_dir / "snapshots.db"
        self._snapshots_conn: Optional[aiosqlite.Connection] = None
        self._snapshots_initialized = False

        # WAL instances per collection
        self._wals: Dict[str, WriteAheadLog] = {}

        # Snapshot task
        self._snapshot_task: Optional[asyncio.Task] = None
        self._running = False

    async def _init_snapshots_db(self):
        """Initialize snapshots metadata database asynchronously."""
        if self._snapshots_initialized:
            return

        self._snapshots_conn = await aiosqlite.connect(
            str(self.snapshots_db_path)
        )
        await self._snapshots_conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                point_count INTEGER NOT NULL,
                is_incremental INTEGER NOT NULL,
                parent_snapshot_id TEXT,
                checksum TEXT,
                description TEXT,
                metadata TEXT
            )
        """)

        await self._snapshots_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_collection
            ON snapshots(collection_name, created_at DESC)
        """)

        await self._snapshots_conn.commit()
        self._snapshots_initialized = True

    # -------------------------------------------------------------------------
    # Snapshot Management
    # -------------------------------------------------------------------------

    async def create_snapshot(
        self,
        collection_name: str,
        description: Optional[str] = None,
        force_full: bool = False,
    ) -> SnapshotInfo:
        """
        Create a snapshot of a Qdrant collection.

        Args:
            collection_name: Name of the collection to snapshot
            description: Optional description of the snapshot
            force_full: Force a full snapshot even if incremental is available

        Returns:
            SnapshotInfo with metadata about the created snapshot
        """
        snapshot_id = self._generate_snapshot_id(collection_name)
        logger.info(f"Creating snapshot {snapshot_id} for collection {collection_name}")

        try:
            # Get collection info
            collection_info = await self.qdrant.get_collection_info(collection_name)
            if not collection_info:
                raise StorageError(
                    f"Collection {collection_name} not found",
                    {"collection": collection_name}
                )

            point_count = collection_info.points_count or 0
            status = SnapshotStatus.IN_PROGRESS

            # Determine if incremental
            is_incremental = False
            parent_id = None
            if not force_full and self.config.wal_enabled:
                latest = await self.get_latest_snapshot(collection_name)
                if latest:
                    is_incremental = True
                    parent_id = latest.snapshot_id

            # Create snapshot directory
            snapshot_dir = self.backup_dir / "snapshots" / snapshot_id
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Export points
            export_file = snapshot_dir / "points.jsonl"
            size_bytes = await self._export_collection_points(
                collection_name,
                export_file,
            )

            # Calculate checksum
            checksum = None
            if self.config.verify_checksums:
                checksum = self._calculate_file_checksum(export_file)

            # Create snapshot info
            snapshot_info = SnapshotInfo(
                snapshot_id=snapshot_id,
                collection_name=collection_name,
                created_at=datetime.now(timezone.utc),
                status=status,
                size_bytes=size_bytes,
                point_count=point_count,
                is_incremental=is_incremental,
                parent_snapshot_id=parent_id,
                checksum=checksum,
                description=description,
                metadata={
                    "config": {
                        "compression": self.config.compression_enabled,
                        "wal_enabled": self.config.wal_enabled,
                    }
                },
            )

            # Save to database
            await self._save_snapshot_info(snapshot_info)

            # Update status to completed
            snapshot_info.status = SnapshotStatus.COMPLETED
            await self._save_snapshot_info(snapshot_info)

            logger.info(
                f"Snapshot {snapshot_id} completed: "
                f"{point_count} points, {size_bytes} bytes"
            )

            # Enforce retention policy
            await self._enforce_retention_policy(collection_name)

            return snapshot_info

        except Exception as e:
            logger.error(f"Snapshot creation failed: {e}")
            # Mark as failed
            if snapshot_id:
                await self._update_snapshot_status(
                    snapshot_id, SnapshotStatus.FAILED
                )
            raise wrap_storage_exception("backup", "create_snapshot", e)

    async def _export_collection_points(
        self,
        collection_name: str,
        output_path: Path,
        batch_size: int = 100,
    ) -> int:
        """
        Export all points from a collection to a file.

        Args:
            collection_name: Collection to export
            output_path: Output file path
            batch_size: Batch size for scrolling

        Returns:
            Total size in bytes
        """
        total_bytes = 0
        offset = None
        point_count = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            while True:
                records, offset = await self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_vectors=True,
                )

                if not records:
                    break

                for record in records:
                    # Serialize point
                    point_data = self._serialize_point(record)
                    line = dumps(point_data) + '\n'
                    f.write(line)
                    total_bytes += len(line.encode('utf-8'))
                    point_count += 1

                logger.debug(f"Exported {point_count} points from {collection_name}")

        return total_bytes

    def _serialize_point(self, record: Any) -> Dict[str, Any]:
        """Serialize a Qdrant record to a dictionary."""
        result = {
            "id": str(record.id),
            "payload": record.payload or {},
        }

        # Handle vector
        if hasattr(record, 'vector') and record.vector is not None:
            if isinstance(record.vector, dict):
                # Named vectors
                result["vector"] = {
                    k: v.tolist() if hasattr(v, 'tolist') else v
                    for k, v in record.vector.items()
                }
            else:
                # Single vector
                result["vector"] = (
                    record.vector.tolist()
                    if hasattr(record.vector, 'tolist')
                    else record.vector
                )

        # Handle shard key if present
        if hasattr(record, 'shard_key') and record.shard_key is not None:
            result["shard_key"] = record.shard_key

        return result

    async def restore_snapshot(
        self,
        snapshot_id: str,
        target_collection: Optional[str] = None,
        verify_after: bool = True,
    ) -> bool:
        """
        Restore a snapshot to a collection.

        Args:
            snapshot_id: ID of the snapshot to restore
            target_collection: Target collection name (defaults to original)
            verify_after: Verify data after restoration

        Returns:
            True if restoration succeeded
        """
        snapshot_info = await self.get_snapshot_info(snapshot_id)
        if not snapshot_info:
            raise ValidationError(
                "snapshot_id",
                f"Snapshot {snapshot_id} not found",
            )

        target = target_collection or snapshot_info.collection_name
        logger.info(f"Restoring snapshot {snapshot_id} to collection {target}")

        try:
            snapshot_dir = self.backup_dir / "snapshots" / snapshot_id
            points_file = snapshot_dir / "points.jsonl"

            if not points_file.exists():
                raise ValidationError(
                    "snapshot_file",
                    f"Snapshot file not found: {points_file}",
                )

            # Verify checksum if available
            if snapshot_info.checksum and self.config.verify_checksums:
                calculated = self._calculate_file_checksum(points_file)
                if calculated != snapshot_info.checksum:
                    snapshot_info.status = SnapshotStatus.CORRUPTED
                    await self._save_snapshot_info(snapshot_info)
                    raise ValidationError(
                        "checksum",
                        f"Snapshot checksum mismatch: expected {snapshot_info.checksum}, got {calculated}",
                    )

            # Read and restore points
            await self._restore_points_from_file(target, points_file)

            # Verify if requested
            if verify_after and self.config.verify_after_restore:
                await self._verify_restored_data(target, snapshot_info)

            logger.info(f"Snapshot {snapshot_id} restored successfully to {target}")
            return True

        except Exception as e:
            logger.error(f"Snapshot restoration failed: {e}")
            raise wrap_storage_exception("backup", "restore_snapshot", e)

    async def _restore_points_from_file(
        self,
        collection_name: str,
        points_file: Path,
        batch_size: int = 100,
    ):
        """Restore points from a file to a collection."""
        from qdrant_client import models

        batch = []
        total_restored = 0

        with open(points_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                point_data = loads(line)
                point_struct = self._deserialize_point(point_data)
                batch.append(point_struct)

                if len(batch) >= batch_size:
                    await self.qdrant.upsert(collection_name, batch)
                    total_restored += len(batch)
                    batch = []
                    logger.debug(f"Restored {total_restored} points to {collection_name}")

            # Restore remaining
            if batch:
                await self.qdrant.upsert(collection_name, batch)
                total_restored += len(batch)

        logger.info(f"Restored {total_restored} points to {collection_name}")

    def _deserialize_point(self, data: Dict[str, Any]) -> Any:
        """Deserialize a dictionary to a Qdrant PointStruct."""
        from qdrant_client import models

        # Convert vector back to list if it's a numpy array
        vector = data.get("vector")
        if vector is not None:
            if isinstance(vector, dict):
                vector = {k: list(v) for k, v in vector.items()}
            else:
                vector = list(vector)

        return models.PointStruct(
            id=data["id"],
            vector=vector,
            payload=data.get("payload", {}),
        )

    async def _verify_restored_data(
        self,
        collection_name: str,
        snapshot_info: SnapshotInfo,
    ) -> bool:
        """Verify that restored data matches snapshot."""
        collection_info = await self.qdrant.get_collection_info(collection_name)
        actual_count = collection_info.points_count or 0

        if actual_count != snapshot_info.point_count:
            logger.warning(
                f"Verification warning: expected {snapshot_info.point_count} "
                f"points, got {actual_count}"
            )
            return False

        return True

    async def list_snapshots(
        self,
        collection_name: Optional[str] = None,
        status: Optional[SnapshotStatus] = None,
        limit: int = 100,
    ) -> List[SnapshotInfo]:
        """
        List snapshots with optional filters.

        Args:
            collection_name: Filter by collection name
            status: Filter by status
            limit: Maximum number of snapshots to return

        Returns:
            List of SnapshotInfo objects
        """
        if not self._snapshots_initialized:
            await self._init_snapshots_db()

        snapshots = []

        try:
            self._snapshots_conn.row_factory = aiosqlite.Row

            query = "SELECT * FROM snapshots WHERE 1=1"
            params = []

            if collection_name:
                query += " AND collection_name = ?"
                params.append(collection_name)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = await self._snapshots_conn.execute(query, params)

            async for row in cursor:
                snapshots.append(SnapshotInfo.from_dict(dict(row)))

        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")

        return snapshots

    async def get_snapshot_info(self, snapshot_id: str) -> Optional[SnapshotInfo]:
        """Get information about a specific snapshot."""
        if not self._snapshots_initialized:
            await self._init_snapshots_db()

        try:
            self._snapshots_conn.row_factory = aiosqlite.Row

            cursor = await self._snapshots_conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            row = await cursor.fetchone()

            if row:
                return SnapshotInfo.from_dict(dict(row))
        except Exception as e:
            logger.error(f"Failed to get snapshot info: {e}")

        return None

    async def get_latest_snapshot(
        self,
        collection_name: str,
        status: SnapshotStatus = SnapshotStatus.COMPLETED,
    ) -> Optional[SnapshotInfo]:
        """Get the latest snapshot for a collection."""
        snapshots = await self.list_snapshots(
            collection_name=collection_name,
            status=status,
            limit=1,
        )
        return snapshots[0] if snapshots else None

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot and its files.

        Args:
            snapshot_id: ID of the snapshot to delete

        Returns:
            True if deletion succeeded, False if snapshot not found
        """
        logger.info(f"Deleting snapshot {snapshot_id}")

        if not self._snapshots_initialized:
            await self._init_snapshots_db()

        # Check if snapshot exists first
        try:
            cursor = await self._snapshots_conn.execute(
                "SELECT snapshot_id FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
        except Exception as e:
            logger.error(f"Failed to check snapshot existence: {e}")
            return False

        try:
            # Delete files
            snapshot_dir = self.backup_dir / "snapshots" / snapshot_id
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)

            # Delete from database
            await self._snapshots_conn.execute(
                "DELETE FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            await self._snapshots_conn.commit()

            return True
        except Exception as e:
            logger.error(f"Failed to delete snapshot: {e}")
            return False

    # -------------------------------------------------------------------------
    # WAL Integration
    # -------------------------------------------------------------------------

    def get_wal(self, collection_name: str) -> WriteAheadLog:
        """Get or create WAL for a collection."""
        if collection_name not in self._wals:
            wal_path = (
                self.backup_dir
                / "wal"
                / f"{collection_name}.wal"
            )
            self._wals[collection_name] = WriteAheadLog(
                wal_path=wal_path,
                collection_name=collection_name,
                max_size_mb=self.config.wal_max_size_mb,
            )
        return self._wals[collection_name]

    async def log_operation(
        self,
        collection_name: str,
        operation: str,
        point_id: str,
        point_data: Optional[Dict[str, Any]] = None,
    ):
        """Log a write operation to WAL."""
        if not self.config.wal_enabled:
            return

        wal = self.get_wal(collection_name)

        if operation == "delete":
            await wal.log_delete(point_id)
        elif operation == "update":
            await wal.log_update(point_id, point_data)
        else:  # insert
            await wal.log_insert(point_id, point_data)

    async def get_incremental_changes(
        self,
        collection_name: str,
        since_snapshot_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get incremental changes since a snapshot.

        Args:
            collection_name: Collection name
            since_snapshot_id: Base snapshot ID

        Returns:
            List of WAL entries since the snapshot
        """
        snapshot = await self.get_snapshot_info(since_snapshot_id)
        if not snapshot:
            return []

        since_ts = int(snapshot.created_at.timestamp())
        wal = self.get_wal(collection_name)

        return await wal.get_entries_since(since_ts)

    # -------------------------------------------------------------------------
    # Scheduled Snapshots
    # -------------------------------------------------------------------------

    async def start_scheduled_snapshots(self):
        """Start background task for scheduled snapshots."""
        if self._running:
            logger.warning("Scheduled snapshots already running")
            return

        self._running = True
        interval_seconds = self.config.snapshot_interval_hours * 3600

        logger.info(
            f"Starting scheduled snapshots (interval: {interval_seconds}s)"
        )

        async def snapshot_loop():
            while self._running:
                try:
                    await self._create_scheduled_snapshots()
                except Exception as e:
                    logger.error(f"Scheduled snapshot error: {e}")

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

        self._snapshot_task = asyncio.create_task(snapshot_loop())

    async def stop_scheduled_snapshots(self):
        """Stop background scheduled snapshots."""
        self._running = False
        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
            self._snapshot_task = None

        logger.info("Stopped scheduled snapshots")

    async def _create_scheduled_snapshots(self):
        """Create snapshots for all configured collections."""
        collections = [
            self.qdrant.collection_hot,
            self.qdrant.collection_warm,
        ]

        for collection in collections:
            try:
                latest = await self.get_latest_snapshot(collection)
                should_snapshot = False

                if not latest:
                    should_snapshot = True
                else:
                    age = datetime.now(timezone.utc) - latest.created_at
                    if age >= timedelta(hours=self.config.snapshot_interval_hours):
                        should_snapshot = True

                if should_snapshot:
                    description = f"Auto-snapshot at {datetime.now(timezone.utc).isoformat()}"
                    await self.create_snapshot(collection, description)

            except Exception as e:
                logger.error(f"Failed to snapshot {collection}: {e}")

    # -------------------------------------------------------------------------
    # Retention Policy
    # -------------------------------------------------------------------------

    async def _enforce_retention_policy(self, collection_name: str):
        """Enforce retention policy for a collection's snapshots."""
        snapshots = await self.list_snapshots(
            collection_name=collection_name,
            status=SnapshotStatus.COMPLETED,
        )

        if not snapshots:
            return

        to_delete = set()
        now = datetime.now(timezone.utc)

        # Apply max snapshots limit
        if len(snapshots) > self.config.max_snapshots:
            for snapshot in snapshots[self.config.max_snapshots:]:
                to_delete.add(snapshot.snapshot_id)

        # Apply age-based retention
        for snapshot in snapshots:
            age = now - snapshot.created_at
            if age.days > self.config.retention_days:
                to_delete.add(snapshot.snapshot_id)

        # Delete old snapshots
        for snapshot_id in to_delete:
            await self.delete_snapshot(snapshot_id)
            logger.info(f"Deleted old snapshot {snapshot_id} (retention policy)")

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _generate_snapshot_id(self, collection_name: str) -> str:
        """Generate a unique snapshot ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand = os.urandom(4).hex()
        return f"{collection_name}_{timestamp}_{rand}"

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        return sha256.hexdigest()

    async def _save_snapshot_info(self, snapshot_info: SnapshotInfo):
        """Save snapshot info to database."""
        if not self._snapshots_initialized:
            await self._init_snapshots_db()

        try:
            await self._snapshots_conn.execute(
                """
                INSERT OR REPLACE INTO snapshots
                (snapshot_id, collection_name, created_at, status, size_bytes,
                 point_count, is_incremental, parent_snapshot_id, checksum,
                 description, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_info.snapshot_id,
                    snapshot_info.collection_name,
                    snapshot_info.created_at.isoformat(),
                    snapshot_info.status.value,
                    snapshot_info.size_bytes,
                    snapshot_info.point_count,
                    int(snapshot_info.is_incremental),
                    snapshot_info.parent_snapshot_id,
                    snapshot_info.checksum,
                    snapshot_info.description,
                    dumps(snapshot_info.metadata),
                ),
            )
            await self._snapshots_conn.commit()
        except Exception as e:
            logger.error(f"Failed to save snapshot info: {e}")

    async def _update_snapshot_status(
        self,
        snapshot_id: str,
        status: SnapshotStatus,
    ):
        """Update snapshot status in database."""
        if not self._snapshots_initialized:
            await self._init_snapshots_db()

        try:
            await self._snapshots_conn.execute(
                "UPDATE snapshots SET status = ? WHERE snapshot_id = ?",
                (status.value, snapshot_id),
            )
            await self._snapshots_conn.commit()
        except Exception as e:
            logger.error(f"Failed to update snapshot status: {e}")

    async def close(self):
        """Clean up resources."""
        await self.stop_scheduled_snapshots()

        for wal in self._wals.values():
            await wal.close()
        self._wals.clear()

        if self._snapshots_conn:
            await self._snapshots_conn.close()
            self._snapshots_conn = None
            self._snapshots_initialized = False

        logger.info("BackupManager closed")


# =============================================================================
# Convenience Functions
# =============================================================================


async def create_backup_manager(
    qdrant_store: Any,
    backup_dir: str = "./backups",
    **config_kwargs,
) -> BackupManager:
    """
    Create and initialize a BackupManager.

    Args:
        qdrant_store: QdrantStore instance
        backup_dir: Directory for backup storage
        **config_kwargs: Additional configuration options

    Returns:
        Initialized BackupManager instance
    """
    config = BackupConfig(backup_dir=backup_dir, **config_kwargs)
    return BackupManager(qdrant_store, config)
