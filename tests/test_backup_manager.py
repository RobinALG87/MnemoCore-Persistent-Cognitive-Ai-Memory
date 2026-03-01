"""
Tests for Backup Manager Module
================================

Tests for the WriteAheadLog and BackupManager components that provide
automated Qdrant snapshots with WAL support for incremental backups.

Coverage:
    - WriteAheadLog: insert/update/delete/clear entries
    - WAL rotation and replay
    - BackupManager.create_snapshot() - full and incremental
    - BackupManager.restore_snapshot() - happy path and corrupt file
    - list_snapshots(), get_snapshot_info(), delete_snapshot()
    - Concurrent WAL writes
"""

import pytest
import asyncio
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from mnemocore.storage.backup_manager import (
    WriteAheadLog,
    BackupManager,
    BackupConfig,
    SnapshotInfo,
    SnapshotStatus,
    RecoverableBackup,
    create_backup_manager,
)
from mnemocore.utils.json_compat import dumps, loads


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_backup_dir(tmp_path):
    """Create a temporary directory for backup testing."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


@pytest.fixture
def temp_wal_path(temp_backup_dir):
    """Create a temporary WAL path."""
    wal_dir = temp_backup_dir / "wal"
    wal_dir.mkdir(parents=True, exist_ok=True)
    return wal_dir / "test_collection.wal"


@pytest.fixture
def backup_config(temp_backup_dir):
    """Create a BackupConfig with temporary directory."""
    return BackupConfig(
        backup_dir=str(temp_backup_dir),
        wal_enabled=True,
        wal_max_size_mb=10,
        max_snapshots=5,
        retention_days=30,
        verify_checksums=True,
    )


@pytest.fixture
def mock_qdrant_store():
    """Create a mock QdrantStore for testing."""
    mock = MagicMock()
    mock.collection_hot = "haim_hot"
    mock.collection_warm = "haim_warm"

    # Mock collection info
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 100
    mock.get_collection_info = AsyncMock(return_value=mock_collection_info)

    # Mock scroll for exporting points
    mock.scroll = AsyncMock(return_value=([], None))

    # Mock upsert for restoring
    mock.upsert = AsyncMock(return_value=None)

    return mock


@pytest.fixture
async def wal(temp_wal_path):
    """Create and initialize a WriteAheadLog for testing."""
    wal = WriteAheadLog(
        wal_path=temp_wal_path,
        collection_name="test_collection",
        max_size_mb=10,
    )
    yield wal
    await wal.close()


@pytest.fixture
async def backup_manager(mock_qdrant_store, backup_config):
    """Create and initialize a BackupManager for testing."""
    manager = BackupManager(
        qdrant_store=mock_qdrant_store,
        config=backup_config,
    )
    yield manager
    await manager.close()


# =============================================================================
# WriteAheadLog Tests
# =============================================================================

class TestWriteAheadLogInit:
    """Tests for WriteAheadLog initialization."""

    def test_wal_init(self, temp_wal_path):
        """WAL initializes with correct path and settings."""
        wal = WriteAheadLog(
            wal_path=temp_wal_path,
            collection_name="test_collection",
            max_size_mb=10,
        )
        assert wal.collection_name == "test_collection"
        assert wal.max_size_bytes == 10 * 1024 * 1024
        assert wal._initialized is False

    @pytest.mark.asyncio
    async def test_wal_lazy_init(self, wal):
        """WAL initializes database on first operation."""
        # Initialization happens on first operation
        result = await wal.log_insert(
            point_id="test_id",
            point_data={"vector": [0.1, 0.2, 0.3], "payload": {"key": "value"}},
        )
        assert result is True
        assert wal._initialized is True


class TestWriteAheadLogInsert:
    """Tests for WAL insert operations."""

    @pytest.mark.asyncio
    async def test_log_insert_basic(self, wal):
        """Basic insert operation is logged correctly."""
        point_data = {
            "vector": [0.1, 0.2, 0.3, 0.4],
            "payload": {"content": "test content", "tier": "hot"},
        }

        result = await wal.log_insert(
            point_id="point_001",
            point_data=point_data,
            metadata={"source": "test"},
        )

        assert result is True

        # Verify entry was logged
        entries = await wal.get_entries_since(0)
        assert len(entries) == 1
        assert entries[0]["operation"] == "insert"
        assert entries[0]["point_id"] == "point_001"
        assert entries[0]["point_data"] == point_data

    @pytest.mark.asyncio
    async def test_log_insert_without_metadata(self, wal):
        """Insert without metadata works correctly."""
        result = await wal.log_insert(
            point_id="point_002",
            point_data={"vector": [1.0, 2.0]},
        )

        assert result is True
        entries = await wal.get_entries_since(0)
        assert len(entries) == 1
        assert entries[0]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_log_insert_large_payload(self, wal):
        """Insert with large payload is handled correctly."""
        large_payload = {"data": "x" * 10000}
        result = await wal.log_insert(
            point_id="point_large",
            point_data=large_payload,
        )

        assert result is True
        entries = await wal.get_entries_since(0)
        assert len(entries[0]["point_data"]["data"]) == 10000


class TestWriteAheadLogUpdate:
    """Tests for WAL update operations."""

    @pytest.mark.asyncio
    async def test_log_update_basic(self, wal):
        """Basic update operation is logged correctly."""
        point_data = {
            "vector": [0.5, 0.6, 0.7],
            "payload": {"updated": True},
        }

        result = await wal.log_update(
            point_id="point_001",
            point_data=point_data,
            metadata={"reason": "content change"},
        )

        assert result is True
        entries = await wal.get_entries_since(0)
        assert len(entries) == 1
        assert entries[0]["operation"] == "update"

    @pytest.mark.asyncio
    async def test_log_update_preserves_insert(self, wal):
        """Update after insert preserves both entries."""
        await wal.log_insert("point_001", {"v": [1.0]})
        await wal.log_update("point_001", {"v": [2.0]})

        entries = await wal.get_entries_since(0)
        assert len(entries) == 2
        assert entries[0]["operation"] == "insert"
        assert entries[1]["operation"] == "update"


class TestWriteAheadLogDelete:
    """Tests for WAL delete operations."""

    @pytest.mark.asyncio
    async def test_log_delete_basic(self, wal):
        """Basic delete operation is logged correctly."""
        result = await wal.log_delete(
            point_id="point_001",
            metadata={"reason": "expired"},
        )

        assert result is True
        entries = await wal.get_entries_since(0)
        assert len(entries) == 1
        assert entries[0]["operation"] == "delete"
        assert entries[0]["point_data"] is None

    @pytest.mark.asyncio
    async def test_log_delete_no_point_data(self, wal):
        """Delete operations do not store point data."""
        await wal.log_delete("point_to_delete")

        entries = await wal.get_entries_since(0)
        assert entries[0]["point_data"] is None


class TestWriteAheadLogGetEntries:
    """Tests for WAL entry retrieval."""

    @pytest.mark.asyncio
    async def test_get_entries_since_timestamp(self, wal):
        """Entries are filtered by timestamp correctly."""
        import time

        # Log some entries
        await wal.log_insert("point_001", {"v": [1.0]})
        await wal.log_insert("point_002", {"v": [2.0]})

        # Wait a bit and record timestamp
        await asyncio.sleep(1.1)
        cutoff_time = int(datetime.now().timestamp())

        # Log more entries
        await wal.log_insert("point_003", {"v": [3.0]})
        await wal.log_insert("point_004", {"v": [4.0]})

        # Get entries since cutoff
        entries = await wal.get_entries_since(cutoff_time)
        assert len(entries) == 2
        assert entries[0]["point_id"] == "point_003"
        assert entries[1]["point_id"] == "point_004"

    @pytest.mark.asyncio
    async def test_get_entries_with_limit(self, wal):
        """Entry retrieval respects limit parameter."""
        for i in range(10):
            await wal.log_insert(f"point_{i:03d}", {"v": [float(i)]})

        entries = await wal.get_entries_since(0, limit=5)
        assert len(entries) == 5

    @pytest.mark.asyncio
    async def test_get_entries_empty_wal(self, wal):
        """Empty WAL returns empty list."""
        entries = await wal.get_entries_since(0)
        assert entries == []


class TestWriteAheadLogTruncate:
    """Tests for WAL truncation."""

    @pytest.mark.asyncio
    async def test_truncate_old_entries(self, wal):
        """Old entries are removed by truncation."""
        import time

        # Log some entries
        await wal.log_insert("old_point_1", {"v": [1.0]})
        await wal.log_insert("old_point_2", {"v": [2.0]})

        # Wait and record timestamp
        await asyncio.sleep(1.1)
        cutoff_time = int(datetime.now().timestamp())

        # Log more entries
        await wal.log_insert("new_point_1", {"v": [3.0]})

        # Truncate old entries
        removed = await wal.truncate(cutoff_time)
        assert removed == 2

        # Verify only new entries remain
        entries = await wal.get_entries_since(0)
        assert len(entries) == 1
        assert entries[0]["point_id"] == "new_point_1"

    @pytest.mark.asyncio
    async def test_truncate_no_matching_entries(self, wal):
        """Truncate with past timestamp before all entries removes nothing."""
        await wal.log_insert("point_001", {"v": [1.0]})

        # Use a time well in the past (before the entry was logged)
        past_time = int((datetime.now() - timedelta(days=1)).timestamp())
        removed = await wal.truncate(past_time)
        assert removed == 0


class TestWriteAheadLogRotation:
    """Tests for WAL rotation when size limit is reached."""

    @pytest.mark.asyncio
    async def test_wal_rotation_on_size_limit(self, temp_wal_path):
        """WAL rotates when size limit is reached."""
        # Create WAL with very small size limit
        wal = WriteAheadLog(
            wal_path=temp_wal_path,
            collection_name="test_collection",
            max_size_mb=0.001,  # 1KB limit
        )

        # Log enough entries to trigger rotation
        large_data = {"data": "x" * 500}
        for i in range(5):
            await wal.log_insert(f"point_{i}", large_data)

        # Check that rotation occurred (archive file created)
        wal_dir = temp_wal_path.parent
        archive_files = list(wal_dir.glob("*.bak"))
        assert len(archive_files) >= 1

        await wal.close()

    @pytest.mark.asyncio
    async def test_wal_rotation_clears_entries(self, temp_wal_path):
        """After rotation, WAL entries are cleared."""
        wal = WriteAheadLog(
            wal_path=temp_wal_path,
            collection_name="test_collection",
            max_size_mb=0.001,
        )

        # Trigger rotation
        large_data = {"data": "x" * 500}
        for i in range(5):
            await wal.log_insert(f"point_{i}", large_data)

        # Entries should be cleared after rotation
        entries = await wal.get_entries_since(0)
        # After rotation, WAL should be empty or have few entries
        assert len(entries) < 5

        await wal.close()


class TestWriteAheadLogConcurrency:
    """Tests for concurrent WAL operations."""

    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, wal):
        """Concurrent insert operations are handled correctly."""
        async def insert_point(i):
            return await wal.log_insert(
                point_id=f"concurrent_point_{i}",
                point_data={"index": i},
            )

        # Run 10 concurrent inserts
        tasks = [insert_point(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(results)
        entries = await wal.get_entries_since(0)
        assert len(entries) == 10

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, wal):
        """Concurrent mixed operations are handled correctly."""
        async def do_operation(op_type, i):
            if op_type == "insert":
                return await wal.log_insert(f"point_{i}", {"v": [i]})
            elif op_type == "update":
                return await wal.log_update(f"point_{i}", {"v": [i * 2]})
            else:
                return await wal.log_delete(f"point_{i}")

        # Mix of operations
        tasks = [
            do_operation("insert", 0),
            do_operation("insert", 1),
            do_operation("update", 0),
            do_operation("delete", 2),
            do_operation("insert", 3),
        ]
        results = await asyncio.gather(*tasks)

        assert all(results)
        entries = await wal.get_entries_since(0)
        assert len(entries) == 5


# =============================================================================
# SnapshotInfo Tests
# =============================================================================

class TestSnapshotInfo:
    """Tests for SnapshotInfo dataclass."""

    def test_snapshot_info_creation(self):
        """SnapshotInfo is created correctly."""
        info = SnapshotInfo(
            snapshot_id="test_snapshot_001",
            collection_name="test_collection",
            created_at=datetime.now(),
            status=SnapshotStatus.COMPLETED,
            size_bytes=1024,
            point_count=100,
            is_incremental=False,
            checksum="abc123",
            description="Test snapshot",
        )

        assert info.snapshot_id == "test_snapshot_001"
        assert info.status == SnapshotStatus.COMPLETED
        assert info.is_incremental is False

    def test_snapshot_info_to_dict(self):
        """SnapshotInfo serializes to dictionary correctly."""
        now = datetime.now()
        info = SnapshotInfo(
            snapshot_id="snap_001",
            collection_name="collection",
            created_at=now,
            status=SnapshotStatus.COMPLETED,
            size_bytes=2048,
            point_count=50,
            is_incremental=True,
            parent_snapshot_id="parent_snap",
        )

        data = info.to_dict()

        assert data["snapshot_id"] == "snap_001"
        assert data["status"] == "completed"
        assert data["is_incremental"] is True
        assert data["created_at"] == now.isoformat()

    def test_snapshot_info_from_dict(self):
        """SnapshotInfo deserializes from dictionary correctly."""
        now = datetime.now()
        data = {
            "snapshot_id": "snap_002",
            "collection_name": "collection",
            "created_at": now.isoformat(),
            "status": "pending",
            "size_bytes": 512,
            "point_count": 25,
            "is_incremental": False,
            "parent_snapshot_id": None,
            "checksum": "def456",
            "description": "Another snapshot",
            "metadata": {"key": "value"},
        }

        info = SnapshotInfo.from_dict(data)

        assert info.snapshot_id == "snap_002"
        assert info.status == SnapshotStatus.PENDING
        assert info.checksum == "def456"

    def test_snapshot_info_round_trip(self):
        """SnapshotInfo survives serialization round-trip."""
        original = SnapshotInfo(
            snapshot_id="snap_rt",
            collection_name="collection",
            created_at=datetime.now(),
            status=SnapshotStatus.IN_PROGRESS,
            size_bytes=4096,
            point_count=200,
            is_incremental=False,
            checksum="xyz789",
            description="Round trip test",
            metadata={"custom": "data"},
        )

        data = original.to_dict()
        restored = SnapshotInfo.from_dict(data)

        assert restored.snapshot_id == original.snapshot_id
        assert restored.status == original.status
        assert restored.metadata == original.metadata


# =============================================================================
# BackupManager Tests
# =============================================================================

class TestBackupManagerInit:
    """Tests for BackupManager initialization."""

    def test_backup_manager_creation(self, mock_qdrant_store, backup_config):
        """BackupManager initializes correctly."""
        manager = BackupManager(
            qdrant_store=mock_qdrant_store,
            config=backup_config,
        )

        assert manager.qdrant == mock_qdrant_store
        assert manager.config == backup_config
        assert manager.backup_dir.exists()

    def test_backup_manager_default_config(self, mock_qdrant_store, temp_backup_dir):
        """BackupManager uses default config when none provided."""
        manager = BackupManager(
            qdrant_store=mock_qdrant_store,
            config=BackupConfig(backup_dir=str(temp_backup_dir)),
        )

        assert manager.config.snapshot_interval_hours == 24
        assert manager.config.max_snapshots == 7
        assert manager.config.wal_enabled is True


class TestBackupManagerCreateSnapshot:
    """Tests for BackupManager.create_snapshot()."""

    @pytest.mark.asyncio
    async def test_create_full_snapshot(self, backup_manager, mock_qdrant_store, temp_backup_dir):
        """Full snapshot is created successfully."""
        # Setup mock to return some points
        mock_point = MagicMock()
        mock_point.id = "point_001"
        mock_point.vector = [0.1, 0.2, 0.3]
        mock_point.payload = {"content": "test"}

        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                ([mock_point], None),  # First call returns point
                ([], None),  # Second call returns empty (end of scroll)
            ]
        )

        snapshot = await backup_manager.create_snapshot(
            collection_name="test_collection",
            description="Test full snapshot",
            force_full=True,
        )

        assert snapshot is not None
        assert snapshot.status == SnapshotStatus.COMPLETED
        assert snapshot.is_incremental is False
        assert snapshot.description == "Test full snapshot"
        assert snapshot.checksum is not None

    @pytest.mark.asyncio
    async def test_create_incremental_snapshot(self, backup_manager, mock_qdrant_store):
        """Incremental snapshot is created when previous snapshot exists."""
        # Create a full snapshot first
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        full_snapshot = await backup_manager.create_snapshot(
            collection_name="test_collection",
            force_full=True,
        )

        # Create incremental snapshot
        incremental_snapshot = await backup_manager.create_snapshot(
            collection_name="test_collection",
            force_full=False,
        )

        assert incremental_snapshot.is_incremental is True
        assert incremental_snapshot.parent_snapshot_id == full_snapshot.snapshot_id

    @pytest.mark.asyncio
    async def test_create_snapshot_creates_directory(self, backup_manager, mock_qdrant_store):
        """Snapshot creates snapshot directory structure."""
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        snapshot = await backup_manager.create_snapshot(
            collection_name="test_collection",
        )

        snapshot_dir = backup_manager.backup_dir / "snapshots" / snapshot.snapshot_id
        assert snapshot_dir.exists()

    @pytest.mark.asyncio
    async def test_create_snapshot_exports_points(self, backup_manager, mock_qdrant_store):
        """Snapshot exports points to JSONL file."""
        mock_point1 = MagicMock()
        mock_point1.id = "point_001"
        mock_point1.vector = [0.1, 0.2]
        mock_point1.payload = {"content": "test1"}

        mock_point2 = MagicMock()
        mock_point2.id = "point_002"
        mock_point2.vector = [0.3, 0.4]
        mock_point2.payload = {"content": "test2"}

        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                ([mock_point1, mock_point2], None),
                ([], None),
            ]
        )

        snapshot = await backup_manager.create_snapshot(
            collection_name="test_collection",
        )

        # Check points file exists
        points_file = (
            backup_manager.backup_dir / "snapshots" / snapshot.snapshot_id / "points.jsonl"
        )
        assert points_file.exists()

        # Verify content
        with open(points_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_create_snapshot_calculates_checksum(self, backup_manager, mock_qdrant_store):
        """Snapshot calculates file checksum correctly."""
        mock_point = MagicMock()
        mock_point.id = "checksum_test"
        mock_point.vector = [1.0, 2.0, 3.0]
        mock_point.payload = {}

        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                ([mock_point], None),
                ([], None),
            ]
        )

        snapshot = await backup_manager.create_snapshot(
            collection_name="test_collection",
        )

        assert snapshot.checksum is not None
        assert len(snapshot.checksum) == 64  # SHA-256 hex digest length

    @pytest.mark.asyncio
    async def test_create_snapshot_failure_recovers(self, backup_manager, mock_qdrant_store):
        """Snapshot failure is handled gracefully."""
        mock_qdrant_store.get_collection_info = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with pytest.raises(Exception):
            await backup_manager.create_snapshot(
                collection_name="test_collection",
            )


class TestBackupManagerRestoreSnapshot:
    """Tests for BackupManager.restore_snapshot()."""

    @pytest.mark.asyncio
    async def test_restore_snapshot_happy_path(self, backup_manager, mock_qdrant_store, temp_backup_dir):
        """Snapshot restoration works correctly."""
        # Create a snapshot first
        mock_point = MagicMock()
        mock_point.id = "restore_test"
        mock_point.vector = [0.5, 0.6]
        mock_point.payload = {"data": "restore"}

        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                ([mock_point], None),
                ([], None),
            ]
        )

        snapshot = await backup_manager.create_snapshot(
            collection_name="test_collection",
        )

        # Reset mock for restore
        mock_qdrant_store.upsert = AsyncMock(return_value=None)

        # Restore the snapshot
        result = await backup_manager.restore_snapshot(
            snapshot_id=snapshot.snapshot_id,
            target_collection="restored_collection",
        )

        assert result is True
        mock_qdrant_store.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_restore_snapshot_to_original_collection(self, backup_manager, mock_qdrant_store):
        """Restore to original collection when target not specified."""
        mock_point = MagicMock()
        mock_point.id = "orig_test"
        mock_point.vector = [1.0]
        mock_point.payload = {}

        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                ([mock_point], None),
                ([], None),
                ([], None),  # For restore verification
            ]
        )

        snapshot = await backup_manager.create_snapshot(
            collection_name="original_collection",
        )

        result = await backup_manager.restore_snapshot(snapshot.snapshot_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_restore_snapshot_corrupt_file(self, backup_manager, mock_qdrant_store, temp_backup_dir):
        """Corrupt snapshot file is detected and rejected."""
        # Create a snapshot
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))
        snapshot = await backup_manager.create_snapshot("test_collection")

        # Corrupt the points file
        points_file = (
            backup_manager.backup_dir / "snapshots" / snapshot.snapshot_id / "points.jsonl"
        )
        with open(points_file, 'w') as f:
            f.write("corrupted data{{{")

        # Attempt restore should fail
        with pytest.raises(Exception):
            await backup_manager.restore_snapshot(snapshot.snapshot_id)

    @pytest.mark.asyncio
    async def test_restore_snapshot_checksum_mismatch(self, backup_manager, mock_qdrant_store):
        """Checksum mismatch is detected during restore."""
        mock_point = MagicMock()
        mock_point.id = "checksum_mismatch"
        mock_point.vector = [1.0]
        mock_point.payload = {}

        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                ([mock_point], None),
                ([], None),
            ]
        )

        snapshot = await backup_manager.create_snapshot("test_collection")

        # Modify the snapshot to change checksum
        snapshot.checksum = "wrong_checksum"
        await backup_manager._save_snapshot_info(snapshot)

        # Restore should fail due to checksum mismatch
        with pytest.raises(Exception):
            await backup_manager.restore_snapshot(snapshot.snapshot_id)

    @pytest.mark.asyncio
    async def test_restore_nonexistent_snapshot(self, backup_manager):
        """Restoring nonexistent snapshot raises error."""
        with pytest.raises(Exception):
            await backup_manager.restore_snapshot("nonexistent_snapshot_id")


class TestBackupManagerListSnapshots:
    """Tests for BackupManager.list_snapshots()."""

    @pytest.mark.asyncio
    async def test_list_snapshots_empty(self, backup_manager):
        """Empty list returned when no snapshots exist."""
        snapshots = await backup_manager.list_snapshots()
        assert snapshots == []

    @pytest.mark.asyncio
    async def test_list_snapshots_with_data(self, backup_manager, mock_qdrant_store):
        """Snapshots are listed correctly."""
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        await backup_manager.create_snapshot("collection_a")
        await backup_manager.create_snapshot("collection_b")
        await backup_manager.create_snapshot("collection_a")

        snapshots = await backup_manager.list_snapshots()
        assert len(snapshots) == 3

    @pytest.mark.asyncio
    async def test_list_snapshots_filter_by_collection(self, backup_manager, mock_qdrant_store):
        """Snapshots are filtered by collection name."""
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        await backup_manager.create_snapshot("collection_a")
        await backup_manager.create_snapshot("collection_b")
        await backup_manager.create_snapshot("collection_a")

        snapshots = await backup_manager.list_snapshots(collection_name="collection_a")
        assert len(snapshots) == 2
        assert all(s.collection_name == "collection_a" for s in snapshots)

    @pytest.mark.asyncio
    async def test_list_snapshots_filter_by_status(self, backup_manager, mock_qdrant_store):
        """Snapshots are filtered by status."""
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        await backup_manager.create_snapshot("collection_a")

        snapshots = await backup_manager.list_snapshots(status=SnapshotStatus.COMPLETED)
        assert all(s.status == SnapshotStatus.COMPLETED for s in snapshots)

    @pytest.mark.asyncio
    async def test_list_snapshots_limit(self, backup_manager, mock_qdrant_store):
        """Snapshot list respects limit parameter."""
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        for i in range(10):
            await backup_manager.create_snapshot(f"collection_{i}")

        snapshots = await backup_manager.list_snapshots(limit=5)
        assert len(snapshots) == 5


class TestBackupManagerGetSnapshotInfo:
    """Tests for BackupManager.get_snapshot_info()."""

    @pytest.mark.asyncio
    async def test_get_snapshot_info_exists(self, backup_manager, mock_qdrant_store):
        """Snapshot info is retrieved correctly."""
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        created = await backup_manager.create_snapshot("test_collection")
        info = await backup_manager.get_snapshot_info(created.snapshot_id)

        assert info is not None
        assert info.snapshot_id == created.snapshot_id
        assert info.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_get_snapshot_info_not_exists(self, backup_manager):
        """None returned for nonexistent snapshot."""
        info = await backup_manager.get_snapshot_info("nonexistent_id")
        assert info is None


class TestBackupManagerDeleteSnapshot:
    """Tests for BackupManager.delete_snapshot()."""

    @pytest.mark.asyncio
    async def test_delete_snapshot_removes_files(self, backup_manager, mock_qdrant_store):
        """Delete removes snapshot files from disk."""
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        snapshot = await backup_manager.create_snapshot("test_collection")
        snapshot_dir = backup_manager.backup_dir / "snapshots" / snapshot.snapshot_id
        assert snapshot_dir.exists()

        result = await backup_manager.delete_snapshot(snapshot.snapshot_id)
        assert result is True
        assert not snapshot_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_snapshot_removes_from_db(self, backup_manager, mock_qdrant_store):
        """Delete removes snapshot from database."""
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        snapshot = await backup_manager.create_snapshot("test_collection")
        await backup_manager.delete_snapshot(snapshot.snapshot_id)

        info = await backup_manager.get_snapshot_info(snapshot.snapshot_id)
        assert info is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_snapshot(self, backup_manager):
        """Deleting nonexistent snapshot returns False."""
        result = await backup_manager.delete_snapshot("nonexistent_id")
        assert result is False


class TestBackupManagerWALIntegration:
    """Tests for BackupManager WAL integration."""

    @pytest.mark.asyncio
    async def test_get_wal_creates_wal(self, backup_manager):
        """get_wal creates WAL instance for collection."""
        wal = backup_manager.get_wal("test_collection")

        assert wal is not None
        assert wal.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_log_operation_insert(self, backup_manager):
        """log_operation logs insert correctly."""
        await backup_manager.log_operation(
            collection_name="test_collection",
            operation="insert",
            point_id="point_001",
            point_data={"vector": [1.0]},
        )

        wal = backup_manager.get_wal("test_collection")
        entries = await wal.get_entries_since(0)
        assert len(entries) == 1
        assert entries[0]["operation"] == "insert"

    @pytest.mark.asyncio
    async def test_log_operation_update(self, backup_manager):
        """log_operation logs update correctly."""
        await backup_manager.log_operation(
            collection_name="test_collection",
            operation="update",
            point_id="point_001",
            point_data={"vector": [2.0]},
        )

        wal = backup_manager.get_wal("test_collection")
        entries = await wal.get_entries_since(0)
        assert entries[0]["operation"] == "update"

    @pytest.mark.asyncio
    async def test_log_operation_delete(self, backup_manager):
        """log_operation logs delete correctly."""
        await backup_manager.log_operation(
            collection_name="test_collection",
            operation="delete",
            point_id="point_001",
        )

        wal = backup_manager.get_wal("test_collection")
        entries = await wal.get_entries_since(0)
        assert entries[0]["operation"] == "delete"

    @pytest.mark.asyncio
    async def test_get_incremental_changes(self, backup_manager, mock_qdrant_store):
        """Incremental changes since snapshot are retrieved."""
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        # Create snapshot
        snapshot = await backup_manager.create_snapshot("test_collection")

        # Log some operations
        await backup_manager.log_operation("test_collection", "insert", "new_point", {"v": [1.0]})

        # Get incremental changes
        changes = await backup_manager.get_incremental_changes(
            collection_name="test_collection",
            since_snapshot_id=snapshot.snapshot_id,
        )

        assert len(changes) == 1
        assert changes[0]["operation"] == "insert"


class TestBackupManagerRetentionPolicy:
    """Tests for retention policy enforcement."""

    @pytest.mark.asyncio
    async def test_max_snapshots_enforced(self, backup_manager, mock_qdrant_store):
        """Max snapshots limit is enforced."""
        backup_manager.config.max_snapshots = 3
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        # Create 5 snapshots
        for i in range(5):
            await backup_manager.create_snapshot("test_collection")

        snapshots = await backup_manager.list_snapshots("test_collection")
        assert len(snapshots) <= 3


class TestBackupManagerUtilityMethods:
    """Tests for utility methods."""

    def test_generate_snapshot_id(self, backup_manager):
        """Snapshot ID is generated correctly."""
        snapshot_id = backup_manager._generate_snapshot_id("test_collection")

        assert snapshot_id.startswith("test_collection_")
        parts = snapshot_id.split("_")
        assert len(parts) >= 4  # collection_timestamp_random

    def test_calculate_file_checksum(self, backup_manager, temp_backup_dir):
        """File checksum is calculated correctly."""
        test_file = temp_backup_dir / "test_file.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        checksum = backup_manager._calculate_file_checksum(test_file)

        # Verify it's a valid SHA-256 hex digest
        assert len(checksum) == 64
        expected = hashlib.sha256(test_content.encode()).hexdigest()
        assert checksum == expected


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestCreateBackupManager:
    """Tests for create_backup_manager convenience function."""

    @pytest.mark.asyncio
    async def test_create_backup_manager_function(self, mock_qdrant_store, temp_backup_dir):
        """create_backup_manager creates configured manager."""
        manager = await create_backup_manager(
            qdrant_store=mock_qdrant_store,
            backup_dir=str(temp_backup_dir),
            max_snapshots=10,
        )

        assert manager.config.max_snapshots == 10
        assert manager.backup_dir == temp_backup_dir

        await manager.close()
