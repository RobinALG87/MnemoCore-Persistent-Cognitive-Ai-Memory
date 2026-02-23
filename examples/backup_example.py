"""
Example: Backup & Snapshotting Usage
======================================

Demonstrates how to use the MnemoCore backup system for automated
Qdrant snapshots, WAL (Write-Ahead Log), and import/export.
"""

import asyncio
from pathlib import Path

from mnemocore.core.qdrant_store import QdrantStore
from mnemocore.storage import (
    BackupManager,
    BackupConfig,
    MemoryExporter,
    MemoryImporter,
    ExportFormat,
    ImportOptions,
    DeduplicationStrategy,
)


async def main():
    """Demonstrate backup functionality."""

    # Initialize Qdrant store
    qdrant = QdrantStore(
        url="http://localhost:6333",
        api_key=None,
        dimensionality=16384,
    )

    # Ensure collections exist
    await qdrant.ensure_collections()

    # -------------------------------------------------------------------------
    # 1. Backup Manager - Automated Snapshots
    # -------------------------------------------------------------------------

    # Configure backup
    backup_config = BackupConfig(
        auto_snapshot_enabled=True,
        snapshot_interval_hours=24,
        max_snapshots=7,
        wal_enabled=True,
        backup_dir="./backups",
        retention_days=30,
    )

    # Create backup manager
    backup_mgr = BackupManager(qdrant, config=backup_config)

    # Create a manual snapshot
    print("Creating snapshot...")
    snapshot = await backup_mgr.create_snapshot(
        collection_name="haim_hot",
        description="Manual backup example",
    )
    print(f"Snapshot created: {snapshot.snapshot_id}")
    print(f"  Points: {snapshot.point_count}")
    print(f"  Size: {snapshot.size_bytes} bytes")

    # List snapshots
    snapshots = await backup_mgr.list_snapshots("haim_hot")
    print(f"\nFound {len(snapshots)} snapshots")

    # Start automated snapshots
    await backup_mgr.start_scheduled_snapshots()

    # -------------------------------------------------------------------------
    # 2. Memory Exporter - Export to JSON/Parquet
    # -------------------------------------------------------------------------

    exporter = MemoryExporter(qdrant)

    # Export to JSON
    print("\nExporting to JSON...")
    json_result = await exporter.export(
        collection_name="haim_hot",
        output_path="./exports/hot_memories.json",
        format=ExportFormat.JSON,
    )
    print(f"Exported {json_result.records_exported} records to JSON")

    # Export to Parquet (requires pyarrow)
    try:
        print("\nExporting to Parquet...")
        parquet_result = await exporter.export(
            collection_name="haim_hot",
            output_path="./exports/hot_memories.parquet",
            format=ExportFormat.PARQUET,
        )
        print(f"Exported {parquet_result.records_exported} records to Parquet")
        print(f"  Compression ratio: {parquet_result.size_bytes / json_result.size_bytes:.2f}")
    except ImportError:
        print("  PyArrow not available, skipping Parquet export")

    # -------------------------------------------------------------------------
    # 3. Memory Importer - Import with Deduplication
    # -------------------------------------------------------------------------

    importer = MemoryImporter(qdrant)

    # Import with skip-on-duplicate strategy
    print("\nImporting with deduplication...")
    import_options = ImportOptions(
        deduplication=DeduplicationStrategy.SKIP,
        validation=ImportOptions.validation.STRICT,
        batch_size=100,
    )

    # Import from previously exported file
    import_result = await importer.import_file(
        collection_name="haim_warm",
        input_path="./exports/hot_memories.json",
        options=import_options,
    )

    print(f"Import result:")
    print(f"  Processed: {import_result.records_processed}")
    print(f"  Imported: {import_result.records_imported}")
    print(f"  Skipped: {import_result.records_skipped}")
    print(f"  Duplicates found: {import_result.duplicates_found}")

    # -------------------------------------------------------------------------
    # 4. Disaster Recovery
    # -------------------------------------------------------------------------

    # Restore from snapshot
    print("\nRestoring from snapshot...")
    success = await backup_mgr.restore_snapshot(
        snapshot_id=snapshot.snapshot_id,
        target_collection="haim_restored",
        verify_after=True,
    )
    print(f"Restore {'succeeded' if success else 'failed'}")

    # Cleanup
    await backup_mgr.close()
    await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
