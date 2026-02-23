"""
Memory Importer for MnemoCore
==============================

Import memories from exported files into Qdrant with validation
and deduplication.

Features:
    - Import from JSON, JSONL, and Parquet formats
    - Schema validation before import
    - Automatic deduplication strategies
    - Batch processing for efficient imports
    - Progress tracking and error recovery
    - ID mapping for tracking imported records

Usage:
    ```python
    from mnemocore.storage import MemoryImporter, DeduplicationStrategy

    importer = MemoryImporter(qdrant_store)

    # Import with deduplication
    result = await importer.import_file(
        collection_name="haim_hot",
        input_path="./exports/memories.json",
        deduplication=DeduplicationStrategy.SKIP,
    )
    ```
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from loguru import logger

from mnemocore.core.exceptions import (
    ValidationError,
    wrap_storage_exception,
)
from mnemocore.utils.json_compat import dumps, loads


# =============================================================================
# Enums and Configuration
# =============================================================================


class ImportFormat(Enum):
    """Supported import formats."""
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    AUTO = "auto"  # Detect from file extension


class DeduplicationStrategy(Enum):
    """How to handle duplicate IDs during import."""
    SKIP = "skip"               # Skip duplicate records
    OVERWRITE = "overwrite"     # Replace existing records
    ERROR = "error"             # Fail on duplicate
    MERGE = "merge"             # Merge payloads (keep newer)


class ValidationLevel(Enum):
    """Level of validation to perform."""
    NONE = "none"               # No validation
    BASIC = "basic"             # Check required fields
    STRICT = "strict"           # Full validation including vectors


@dataclass
class ImportOptions:
    """Options for memory import."""
    # Deduplication
    deduplication: DeduplicationStrategy = DeduplicationStrategy.SKIP
    validation: ValidationLevel = ValidationLevel.STRICT

    # Processing
    batch_size: int = 100
    parallel_batches: int = 1
    skip_errors: bool = True
    max_errors: int = 100

    # Transformations
    transform_ids: Optional[Callable[[str], str]] = None
    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None

    # Post-import
    create_collection: bool = False
    collection_config: Optional[Dict[str, Any]] = None

    # Tracking
    track_imports: bool = True
    import_log_path: str = "./data/import_log.db"


@dataclass
class ImportResult:
    """Result of an import operation."""
    success: bool
    records_processed: int
    records_imported: int
    records_skipped: int
    records_failed: int
    duplicates_found: int
    duration_seconds: float
    error_message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.records_processed == 0:
            return 0.0
        return (self.records_imported / self.records_processed) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "records_processed": self.records_processed,
            "records_imported": self.records_imported,
            "records_skipped": self.records_skipped,
            "records_failed": self.records_failed,
            "duplicates_found": self.duplicates_found,
            "duration_seconds": self.duration_seconds,
            "success_rate": self.success_rate,
            "error_message": self.error_message,
            "errors": self.errors[:10],  # Limit error messages
            "metadata": self.metadata,
        }


@dataclass
class ImportProgress:
    """Progress information during import."""
    total_records: int
    processed_records: int
    imported_records: int
    skipped_records: int
    failed_records: int
    current_batch: int
    percent_complete: float


# =============================================================================
# Import Log Database
# =============================================================================


class ImportLog:
    """
    Tracks import history for deduplication and recovery.

    Maintains a record of all imported memory IDs across imports
    to enable duplicate detection and import resume capability.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize import log database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS imported_ids (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    import_timestamp TEXT NOT NULL,
                    import_source TEXT,
                    checksum TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection
                ON imported_ids(collection_name, import_timestamp)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS import_history (
                    import_id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    source_file TEXT,
                    records_processed INTEGER,
                    records_imported INTEGER,
                    records_skipped INTEGER,
                    records_failed INTEGER,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            """)

    def is_imported(
        self,
        memory_id: str,
        collection_name: str,
    ) -> bool:
        """Check if a memory ID was previously imported."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT 1 FROM imported_ids
                WHERE id = ? AND collection_name = ?
                LIMIT 1
                """,
                (memory_id, collection_name),
            )
            return cursor.fetchone() is not None

    def record_import(
        self,
        memory_id: str,
        collection_name: str,
        source: Optional[str] = None,
        checksum: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a successful import."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO imported_ids
                (id, collection_name, import_timestamp, import_source, checksum, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    collection_name,
                    datetime.now().isoformat(),
                    source,
                    checksum,
                    dumps(metadata or {}),
                ),
            )

    def start_import(
        self,
        import_id: str,
        collection_name: str,
        source_file: Optional[str] = None,
    ) -> bool:
        """Start tracking a new import operation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO import_history
                    (import_id, collection_name, started_at, source_file,
                     records_processed, records_imported, records_skipped,
                     records_failed, status)
                    VALUES (?, ?, ?, ?, 0, 0, 0, 0, 'in_progress')
                    """,
                    (import_id, collection_name, datetime.now().isoformat(), source_file),
                )
            return True
        except Exception as e:
            logger.warning(f"Failed to start import log: {e}")
            return False

    def complete_import(
        self,
        import_id: str,
        processed: int,
        imported: int,
        skipped: int,
        failed: int,
        error_message: Optional[str] = None,
    ):
        """Mark an import as complete."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE import_history
                    SET completed_at = ?,
                        records_processed = ?,
                        records_imported = ?,
                        records_skipped = ?,
                        records_failed = ?,
                        status = ?,
                        error_message = ?
                    WHERE import_id = ?
                    """,
                    (
                        datetime.now().isoformat(),
                        processed,
                        imported,
                        skipped,
                        failed,
                        'completed' if error_message is None else 'failed',
                        error_message,
                        import_id,
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to complete import log: {e}")

    def get_import_history(
        self,
        collection_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get import history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM import_history WHERE 1=1"
            params = []

            if collection_name:
                query += " AND collection_name = ?"
                params.append(collection_name)

            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# Memory Importer
# =============================================================================


class MemoryImporter:
    """
    Import memories from files into Qdrant.

    Supports multiple formats with validation and deduplication.
    """

    def __init__(
        self,
        qdrant_store: Any,  # QdrantStore
        default_options: Optional[ImportOptions] = None,
    ):
        """
        Initialize MemoryImporter.

        Args:
            qdrant_store: QdrantStore instance
            default_options: Default import options
        """
        self.qdrant = qdrant_store
        self.default_options = default_options or ImportOptions()
        self._import_logs: Dict[str, ImportLog] = {}

    def get_import_log(self, options: Optional[ImportOptions] = None) -> ImportLog:
        """Get or create import log."""
        opts = options or self.default_options
        log_path = opts.import_log_path

        if log_path not in self._import_logs:
            self._import_logs[log_path] = ImportLog(log_path)

        return self._import_logs[log_path]

    async def import_file(
        self,
        collection_name: str,
        input_path: Union[str, Path],
        options: Optional[ImportOptions] = None,
        progress_callback: Optional[Callable[[ImportProgress], None]] = None,
    ) -> ImportResult:
        """
        Import memories from a file.

        Args:
            collection_name: Target collection name
            input_path: Path to input file
            options: Import options
            progress_callback: Optional progress callback

        Returns:
            ImportResult with import statistics
        """
        opts = options or self.default_options
        input_file = Path(input_path)

        start_time = asyncio.get_event_loop().time()
        import_id = self._generate_import_id(collection_name)

        logger.info(f"Starting import {import_id} from {input_file}")

        # Initialize result
        result = ImportResult(
            success=False,
            records_processed=0,
            records_imported=0,
            records_skipped=0,
            records_failed=0,
            duplicates_found=0,
            duration_seconds=0,
            metadata={
                "collection_name": collection_name,
                "source_file": str(input_file),
                "import_id": import_id,
            },
        )

        # Start import log
        if opts.track_imports:
            log = self.get_import_log(opts)
            log.start_import(import_id, collection_name, str(input_file))

        try:
            # Ensure collection exists
            if opts.create_collection:
                await self._ensure_collection(collection_name, opts)

            # Detect format if AUTO
            format = opts.format if hasattr(opts, 'format') else ImportFormat.AUTO
            if format == ImportFormat.AUTO:
                format = self._detect_format(input_file)

            # Read records based on format
            if format == ImportFormat.JSON:
                records = await self._read_json(input_file, opts)
            elif format == ImportFormat.JSONL:
                records = await self._read_jsonl(input_file, opts)
            elif format == ImportFormat.PARQUET:
                records = await self._read_parquet(input_file, opts)
            else:
                raise ValidationError(
                    "format",
                    f"Unsupported import format: {format}",
                )

            result.records_processed = len(records)
            logger.info(f"Read {len(records)} records from {input_file}")

            # Import records
            imported, skipped, failed, duplicates = await self._import_records(
                collection_name,
                records,
                opts,
                progress_callback,
            )

            result.records_imported = imported
            result.records_skipped = skipped
            result.records_failed = failed
            result.duplicates_found = duplicates
            result.success = failed < opts.max_errors or opts.skip_errors

            duration = asyncio.get_event_loop().time() - start_time
            result.duration_seconds = duration

            logger.info(
                f"Import completed: {imported} imported, {skipped} skipped, "
                f"{failed} failed in {duration:.2f}s"
            )

            # Complete import log
            if opts.track_imports:
                log.complete_import(
                    import_id,
                    result.records_processed,
                    result.records_imported,
                    result.records_skipped,
                    result.records_failed,
                    result.error_message,
                )

            return result

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            result.duration_seconds = duration
            result.error_message = str(e)
            result.success = False

            logger.error(f"Import failed: {e}")

            # Complete import log with error
            if opts.track_imports:
                log = self.get_import_log(opts)
                log.complete_import(
                    import_id,
                    result.records_processed,
                    result.records_imported,
                    result.records_skipped,
                    result.records_failed,
                    str(e),
                )

            return result

    async def _import_records(
        self,
        collection_name: str,
        records: List[Dict[str, Any]],
        options: ImportOptions,
        progress_callback: Optional[Callable[[ImportProgress], None]],
    ) -> Tuple[int, int, int, int]:
        """
        Import a list of records into Qdrant.

        Returns:
            Tuple of (imported, skipped, failed, duplicates)
        """
        imported = 0
        skipped = 0
        failed = 0
        duplicates = 0

        # Track imported IDs for deduplication
        imported_ids: Set[str] = set()
        import_log = self.get_import_log(options) if options.track_imports else None

        # Process in batches
        batch = []
        batch_num = 0

        for record_data in records:
            # Apply filter if provided
            if options.filter_fn and not options.filter_fn(record_data):
                skipped += 1
                continue

            # Validate
            if options.validation != ValidationLevel.NONE:
                if not self._validate_record(record_data, options.validation):
                    logger.warning(f"Validation failed for record: {record_data.get('id')}")
                    failed += 1
                    if not options.skip_errors:
                        break
                    continue

            # Get ID
            record_id = str(record_data.get("id", ""))

            # Apply ID transformation
            if options.transform_ids:
                record_id = options.transform_ids(record_id)

            # Check deduplication
            is_duplicate = False

            if record_id in imported_ids:
                is_duplicate = True
            elif import_log and import_log.is_imported(record_id, collection_name):
                is_duplicate = True

            if is_duplicate:
                duplicates += 1
                if options.deduplication == DeduplicationStrategy.SKIP:
                    skipped += 1
                    continue
                elif options.deduplication == DeduplicationStrategy.ERROR:
                    failed += 1
                    continue

            # Deserialize to PointStruct
            try:
                point = self._deserialize_record(record_data, options)
            except Exception as e:
                logger.warning(f"Failed to deserialize record {record_id}: {e}")
                failed += 1
                if not options.skip_errors:
                    break
                continue

            batch.append(point)
            imported_ids.add(record_id)

            # Process batch
            if len(batch) >= options.batch_size:
                batch_imported, batch_failed = await self._upsert_batch(
                    collection_name, batch, options
                )
                imported += batch_imported
                failed += batch_failed
                batch = []
                batch_num += 1

                # Report progress
                if progress_callback:
                    progress = ImportProgress(
                        total_records=len(records),
                        processed_records=imported + skipped + failed,
                        imported_records=imported,
                        skipped_records=skipped,
                        failed_records=failed,
                        current_batch=batch_num,
                        percent_complete=(
                            (imported + skipped + failed) / len(records) * 100
                            if records else 0
                        ),
                    )
                    progress_callback(progress)

        # Process remaining batch
        if batch:
            batch_imported, batch_failed = await self._upsert_batch(
                collection_name, batch, options
            )
            imported += batch_imported
            failed += batch_failed

        # Record imported IDs
        if import_log:
            for record_id in imported_ids:
                import_log.record_import(
                    record_id,
                    collection_name,
                    source=options.import_log_path,
                )

        return imported, skipped, failed, duplicates

    async def _upsert_batch(
        self,
        collection_name: str,
        batch: List[Any],
        options: ImportOptions,
    ) -> Tuple[int, int]:
        """Upsert a batch of points to Qdrant."""
        try:
            await self.qdrant.upsert(collection_name, batch)
            return len(batch), 0
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
            if options.skip_errors:
                # Try individual upserts
                success = 0
                for point in batch:
                    try:
                        await self.qdrant.upsert(collection_name, [point])
                        success += 1
                    except Exception:
                        pass
                return success, len(batch) - success
            else:
                return 0, len(batch)

    async def _read_json(
        self,
        input_path: Path,
        options: ImportOptions,
    ) -> List[Dict[str, Any]]:
        """Read records from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = loads(f.read())

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "records" in data:
            return data["records"]
        else:
            raise ValidationError(
                "file_format",
                "JSON file must contain an array of records",
            )

    async def _read_jsonl(
        self,
        input_path: Path,
        options: ImportOptions,
    ) -> List[Dict[str, Any]]:
        """Read records from JSONL file."""
        records = []

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    records.append(loads(line))
                except Exception as e:
                    if not options.skip_errors:
                        raise ValidationError(
                            "file_format",
                            f"Invalid JSON line: {e}",
                        )
                    logger.warning(f"Skipping invalid JSON line: {e}")

        return records

    async def _read_parquet(
        self,
        input_path: Path,
        options: ImportOptions,
    ) -> List[Dict[str, Any]]:
        """Read records from Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ValidationError(
                "parquet",
                "PyArrow is required for Parquet import. "
                "Install with: pip install pyarrow",
            )

        table = pq.read_table(input_path)
        return self._arrow_table_to_records(table)

    def _arrow_table_to_records(self, table) -> List[Dict[str, Any]]:
        """Convert PyArrow table to list of record dicts."""
        records = []

        for i in range(table.num_rows):
            record = {}

            for col_name in table.column_names:
                col = table.column(col_name)
                val = col[i].as_py()

                if col_name == "payload" and isinstance(val, str):
                    # Deserialize JSON payload
                    try:
                        record[col_name] = loads(val)
                    except Exception:
                        record[col_name] = val
                elif col_name == "vector" and val is not None:
                    # Handle vector deserialization
                    if isinstance(val, dict):
                        # Compressed format
                        record[col_name] = self._decompress_vector(val)
                    else:
                        record[col_name] = val
                else:
                    record[col_name] = val

            records.append(record)

        return records

    def _decompress_vector(self, compressed: Dict[str, Any]) -> List[float]:
        """Decompress a vector from compressed format."""
        data = np.array(compressed["data"], dtype=np.uint8)
        shape = compressed["shape"]
        vmin = compressed["min"]
        vmax = compressed["max"]

        # Dequantize
        normalized = data.astype(np.float32) / 255.0
        vector = normalized * (vmax - vmin) + vmin

        return vector.tolist()

    def _deserialize_record(
        self,
        data: Dict[str, Any],
        options: ImportOptions,
    ) -> Any:
        """Deserialize a record dict to Qdrant PointStruct."""
        from qdrant_client import models

        record_id = str(data.get("id", ""))

        if options.transform_ids:
            record_id = options.transform_ids(record_id)

        # Handle vector
        vector = data.get("vector")
        if vector is None:
            raise ValidationError("vector", "Vector is required")

        # Handle payload
        payload = data.get("payload", {})

        return models.PointStruct(
            id=record_id,
            vector=vector,
            payload=payload,
        )

    def _validate_record(
        self,
        data: Dict[str, Any],
        level: ValidationLevel,
    ) -> bool:
        """Validate a record."""
        if level == ValidationLevel.NONE:
            return True

        # Basic validation
        if "id" not in data:
            return False

        if "vector" not in data:
            return False

        vector = data["vector"]
        if vector is None:
            return False

        # Strict validation
        if level == ValidationLevel.STRICT:
            # Check vector is valid
            if isinstance(vector, list):
                if not vector:
                    return False
                # Check all elements are numbers
                try:
                    [float(v) for v in vector]
                except (ValueError, TypeError):
                    return False
            elif isinstance(vector, dict) and "data" in vector:
                # Compressed format
                if not isinstance(vector["data"], list):
                    return False

        return True

    def _detect_format(self, file_path: Path) -> ImportFormat:
        """Detect format from file extension."""
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return ImportFormat.JSON
        elif suffix == ".jsonl":
            return ImportFormat.JSONL
        elif suffix == ".parquet" or suffix == ".pq":
            return ImportFormat.PARQUET
        else:
            raise ValidationError(
                "file_format",
                f"Unknown file format: {suffix}",
            )

    def _generate_import_id(self, collection_name: str) -> str:
        """Generate a unique import ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = os.urandom(4).hex() if 'os' in globals() else str(hash(timestamp))
        return f"{collection_name}_{timestamp}_{rand}"

    async def _ensure_collection(
        self,
        collection_name: str,
        options: ImportOptions,
    ):
        """Ensure the target collection exists."""
        try:
            exists = await self.qdrant.client.collection_exists(collection_name)
            if exists:
                return

            # Create collection with default config
            from qdrant_client import models

            config = options.collection_config or {}

            dimension = config.get("dimension", 16384)

            await self.qdrant.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=config.get("distance", models.Distance.DOT),
                    on_disk=config.get("on_disk", False),
                ),
            )

            logger.info(f"Created collection {collection_name}")

        except Exception as e:
            raise wrap_storage_exception("qdrant", "create_collection", e)


# =============================================================================
# Convenience Functions
# =============================================================================


async def import_memories(
    qdrant_store: Any,
    collection_name: str,
    input_path: Union[str, Path],
    deduplication: DeduplicationStrategy = DeduplicationStrategy.SKIP,
    **options_kwargs,
) -> ImportResult:
    """
    Convenience function to import memories.

    Args:
        qdrant_store: QdrantStore instance
        collection_name: Target collection name
        input_path: Input file path
        deduplication: Deduplication strategy
        **options_kwargs: Additional import options

    Returns:
        ImportResult with import statistics
    """
    options = ImportOptions(deduplication=deduplication, **options_kwargs)
    importer = MemoryImporter(qdrant_store, options)
    return await importer.import_file(collection_name, input_path, options)


import os
