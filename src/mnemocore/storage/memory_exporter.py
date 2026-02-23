"""
Memory Exporter for MnemoCore
==============================

Export memories from Qdrant to various formats (JSON, Parquet).

Features:
    - Export to JSON (single file or JSONL)
    - Export to Parquet for efficient storage and analysis
    - Filtering by collection, time range, or metadata
    - Batch processing for large exports
    - Progress tracking and error handling

Usage:
    ```python
    from mnemocore.storage import MemoryExporter, ExportFormat

    exporter = MemoryExporter(qdrant_store)

    # Export to JSON
    result = await exporter.export(
        collection_name="haim_hot",
        output_path="./exports/hot_memories.json",
        format=ExportFormat.JSON,
    )

    # Export to Parquet
    result = await exporter.export(
        collection_name="haim_warm",
        output_path="./exports/warm_memories.parquet",
        format=ExportFormat.PARQUET,
    )
    ```
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json

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


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"           # Single JSON array
    JSONL = "jsonl"         # JSON Lines (one JSON object per line)
    PARQUET = "parquet"     # Apache Parquet format


class VectorExportMode(Enum):
    """How to handle vectors in exports."""
    FULL = "full"               # Export complete vectors
    COMPRESSED = "compressed"   # Export compressed/quantized vectors
    NONE = "none"              # Skip vectors (metadata only)


@dataclass
class ExportOptions:
    """Options for memory export."""
    # Format settings
    format: ExportFormat = ExportFormat.JSON
    vector_mode: VectorExportMode = VectorExportMode.FULL
    compression: bool = True

    # Filtering
    time_range: Optional[Tuple[datetime, datetime]] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None

    # Processing
    batch_size: int = 1000
    include_payload: bool = True
    pretty_print: bool = False  # For JSON

    # Parquet-specific
    parquet_compression: str = "snappy"  # snappy, gzip, brotli
    parquet_row_group_size: int = 10000


@dataclass
class ExportResult:
    """Result of an export operation."""
    success: bool
    output_path: Path
    format: ExportFormat
    records_exported: int
    size_bytes: int
    duration_seconds: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output_path": str(self.output_path),
            "format": self.format.value,
            "records_exported": self.records_exported,
            "size_bytes": self.size_bytes,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class ExportProgress:
    """Progress information for an export operation."""
    total_records: int
    exported_records: int
    current_batch: int
    percent_complete: float
    eta_seconds: Optional[float] = None


# =============================================================================
# Memory Exporter
# =============================================================================


class MemoryExporter:
    """
    Export memories from Qdrant to various formats.

    Supports JSON, JSONL, and Parquet formats with flexible filtering
    and vector handling options.
    """

    def __init__(
        self,
        qdrant_store: Any,  # QdrantStore
        default_options: Optional[ExportOptions] = None,
    ):
        """
        Initialize MemoryExporter.

        Args:
            qdrant_store: QdrantStore instance
            default_options: Default export options
        """
        self.qdrant = qdrant_store
        self.default_options = default_options or ExportOptions()

    async def export(
        self,
        collection_name: str,
        output_path: Union[str, Path],
        options: Optional[ExportOptions] = None,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None,
    ) -> ExportResult:
        """
        Export memories from a collection.

        Args:
            collection_name: Name of the collection to export
            output_path: Output file path
            options: Export options (uses defaults if None)
            progress_callback: Optional callback for progress updates

        Returns:
            ExportResult with export statistics
        """
        opts = options or self.default_options
        start_time = asyncio.get_event_loop().time()
        output = Path(output_path)

        logger.info(
            f"Starting export of {collection_name} to {output} "
            f"(format: {opts.format.value})"
        )

        try:
            # Ensure output directory exists
            output.parent.mkdir(parents=True, exist_ok=True)

            # Get collection info for total count
            collection_info = await self.qdrant.get_collection_info(collection_name)
            total_records = collection_info.points_count or 0

            # Export based on format
            if opts.format == ExportFormat.JSON:
                records, size = await self._export_json(
                    collection_name, output, opts, progress_callback, total_records
                )
            elif opts.format == ExportFormat.JSONL:
                records, size = await self._export_jsonl(
                    collection_name, output, opts, progress_callback, total_records
                )
            elif opts.format == ExportFormat.PARQUET:
                records, size = await self._export_parquet(
                    collection_name, output, opts, progress_callback, total_records
                )
            else:
                raise ValidationError(
                    "format",
                    f"Unsupported export format: {opts.format}",
                )

            duration = asyncio.get_event_loop().time() - start_time

            result = ExportResult(
                success=True,
                output_path=output,
                format=opts.format,
                records_exported=records,
                size_bytes=size,
                duration_seconds=duration,
                metadata={
                    "collection_name": collection_name,
                    "total_estimated": total_records,
                    "vector_mode": opts.vector_mode.value,
                },
            )

            logger.info(
                f"Export completed: {records} records, {size} bytes "
                f"in {duration:.2f}s"
            )

            return result

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(f"Export failed: {e}")

            return ExportResult(
                success=False,
                output_path=output,
                format=opts.format,
                records_exported=0,
                size_bytes=0,
                duration_seconds=duration,
                error_message=str(e),
            )

    async def _export_json(
        self,
        collection_name: str,
        output_path: Path,
        options: ExportOptions,
        progress_callback: Optional[Callable[[ExportProgress], None]],
        total_records: int,
    ) -> Tuple[int, int]:
        """Export to JSON array format."""
        records = []
        size_bytes = 0
        batch_num = 0

        offset = None
        while True:
            batch_records, offset = await self.qdrant.scroll(
                collection_name=collection_name,
                limit=options.batch_size,
                offset=offset,
                with_vectors=options.vector_mode != VectorExportMode.NONE,
            )

            if not batch_records:
                break

            for record in batch_records:
                data = self._serialize_record(record, options)
                records.append(data)

            # Report progress
            batch_num += 1
            if progress_callback:
                progress = ExportProgress(
                    total_records=total_records,
                    exported_records=len(records),
                    current_batch=batch_num,
                    percent_complete=(len(records) / total_records * 100) if total_records > 0 else 0,
                )
                progress_callback(progress)

            # Check limit
            if options.limit and len(records) >= options.limit:
                records = records[:options.limit]
                break

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json_str = dumps(
                records,
                indent=2 if options.pretty_print else None,
            )
            f.write(json_str)
            size_bytes = len(json_str.encode('utf-8'))

        return len(records), size_bytes

    async def _export_jsonl(
        self,
        collection_name: str,
        output_path: Path,
        options: ExportOptions,
        progress_callback: Optional[Callable[[ExportProgress], None]],
        total_records: int,
    ) -> Tuple[int, int]:
        """Export to JSON Lines format."""
        total_exported = 0
        batch_num = 0
        size_bytes = 0

        offset = None

        with open(output_path, 'w', encoding='utf-8') as f:
            while True:
                batch_records, offset = await self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=options.batch_size,
                    offset=offset,
                    with_vectors=options.vector_mode != VectorExportMode.NONE,
                )

                if not batch_records:
                    break

                for record in batch_records:
                    # Check limit before writing
                    if options.limit and total_exported >= options.limit:
                        break

                    data = self._serialize_record(record, options)
                    line = dumps(data) + '\n'
                    f.write(line)
                    size_bytes += len(line.encode('utf-8'))
                    total_exported += 1

                # Report progress
                batch_num += 1
                if progress_callback:
                    progress = ExportProgress(
                        total_records=total_records,
                        exported_records=total_exported,
                        current_batch=batch_num,
                        percent_complete=(total_exported / total_records * 100) if total_records > 0 else 0,
                    )
                    progress_callback(progress)

                # Check if limit reached
                if options.limit and total_exported >= options.limit:
                    break

        return total_exported, size_bytes

    async def _export_parquet(
        self,
        collection_name: str,
        output_path: Path,
        options: ExportOptions,
        progress_callback: Optional[Callable[[ExportProgress], None]],
        total_records: int,
    ) -> Tuple[int, int]:
        """Export to Parquet format."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ValidationError(
                "parquet",
                "PyArrow is required for Parquet export. "
                "Install with: pip install pyarrow",
            )

        # Collect all records
        all_records = []
        batch_num = 0
        offset = None

        while True:
            batch_records, offset = await self.qdrant.scroll(
                collection_name=collection_name,
                limit=options.batch_size,
                offset=offset,
                with_vectors=options.vector_mode != VectorExportMode.NONE,
            )

            if not batch_records:
                break

            for record in batch_records:
                data = self._serialize_record(record, options)
                all_records.append(data)

            # Report progress
            batch_num += 1
            if progress_callback:
                progress = ExportProgress(
                    total_records=total_records,
                    exported_records=len(all_records),
                    current_batch=batch_num,
                    percent_complete=(len(all_records) / total_records * 100) if total_records > 0 else 0,
                )
                progress_callback(progress)

            # Check limit
            if options.limit and len(all_records) >= options.limit:
                all_records = all_records[:options.limit]
                break

        # Convert to Arrow table
        table = self._records_to_arrow_table(all_records, options)

        # Write to Parquet
        pq.write_table(
            table,
            output_path,
            compression=options.parquet_compression,
            row_group_size=options.parquet_row_group_size,
        )

        size_bytes = output_path.stat().st_size
        return len(all_records), size_bytes

    def _serialize_record(
        self,
        record: Any,
        options: ExportOptions,
    ) -> Dict[str, Any]:
        """
        Serialize a Qdrant record to a dictionary.

        Handles vectors, payloads, and other record attributes.
        """
        result = {
            "id": str(record.id),
        }

        # Handle vector
        if options.vector_mode != VectorExportMode.NONE:
            if hasattr(record, 'vector') and record.vector is not None:
                if isinstance(record.vector, dict):
                    # Named vectors
                    if options.vector_mode == VectorExportMode.COMPRESSED:
                        # Compress vectors (e.g., quantize to uint8)
                        result["vector"] = {
                            k: self._compress_vector(v)
                            for k, v in record.vector.items()
                        }
                    else:
                        result["vector"] = {
                            k: v.tolist() if hasattr(v, 'tolist') else list(v)
                            for k, v in record.vector.items()
                        }
                else:
                    # Single vector
                    if options.vector_mode == VectorExportMode.COMPRESSED:
                        result["vector"] = self._compress_vector(record.vector)
                    else:
                        result["vector"] = (
                            record.vector.tolist()
                            if hasattr(record.vector, 'tolist')
                            else list(record.vector)
                        )

        # Handle payload
        if options.include_payload and hasattr(record, 'payload'):
            result["payload"] = record.payload or {}

        # Handle other attributes
        if hasattr(record, 'shard_key') and record.shard_key is not None:
            result["shard_key"] = record.shard_key

        return result

    def _compress_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Compress a vector for more efficient storage.

        Uses quantization to reduce size while maintaining
        reasonable precision for approximate operations.
        """
        # Normalize to [0, 1] range
        vec = np.asarray(vector)
        vmin, vmax = vec.min(), vec.max()
        if vmax > vmin:
            normalized = (vec - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(vec)

        # Quantize to uint8
        quantized = (normalized * 255).astype(np.uint8)

        return {
            "data": quantized.tolist(),
            "shape": vec.shape,
            "min": float(vmin),
            "max": float(vmax),
            "dtype": str(vec.dtype),
        }

    def _records_to_arrow_table(
        self,
        records: List[Dict[str, Any]],
        options: ExportOptions,
    ):
        """Convert records to a PyArrow Table for Parquet export."""
        try:
            import pyarrow as pa
        except ImportError:
            raise ValidationError(
                "parquet",
                "PyArrow is required for Parquet export",
            )

        # Build schema dynamically from records
        fields = []
        sample = records[0] if records else {}

        # ID field (always present)
        fields.append(pa.field("id", pa.string()))

        # Vector field (if included)
        if "vector" in sample and options.vector_mode != VectorExportMode.NONE:
            if isinstance(sample["vector"], dict):
                # Check if it's compressed format
                if "data" in sample.get("vector", {}):
                    fields.append(pa.field("vector", pa.struct([
                        pa.field("data", pa.list_(pa.uint8())),
                        pa.field("shape", pa.list_(pa.int64())),
                        pa.field("min", pa.float64()),
                        pa.field("max", pa.float64()),
                        pa.field("dtype", pa.string()),
                    ])))
                else:
                    # Named vectors - use map type
                    fields.append(pa.field("vector", pa.map_(pa.string(), pa.list_(pa.float64()))))
            else:
                # Single vector
                if options.vector_mode == VectorExportMode.COMPRESSED:
                    fields.append(pa.field("vector", pa.struct([
                        pa.field("data", pa.list_(pa.uint8())),
                        pa.field("shape", pa.list_(pa.int64())),
                        pa.field("min", pa.float64()),
                        pa.field("max", pa.float64()),
                        pa.field("dtype", pa.string()),
                    ])))
                else:
                    fields.append(pa.field("vector", pa.list_(pa.float64())))

        # Payload field (if included)
        if options.include_payload and "payload" in sample:
            # Use a string-serialized JSON for flexible schema
            fields.append(pa.field("payload", pa.string()))

        # Shard key (if present)
        if "shard_key" in sample:
            fields.append(pa.field("shard_key", pa.string()))

        # Build arrays
        arrays = []
        for field_name in [f.name for f in fields]:
            if field_name == "id":
                arrays.append(pa.array([r["id"] for r in records], pa.string()))
            elif field_name == "vector":
                vec_data = [r.get("vector") for r in records]
                arrays.append(self._vector_array_to_arrow(vec_data, options))
            elif field_name == "payload":
                payload_data = [dumps(r.get("payload", {})) for r in records]
                arrays.append(pa.array(payload_data, pa.string()))
            elif field_name == "shard_key":
                shard_data = [r.get("shard_key") for r in records]
                arrays.append(pa.array(shard_data, pa.string()))

        schema = pa.schema(fields)
        return pa.Table.from_arrays(arrays, schema=schema)

    def _vector_array_to_arrow(
        self,
        vectors: List[Any],
        options: ExportOptions,
    ):
        """Convert vector data to PyArrow array."""
        try:
            import pyarrow as pa
        except ImportError:
            raise ValidationError(
                "parquet",
                "PyArrow is required for Parquet export",
            )

        if not vectors:
            return pa.array([], pa.null())

        sample = vectors[0]

        # Handle None values
        vectors_clean = [v if v is not None else None for v in vectors]

        if isinstance(sample, dict) and "data" in sample:
            # Compressed format
            struct_arrays = []
            for v in vectors_clean:
                if v is None:
                    struct_arrays.append(None)
                else:
                    struct_arrays.append({
                        'data': v['data'],
                        'shape': v['shape'],
                        'min': v['min'],
                        'max': v['max'],
                        'dtype': v['dtype'],
                    })
            return pa.array(struct_arrays, type=pa.struct([
                pa.field("data", pa.list_(pa.uint8())),
                pa.field("shape", pa.list_(pa.int64())),
                pa.field("min", pa.float64()),
                pa.field("max", pa.float64()),
                pa.field("dtype", pa.string()),
            ]))
        elif isinstance(sample, dict):
            # Named vectors - serialize as JSON
            json_data = [dumps(v) if v is not None else None for v in vectors_clean]
            return pa.array(json_data, pa.string())
        else:
            # Simple vector list
            list_data = [
                list(v) if v is not None else None
                for v in vectors_clean
            ]
            return pa.array(list_data, pa.list_(pa.float64()))

    async def export_batch(
        self,
        collection_name: str,
        output_dir: Union[str, Path],
        options: Optional[ExportOptions] = None,
        max_file_records: int = 100000,
    ) -> List[ExportResult]:
        """
        Export a large collection in batches across multiple files.

        Args:
            collection_name: Collection to export
            output_dir: Output directory for batch files
            options: Export options
            max_file_records: Maximum records per file

        Returns:
            List of ExportResult objects, one per file
        """
        opts = options or self.default_options
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        file_num = 0
        total_exported = 0

        offset = None

        while True:
            # Fetch batch
            batch_records, offset = await self.qdrant.scroll(
                collection_name=collection_name,
                limit=opts.batch_size,
                offset=offset,
                with_vectors=opts.vector_mode != VectorExportMode.NONE,
            )

            if not batch_records:
                break

            # Process in file-sized chunks
            file_records = []
            file_num += 1

            for record in batch_records:
                file_records.append(record)

                if len(file_records) >= max_file_records:
                    break

            # Write file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{collection_name}_{timestamp}_{file_num:04d}.{opts.format.value}"
            file_path = output_dir / file_name

            result = await self._write_batch_file(
                file_records, file_path, opts
            )
            results.append(result)
            total_exported += result.records_exported

            logger.info(
                f"Exported batch file {file_num}: "
                f"{result.records_exported} records to {file_path}"
            )

            # Stop if limit reached
            if opts.limit and total_exported >= opts.limit:
                break

        logger.info(
            f"Batch export complete: {total_exported} records "
            f"across {len(results)} files"
        )

        return results

    async def _write_batch_file(
        self,
        records: List[Any],
        output_path: Path,
        options: ExportOptions,
    ) -> ExportResult:
        """Write a batch of records to a file."""
        start_time = asyncio.get_event_loop().time()

        if options.format == ExportFormat.JSON:
            with open(output_path, 'w', encoding='utf-8') as f:
                data = [self._serialize_record(r, options) for r in records]
                f.write(dumps(data, indent=2 if options.pretty_print else None))
            size_bytes = output_path.stat().st_size

        elif options.format == ExportFormat.JSONL:
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in records:
                    data = self._serialize_record(record, options)
                    f.write(dumps(data) + '\n')
            size_bytes = output_path.stat().st_size

        elif options.format == ExportFormat.PARQUET:
            serialized = [self._serialize_record(r, options) for r in records]
            table = self._records_to_arrow_table(serialized, options)

            import pyarrow.parquet as pq
            pq.write_table(
                table,
                output_path,
                compression=options.parquet_compression,
                row_group_size=options.parquet_row_group_size,
            )
            size_bytes = output_path.stat().st_size

        else:
            raise ValidationError(
                "format",
                f"Unsupported format: {options.format}",
            )

        duration = asyncio.get_event_loop().time() - start_time

        return ExportResult(
            success=True,
            output_path=output_path,
            format=options.format,
            records_exported=len(records),
            size_bytes=size_bytes,
            duration_seconds=duration,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def export_memories(
    qdrant_store: Any,
    collection_name: str,
    output_path: Union[str, Path],
    format: ExportFormat = ExportFormat.JSON,
    **options_kwargs,
) -> ExportResult:
    """
    Convenience function to export memories.

    Args:
        qdrant_store: QdrantStore instance
        collection_name: Collection to export
        output_path: Output file path
        format: Export format
        **options_kwargs: Additional export options

    Returns:
        ExportResult with export statistics
    """
    options = ExportOptions(format=format, **options_kwargs)
    exporter = MemoryExporter(qdrant_store, options)
    return await exporter.export(collection_name, output_path, options)
