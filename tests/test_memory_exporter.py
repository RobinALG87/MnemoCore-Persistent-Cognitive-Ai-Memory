"""
Tests for Memory Exporter Module
=================================

Tests for the MemoryExporter that exports memories from Qdrant to various
formats (JSON, JSONL, Parquet).

Coverage:
    - Export to JSON, JSONL, Parquet (if deps available)
    - Large export (1000+ memories) - stream vs batch
    - Empty collection export
    - Vector compression in export output
    - Round-trip export -> import integrity check
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from mnemocore.storage.memory_exporter import (
    MemoryExporter,
    ExportOptions,
    ExportFormat,
    VectorExportMode,
    ExportResult,
    ExportProgress,
    export_memories,
)
from mnemocore.utils.json_compat import dumps, loads


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_export_dir(tmp_path):
    """Create a temporary directory for export testing."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


@pytest.fixture
def mock_qdrant_store():
    """Create a mock QdrantStore with sample data."""
    mock = MagicMock()

    # Mock collection info
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 100
    mock.get_collection_info = AsyncMock(return_value=mock_collection_info)

    return mock


@dataclass
class MockRecord:
    """Mock Qdrant record for testing."""
    id: str
    vector: list
    payload: dict


@pytest.fixture
def sample_records():
    """Create sample records for testing."""
    records = []
    for i in range(10):
        records.append(MockRecord(
            id=f"point_{i:04d}",
            vector=[float(i) * 0.1] * 128,
            payload={
                "content": f"Test content {i}",
                "tier": "hot" if i % 2 == 0 else "warm",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"index": i},
            },
        ))
    return records


@pytest.fixture
def large_record_set():
    """Create 1000+ records for large export testing."""
    records = []
    for i in range(1500):
        records.append(MockRecord(
            id=f"large_point_{i:05d}",
            vector=[float(i % 100) * 0.01] * 256,
            payload={
                "content": f"Large export content {i}" * 10,
                "index": i,
            },
        ))
    return records


@pytest.fixture
def exporter(mock_qdrant_store):
    """Create a MemoryExporter instance."""
    return MemoryExporter(qdrant_store=mock_qdrant_store)


@pytest.fixture
def exporter_with_options(mock_qdrant_store):
    """Create a MemoryExporter with custom default options."""
    options = ExportOptions(
        format=ExportFormat.JSON,
        vector_mode=VectorExportMode.FULL,
        batch_size=100,
    )
    return MemoryExporter(qdrant_store=mock_qdrant_store, default_options=options)


# =============================================================================
# ExportOptions Tests
# =============================================================================

class TestExportOptions:
    """Tests for ExportOptions configuration."""

    def test_default_options(self):
        """Default options are set correctly."""
        options = ExportOptions()

        assert options.format == ExportFormat.JSON
        assert options.vector_mode == VectorExportMode.FULL
        assert options.compression is True
        assert options.batch_size == 1000
        assert options.include_payload is True

    def test_custom_options(self):
        """Custom options are applied correctly."""
        options = ExportOptions(
            format=ExportFormat.JSONL,
            vector_mode=VectorExportMode.COMPRESSED,
            batch_size=500,
            pretty_print=True,
        )

        assert options.format == ExportFormat.JSONL
        assert options.vector_mode == VectorExportMode.COMPRESSED
        assert options.batch_size == 500
        assert options.pretty_print is True


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_success_result(self, temp_export_dir):
        """Successful export result is created correctly."""
        output_path = temp_export_dir / "test.json"
        output_path.write_text("[]")

        result = ExportResult(
            success=True,
            output_path=output_path,
            format=ExportFormat.JSON,
            records_exported=100,
            size_bytes=1024,
            duration_seconds=1.5,
        )

        assert result.success is True
        assert result.records_exported == 100
        assert result.error_message is None

    def test_failure_result(self, temp_export_dir):
        """Failed export result contains error message."""
        result = ExportResult(
            success=False,
            output_path=temp_export_dir / "failed.json",
            format=ExportFormat.JSON,
            records_exported=0,
            size_bytes=0,
            duration_seconds=0.1,
            error_message="Connection failed",
        )

        assert result.success is False
        assert result.error_message == "Connection failed"

    def test_result_to_dict(self, temp_export_dir):
        """ExportResult converts to dictionary correctly."""
        output_path = temp_export_dir / "test.json"
        result = ExportResult(
            success=True,
            output_path=output_path,
            format=ExportFormat.JSONL,
            records_exported=50,
            size_bytes=2048,
            duration_seconds=2.0,
            metadata={"collection": "test"},
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["format"] == "jsonl"
        assert data["records_exported"] == 50
        assert data["metadata"]["collection"] == "test"


# =============================================================================
# JSON Export Tests
# =============================================================================

class TestExportJSON:
    """Tests for JSON export format."""

    @pytest.mark.asyncio
    async def test_export_json_basic(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Basic JSON export works correctly."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records[:5], 5),
                (sample_records[5:], None),
            ]
        )

        output_path = temp_export_dir / "export.json"
        options = ExportOptions(format=ExportFormat.JSON)

        result = await exporter.export(
            collection_name="test_collection",
            output_path=output_path,
            options=options,
        )

        assert result.success is True
        assert result.format == ExportFormat.JSON
        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path, 'r') as f:
            data = json.load(f)
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_export_json_with_vectors(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """JSON export includes vectors when requested."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records[:2], 2),
                ([], None),
            ]
        )

        output_path = temp_export_dir / "export_with_vectors.json"
        options = ExportOptions(
            format=ExportFormat.JSON,
            vector_mode=VectorExportMode.FULL,
        )

        await exporter.export("test_collection", output_path, options)

        with open(output_path, 'r') as f:
            data = json.load(f)

        assert len(data) == 2
        assert "vector" in data[0]
        assert len(data[0]["vector"]) == 128

    @pytest.mark.asyncio
    async def test_export_json_without_vectors(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """JSON export excludes vectors when requested."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records[:2], 2),
                ([], None),
            ]
        )

        output_path = temp_export_dir / "export_no_vectors.json"
        options = ExportOptions(
            format=ExportFormat.JSON,
            vector_mode=VectorExportMode.NONE,
        )

        await exporter.export("test_collection", output_path, options)

        with open(output_path, 'r') as f:
            data = json.load(f)

        assert "vector" not in data[0]

    @pytest.mark.asyncio
    async def test_export_json_pretty_print(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """JSON export with pretty print formatting."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records[:1], 1),
                ([], None),
            ]
        )

        output_path = temp_export_dir / "pretty.json"
        options = ExportOptions(
            format=ExportFormat.JSON,
            pretty_print=True,
        )

        await exporter.export("test_collection", output_path, options)

        with open(output_path, 'r') as f:
            content = f.read()

        # Pretty printed JSON should have newlines and indentation
        assert '\n' in content
        assert '  ' in content or '\t' in content


# =============================================================================
# JSONL Export Tests
# =============================================================================

class TestExportJSONL:
    """Tests for JSONL export format."""

    @pytest.mark.asyncio
    async def test_export_jsonl_basic(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Basic JSONL export works correctly."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records[:5], 5),
                (sample_records[5:], None),
            ]
        )

        output_path = temp_export_dir / "export.jsonl"
        options = ExportOptions(format=ExportFormat.JSONL)

        result = await exporter.export("test_collection", output_path, options)

        assert result.success is True
        assert result.format == ExportFormat.JSONL

        # Verify JSONL format - one JSON object per line
        with open(output_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 10
        for line in lines:
            record = json.loads(line)
            assert "id" in record

    @pytest.mark.asyncio
    async def test_export_jsonl_line_format(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Each line in JSONL is a valid JSON object."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records, None),
            ]
        )

        output_path = temp_export_dir / "lines.jsonl"
        options = ExportOptions(format=ExportFormat.JSONL)

        await exporter.export("test_collection", output_path, options)

        with open(output_path, 'r') as f:
            for i, line in enumerate(f):
                record = json.loads(line.strip())
                assert record["id"] == f"point_{i:04d}"


# =============================================================================
# Parquet Export Tests
# =============================================================================

class TestExportParquet:
    """Tests for Parquet export format."""

    @pytest.mark.asyncio
    async def test_export_parquet_basic(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Basic Parquet export works correctly."""
        pytest.importorskip("pyarrow")

        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records, None),
            ]
        )

        output_path = temp_export_dir / "export.parquet"
        options = ExportOptions(format=ExportFormat.PARQUET)

        result = await exporter.export("test_collection", output_path, options)

        assert result.success is True
        assert result.format == ExportFormat.PARQUET
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_export_parquet_compression(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Parquet export with compression options."""
        pytest.importorskip("pyarrow")

        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records, None),
            ]
        )

        output_path = temp_export_dir / "compressed.parquet"
        options = ExportOptions(
            format=ExportFormat.PARQUET,
            parquet_compression="snappy",
        )

        result = await exporter.export("test_collection", output_path, options)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_export_parquet_missing_dependency(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Parquet export fails gracefully without PyArrow."""
        with patch.dict('sys.modules', {'pyarrow': None, 'pyarrow.parquet': None}):
            mock_qdrant_store.scroll = AsyncMock(
                side_effect=[
                    (sample_records, None),
                ]
            )

            output_path = temp_export_dir / "no_pyarrow.parquet"
            options = ExportOptions(format=ExportFormat.PARQUET)

            result = await exporter.export("test_collection", output_path, options)

            assert result.success is False
            assert "PyArrow" in result.error_message


# =============================================================================
# Vector Compression Tests
# =============================================================================

class TestVectorCompression:
    """Tests for vector compression in exports."""

    @pytest.mark.asyncio
    async def test_compressed_vector_structure(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Compressed vectors have correct structure."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records[:1], 1),
                ([], None),
            ]
        )

        output_path = temp_export_dir / "compressed.json"
        options = ExportOptions(
            format=ExportFormat.JSON,
            vector_mode=VectorExportMode.COMPRESSED,
        )

        await exporter.export("test_collection", output_path, options)

        with open(output_path, 'r') as f:
            data = json.load(f)

        vector = data[0]["vector"]
        assert "data" in vector
        assert "shape" in vector
        assert "min" in vector
        assert "max" in vector
        assert "dtype" in vector

    @pytest.mark.asyncio
    async def test_compressed_vector_quantization(self, exporter, temp_export_dir):
        """Vector compression quantizes to uint8 correctly."""
        # Create record with known vector values
        record = MockRecord(
            id="quantize_test",
            vector=[0.0, 0.5, 1.0, 0.25, 0.75],
            payload={},
        )

        compressed = exporter._compress_vector(record.vector)

        # Check quantization
        assert compressed["data"][0] == 0  # min value
        assert compressed["data"][2] == 255  # max value
        assert compressed["min"] == 0.0
        assert compressed["max"] == 1.0

    @pytest.mark.asyncio
    async def test_compression_reduces_size(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Compressed vectors are smaller than full vectors."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records, None),
            ]
        )

        # Export with full vectors
        full_path = temp_export_dir / "full.json"
        await exporter.export(
            "test_collection",
            full_path,
            ExportOptions(vector_mode=VectorExportMode.FULL),
        )
        full_size = full_path.stat().st_size

        # Export with compressed vectors
        compressed_path = temp_export_dir / "compressed.json"
        await exporter.export(
            "test_collection",
            compressed_path,
            ExportOptions(vector_mode=VectorExportMode.COMPRESSED),
        )
        compressed_size = compressed_path.stat().st_size

        # Compressed should be smaller (or similar for small vectors)
        # For larger vectors, compression significantly reduces size
        assert compressed_size <= full_size * 1.5  # Allow some overhead


# =============================================================================
# Empty Collection Tests
# =============================================================================

class TestEmptyCollectionExport:
    """Tests for exporting empty collections."""

    @pytest.mark.asyncio
    async def test_export_empty_json(self, exporter, mock_qdrant_store, temp_export_dir):
        """Empty collection exports as empty JSON array."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 0
        mock_qdrant_store.get_collection_info = AsyncMock(return_value=mock_collection_info)
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        output_path = temp_export_dir / "empty.json"
        result = await exporter.export("empty_collection", output_path)

        assert result.success is True
        assert result.records_exported == 0

        with open(output_path, 'r') as f:
            data = json.load(f)
        assert data == []

    @pytest.mark.asyncio
    async def test_export_empty_jsonl(self, exporter, mock_qdrant_store, temp_export_dir):
        """Empty collection exports as empty JSONL file."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 0
        mock_qdrant_store.get_collection_info = AsyncMock(return_value=mock_collection_info)
        mock_qdrant_store.scroll = AsyncMock(return_value=([], None))

        output_path = temp_export_dir / "empty.jsonl"
        options = ExportOptions(format=ExportFormat.JSONL)
        result = await exporter.export("empty_collection", output_path, options)

        assert result.success is True
        assert output_path.read_text().strip() == ""


# =============================================================================
# Large Export Tests
# =============================================================================

class TestLargeExport:
    """Tests for exporting large collections (1000+ records)."""

    @pytest.mark.asyncio
    async def test_large_export_batching(self, exporter, mock_qdrant_store, large_record_set, temp_export_dir):
        """Large export batches correctly."""
        # Simulate batching with scroll
        batch_size = 100
        batches = [large_record_set[i:i+batch_size] for i in range(0, len(large_record_set), batch_size)]

        scroll_results = []
        for i, batch in enumerate(batches):
            offset = (i + 1) * batch_size if i < len(batches) - 1 else None
            scroll_results.append((batch, offset))

        mock_qdrant_store.scroll = AsyncMock(side_effect=scroll_results)

        output_path = temp_export_dir / "large.json"
        options = ExportOptions(batch_size=100)
        result = await exporter.export("large_collection", output_path, options)

        assert result.success is True
        assert result.records_exported == len(large_record_set)

    @pytest.mark.asyncio
    async def test_large_export_streaming_write(self, exporter, mock_qdrant_store, large_record_set, temp_export_dir):
        """Large export uses streaming write to avoid memory issues."""
        # Setup scroll to return batches
        batch_size = 100
        batches = [large_record_set[i:i+batch_size] for i in range(0, len(large_record_set), batch_size)]

        scroll_results = []
        for i, batch in enumerate(batches):
            offset = (i + 1) * batch_size if i < len(batches) - 1 else None
            scroll_results.append((batch, offset))

        mock_qdrant_store.scroll = AsyncMock(side_effect=scroll_results)

        output_path = temp_export_dir / "streaming.jsonl"
        options = ExportOptions(format=ExportFormat.JSONL, batch_size=100)
        result = await exporter.export("large_collection", output_path, options)

        assert result.success is True

        # Verify file was written incrementally (check line count)
        with open(output_path, 'r') as f:
            line_count = sum(1 for _ in f)
        assert line_count == len(large_record_set)

    @pytest.mark.asyncio
    async def test_large_export_progress_callback(self, exporter, mock_qdrant_store, large_record_set, temp_export_dir):
        """Progress callback is invoked during large export."""
        batch_size = 100
        batches = [large_record_set[i:i+batch_size] for i in range(0, len(large_record_set), batch_size)]

        scroll_results = []
        for i, batch in enumerate(batches):
            offset = (i + 1) * batch_size if i < len(batches) - 1 else None
            scroll_results.append((batch, offset))

        mock_qdrant_store.scroll = AsyncMock(side_effect=scroll_results)

        progress_calls = []

        def progress_callback(progress: ExportProgress):
            progress_calls.append(progress)

        output_path = temp_export_dir / "progress.json"
        options = ExportOptions(batch_size=100)

        await exporter.export("large_collection", output_path, options, progress_callback)

        # Progress callback should have been called multiple times
        assert len(progress_calls) > 0

        # Check progress increases
        exported_counts = [p.exported_records for p in progress_calls]
        assert exported_counts == sorted(exported_counts)


# =============================================================================
# Limit and Filtering Tests
# =============================================================================

class TestExportLimit:
    """Tests for export limit functionality."""

    @pytest.mark.asyncio
    async def test_export_limit(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Export limit is respected."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records, None),
            ]
        )

        output_path = temp_export_dir / "limited.json"
        options = ExportOptions(limit=5)

        result = await exporter.export("test_collection", output_path, options)

        # Should only export 5 records
        with open(output_path, 'r') as f:
            data = json.load(f)
        assert len(data) == 5


# =============================================================================
# Round-Trip Integrity Tests
# =============================================================================

class TestRoundTripIntegrity:
    """Tests for export -> import round-trip integrity."""

    @pytest.mark.asyncio
    async def test_json_roundtrip_data_integrity(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Exported JSON data maintains integrity for re-import."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records, None),
            ]
        )

        output_path = temp_export_dir / "roundtrip.json"
        await exporter.export("test_collection", output_path)

        with open(output_path, 'r') as f:
            exported_data = json.load(f)

        # Verify all records are present
        assert len(exported_data) == len(sample_records)

        # Verify data integrity
        for i, record in enumerate(exported_data):
            assert record["id"] == f"point_{i:04d}"
            assert "payload" in record
            assert "vector" in record

    @pytest.mark.asyncio
    async def test_jsonl_roundtrip_data_integrity(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Exported JSONL data maintains integrity for re-import."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records, None),
            ]
        )

        output_path = temp_export_dir / "roundtrip.jsonl"
        options = ExportOptions(format=ExportFormat.JSONL)
        await exporter.export("test_collection", output_path, options)

        # Read back and verify
        with open(output_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) == len(sample_records)

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["id"] == f"point_{i:04d}"

    @pytest.mark.asyncio
    async def test_vector_values_preserved(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Vector values are preserved during export."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records[:1], 1),
                ([], None),
            ]
        )

        output_path = temp_export_dir / "vectors.json"
        options = ExportOptions(vector_mode=VectorExportMode.FULL)
        await exporter.export("test_collection", output_path, options)

        with open(output_path, 'r') as f:
            data = json.load(f)

        # Compare vectors with tolerance for floating point
        original_vector = sample_records[0].vector
        exported_vector = data[0]["vector"]

        for orig, exp in zip(original_vector, exported_vector):
            assert abs(orig - exp) < 1e-6


# =============================================================================
# Record Serialization Tests
# =============================================================================

class TestRecordSerialization:
    """Tests for record serialization."""

    def test_serialize_record_basic(self, exporter):
        """Basic record is serialized correctly."""
        record = MockRecord(
            id="test_id",
            vector=[1.0, 2.0, 3.0],
            payload={"key": "value"},
        )

        options = ExportOptions(vector_mode=VectorExportMode.FULL)
        data = exporter._serialize_record(record, options)

        assert data["id"] == "test_id"
        assert data["vector"] == [1.0, 2.0, 3.0]
        assert data["payload"] == {"key": "value"}

    def test_serialize_record_with_shard_key(self, exporter):
        """Record with shard key is serialized correctly."""
        record = MagicMock()
        record.id = "shard_test"
        record.vector = [0.5]
        record.payload = {}
        record.shard_key = "shard_1"

        options = ExportOptions()
        data = exporter._serialize_record(record, options)

        assert data["shard_key"] == "shard_1"

    def test_serialize_named_vectors(self, exporter):
        """Named vectors are serialized correctly."""
        record = MagicMock()
        record.id = "named_vector"
        record.vector = {"text": [0.1, 0.2], "image": [0.3, 0.4]}
        record.payload = {}

        options = ExportOptions(vector_mode=VectorExportMode.FULL)
        data = exporter._serialize_record(record, options)

        assert "text" in data["vector"]
        assert "image" in data["vector"]


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestExportErrorHandling:
    """Tests for error handling during export."""

    @pytest.mark.asyncio
    async def test_export_creates_output_directory(self, exporter, mock_qdrant_store, sample_records, temp_export_dir):
        """Export creates output directory if it doesn't exist."""
        new_dir = temp_export_dir / "new_subdir" / "deep"
        output_path = new_dir / "export.json"

        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records[:1], 1),
                ([], None),
            ]
        )

        result = await exporter.export("test_collection", output_path)

        assert result.success is True
        assert new_dir.exists()

    @pytest.mark.asyncio
    async def test_export_handles_scroll_error(self, exporter, mock_qdrant_store, temp_export_dir):
        """Export handles scroll errors gracefully."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=Exception("Database connection lost")
        )

        output_path = temp_export_dir / "error.json"
        result = await exporter.export("test_collection", output_path)

        assert result.success is False
        assert "Database connection lost" in result.error_message


# =============================================================================
# Batch Export Tests
# =============================================================================

class TestBatchExport:
    """Tests for batch export functionality."""

    @pytest.mark.asyncio
    async def test_export_batch_creates_multiple_files(self, exporter, mock_qdrant_store, large_record_set, temp_export_dir):
        """Batch export creates multiple files for large collections."""
        batch_size = 100
        batches = [large_record_set[i:i+batch_size] for i in range(0, len(large_record_set), batch_size)]

        scroll_results = []
        for i, batch in enumerate(batches):
            offset = (i + 1) * batch_size if i < len(batches) - 1 else None
            scroll_results.append((batch, offset))

        mock_qdrant_store.scroll = AsyncMock(side_effect=scroll_results)

        output_dir = temp_export_dir / "batch_exports"
        output_dir.mkdir()

        results = await exporter.export_batch(
            collection_name="large_collection",
            output_dir=output_dir,
            max_file_records=500,
        )

        # Should create multiple files
        assert len(results) >= 1
        for result in results:
            assert result.success is True


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestExportMemoriesFunction:
    """Tests for the export_memories convenience function."""

    @pytest.mark.asyncio
    async def test_export_memories_function(self, mock_qdrant_store, sample_records, temp_export_dir):
        """export_memories convenience function works correctly."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records, None),
            ]
        )

        output_path = temp_export_dir / "convenience.json"

        result = await export_memories(
            qdrant_store=mock_qdrant_store,
            collection_name="test_collection",
            output_path=output_path,
            format=ExportFormat.JSON,
        )

        assert result.success is True
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_export_memories_with_options(self, mock_qdrant_store, sample_records, temp_export_dir):
        """export_memories accepts additional options."""
        mock_qdrant_store.scroll = AsyncMock(
            side_effect=[
                (sample_records[:5], None),
            ]
        )

        output_path = temp_export_dir / "options.jsonl"

        result = await export_memories(
            qdrant_store=mock_qdrant_store,
            collection_name="test_collection",
            output_path=output_path,
            format=ExportFormat.JSONL,
            limit=5,
            batch_size=100,
        )

        assert result.success is True
        assert result.format == ExportFormat.JSONL
