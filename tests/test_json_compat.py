"""
Tests for JSON Compatibility Layer
===================================
Tests for src/mnemocore/utils/json_compat.py covering FallbackEncoder,
orjson vs stdlib fallback, dumps/loads with various types, and edge cases.
"""

import json as std_json
import pytest
import numpy as np
from datetime import datetime, date
from unittest.mock import patch

from mnemocore.utils.json_compat import (
    loads,
    dumps,
    dump,
    load,
    FallbackEncoder,
    JSONDecodeError,
    ORJSON_AVAILABLE,
)


class TestFallbackEncoder:
    """Tests for FallbackEncoder class."""

    def test_encode_datetime(self):
        """Test encoding datetime objects to ISO format."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        encoder = FallbackEncoder()
        result = encoder.default(dt)
        assert result == "2024-01-15T10:30:00"

    def test_encode_date(self):
        """Test encoding date objects to ISO format."""
        d = date(2024, 1, 15)
        encoder = FallbackEncoder()
        result = encoder.default(d)
        assert result == "2024-01-15"

    def test_encode_numpy_array(self):
        """Test encoding numpy arrays to list."""
        arr = np.array([1, 2, 3, 4, 5])
        encoder = FallbackEncoder()
        result = encoder.default(arr)
        assert result == [1, 2, 3, 4, 5]

    def test_encode_numpy_float_array(self):
        """Test encoding numpy float arrays."""
        arr = np.array([1.5, 2.5, 3.5])
        encoder = FallbackEncoder()
        result = encoder.default(arr)
        assert result == [1.5, 2.5, 3.5]

    def test_encode_numpy_2d_array(self):
        """Test encoding 2D numpy arrays."""
        arr = np.array([[1, 2], [3, 4]])
        encoder = FallbackEncoder()
        result = encoder.default(arr)
        assert result == [[1, 2], [3, 4]]

    def test_encode_custom_object_with_user_default(self):
        """Test encoding custom objects using user-provided default function."""
        class CustomObj:
            def __init__(self, value):
                self.value = value

        def custom_default(obj):
            if isinstance(obj, CustomObj):
                return {"custom": obj.value}
            raise TypeError(f"Cannot serialize {type(obj)}")

        encoder = FallbackEncoder(default=custom_default)
        obj = CustomObj("test_value")
        result = encoder.default(obj)
        assert result == {"custom": "test_value"}

    def test_encode_unknown_object_fallback_to_str(self):
        """Test that unknown objects fall back to string representation."""
        class UnknownObj:
            def __str__(self):
                return "UnknownObject()"

        encoder = FallbackEncoder()
        obj = UnknownObj()
        result = encoder.default(obj)
        assert result == "UnknownObject()"


class TestLoads:
    """Tests for loads function."""

    def test_loads_string(self):
        """Test loading JSON from string."""
        result = loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_loads_bytes(self):
        """Test loading JSON from bytes."""
        result = loads(b'{"key": "value"}')
        assert result == {"key": "value"}

    def test_loads_list(self):
        """Test loading JSON list."""
        result = loads('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_loads_nested(self):
        """Test loading nested JSON."""
        result = loads('{"outer": {"inner": [1, 2, 3]}}')
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_loads_invalid_raises(self):
        """Test that invalid JSON raises JSONDecodeError."""
        with pytest.raises(JSONDecodeError):
            loads('not valid json')


class TestDumps:
    """Tests for dumps function."""

    def test_dumps_basic_dict(self):
        """Test dumping basic dictionary."""
        data = {"key": "value"}
        result = dumps(data)
        # Should be valid JSON string
        parsed = std_json.loads(result)
        assert parsed == data

    def test_dumps_list(self):
        """Test dumping list."""
        data = [1, 2, 3, "four"]
        result = dumps(data)
        parsed = std_json.loads(result)
        assert parsed == data

    def test_dumps_with_datetime(self):
        """Test dumping dictionary with datetime."""
        data = {"timestamp": datetime(2024, 1, 15, 10, 30, 0)}
        result = dumps(data)
        # Should contain ISO formatted date
        assert "2024-01-15" in result

    def test_dumps_with_numpy_array(self):
        """Test dumping dictionary with numpy array."""
        data = {"array": np.array([1, 2, 3])}
        result = dumps(data)
        parsed = std_json.loads(result)
        assert parsed["array"] == [1, 2, 3]

    def test_dumps_with_indent(self):
        """Test dumping with indentation."""
        data = {"key": "value"}
        result = dumps(data, indent=2)
        # Should have newlines for pretty printing
        assert "\n" in result or "  " in result

    def test_dumps_none(self):
        """Test dumping None value."""
        result = dumps(None)
        assert result == "null"

    def test_dumps_boolean(self):
        """Test dumping boolean values."""
        assert std_json.loads(dumps(True)) is True
        assert std_json.loads(dumps(False)) is False

    def test_dumps_number(self):
        """Test dumping numeric values."""
        assert std_json.loads(dumps(42)) == 42
        assert std_json.loads(dumps(3.14)) == 3.14

    def test_dumps_empty_dict(self):
        """Test dumping empty dictionary."""
        result = dumps({})
        assert result == "{}"

    def test_dumps_empty_list(self):
        """Test dumping empty list."""
        result = dumps([])
        assert result == "[]"


class TestDumpsEdgeCases:
    """Tests for dumps function edge cases."""

    def test_dumps_nan(self):
        """Test dumping NaN value."""
        # NaN handling varies by implementation
        data = {"value": float('nan')}
        result = dumps(data)
        # Should not raise an error
        assert isinstance(result, str)

    def test_dumps_infinity(self):
        """Test dumping infinity values."""
        # Infinity handling varies by implementation
        data = {"pos_inf": float('inf'), "neg_inf": float('-inf')}
        result = dumps(data)
        # Should not raise an error
        assert isinstance(result, str)

    def test_dumps_mixed_types(self):
        """Test dumping complex mixed type structure."""
        data = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
        }
        result = dumps(data)
        parsed = std_json.loads(result)
        assert parsed["string"] == "value"
        assert parsed["int"] == 42
        assert parsed["bool"] is True
        assert parsed["null"] is None

    def test_dumps_unicode(self):
        """Test dumping unicode strings."""
        data = {"emoji": "\U0001f600", "chinese": "\u4e2d\u6587"}
        result = dumps(data)
        parsed = std_json.loads(result)
        assert parsed["emoji"] == "\U0001f600"
        assert parsed["chinese"] == "\u4e2d\u6587"

    def test_dumps_special_characters(self):
        """Test dumping strings with special characters."""
        data = {"escaped": "Line1\nLine2", "quote": 'He said "hello"'}
        result = dumps(data)
        # Should be valid JSON
        parsed = std_json.loads(result)
        assert "Line1" in parsed["escaped"]


class TestDumpAndLoad:
    """Tests for dump and load file-like functions."""

    def test_dump_and_load_stringio(self):
        """Test dump and load with StringIO."""
        import io
        data = {"key": "value", "number": 42}
        fp = io.StringIO()
        dump(data, fp)
        fp.seek(0)
        result = load(fp)
        assert result == data

    def test_dump_with_numpy(self):
        """Test dump with numpy array."""
        import io
        data = {"array": np.array([1, 2, 3])}
        fp = io.StringIO()
        dump(data, fp)
        fp.seek(0)
        result = load(fp)
        assert result["array"] == [1, 2, 3]


class TestOrjsonAvailability:
    """Tests for orjson availability detection."""

    def test_orjson_available_flag(self):
        """Test that ORJSON_AVAILABLE is a boolean."""
        assert isinstance(ORJSON_AVAILABLE, bool)

    @patch('mnemocore.utils.json_compat.ORJSON_AVAILABLE', False)
    def test_fallback_to_stdlib(self):
        """Test that stdlib is used when orjson is unavailable."""
        # Re-import to get patched value
        import importlib
        import mnemocore.utils.json_compat as json_compat
        importlib.reload(json_compat)

        # Should still work with stdlib
        data = {"key": "value"}
        result = json_compat.dumps(data)
        assert isinstance(result, str)
        assert json_compat.loads(result) == data


class TestJsonDecodeError:
    """Tests for JSONDecodeError compatibility."""

    def test_json_decode_error_exists(self):
        """Test that JSONDecodeError is available."""
        assert JSONDecodeError is not None

    def test_json_decode_error_raised(self):
        """Test that JSONDecodeError is raised for invalid JSON."""
        with pytest.raises(JSONDecodeError):
            loads('{"invalid": }')


class TestNumpyDtypeHandling:
    """Tests for various numpy dtype handling."""

    def test_numpy_int32(self):
        """Test encoding numpy int32 array."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = dumps({"arr": arr})
        parsed = std_json.loads(result)
        assert parsed["arr"] == [1, 2, 3]

    def test_numpy_float64(self):
        """Test encoding numpy float64 array."""
        arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        result = dumps({"arr": arr})
        parsed = std_json.loads(result)
        assert parsed["arr"] == [1.1, 2.2, 3.3]

    def test_numpy_bool(self):
        """Test encoding numpy boolean array."""
        arr = np.array([True, False, True], dtype=np.bool_)
        result = dumps({"arr": arr})
        parsed = std_json.loads(result)
        assert parsed["arr"] == [True, False, True]
