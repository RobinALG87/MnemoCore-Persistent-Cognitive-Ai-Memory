"""
JSON compatibility layer for MnemoCore.
Auto-detects and uses `orjson` (written in Rust, releases GIL) if available,
falling back to the standard Python `json` library.
This is critical for ensuring serialization and deserialization of massive
HDV arrays does not block the main asyncio event loop on edge devices.
"""
import json as std_json

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False


def loads(obj):
    if ORJSON_AVAILABLE:
        return orjson.loads(obj)
    if isinstance(obj, bytes):
        return std_json.loads(obj.decode('utf-8'))
import numpy as np
from datetime import datetime, date

class FallbackEncoder(std_json.JSONEncoder):
    def __init__(self, **kwargs):
        self.user_default = kwargs.pop('default', None)
        super().__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if self.user_default:
            try:
                return self.user_default(obj)
            except Exception:
                pass
        try:
            return str(obj)
        except Exception:
            return super().default(obj)

def dumps(obj, **kwargs):
    if ORJSON_AVAILABLE:
        # Handle numpy and support str fallbacks automatically
        option = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS | orjson.OPT_PASSTHROUGH_DATETIME
        if kwargs.get('indent') == 2:
            option |= orjson.OPT_INDENT_2
            
        default_fn = kwargs.get('default', str)
        # orjson dumps returns bytes; for drop-in compatibility, return str.
        return orjson.dumps(obj, option=option, default=default_fn).decode('utf-8')
        
    kwargs['cls'] = FallbackEncoder
    return std_json.dumps(obj, **kwargs)


def dump(obj, fp, **kwargs):
    """Note: assumes fp is opened in text mode for drop-in compatibility."""
    fp.write(dumps(obj, **kwargs))


def load(fp):
    return loads(fp.read())
