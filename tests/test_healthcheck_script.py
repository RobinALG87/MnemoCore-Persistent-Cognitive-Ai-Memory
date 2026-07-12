import importlib.util
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts" / "ops" / "healthcheck.py"
SPEC = importlib.util.spec_from_file_location("mnemocore_healthcheck", SCRIPT)
assert SPEC and SPEC.loader
healthcheck = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(healthcheck)


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self._body


def test_healthcheck_uses_loopback_liveness_by_default(monkeypatch):
    monkeypatch.delenv("MNEMOCORE_HEALTHCHECK_MODE", raising=False)
    with patch.object(
        healthcheck.urllib.request,
        "urlopen",
        return_value=_Response(b'{"status":"degraded"}'),
    ) as urlopen:
        assert healthcheck.check_health() is True

    assert urlopen.call_args.args[0].full_url == "http://127.0.0.1:8100/health"


def test_healthcheck_can_select_readiness_with_environment(monkeypatch):
    monkeypatch.setenv("MNEMOCORE_HEALTHCHECK_MODE", "readiness")
    with patch.object(
        healthcheck.urllib.request,
        "urlopen",
        return_value=_Response(b'{"status":"ready"}'),
    ) as urlopen:
        assert healthcheck.check_health() is True

    assert urlopen.call_args.args[0].full_url == "http://127.0.0.1:8100/ready"


def test_healthcheck_rejects_not_ready_response(monkeypatch):
    monkeypatch.setenv("MNEMOCORE_HEALTHCHECK_MODE", "readiness")
    with patch.object(
        healthcheck.urllib.request,
        "urlopen",
        return_value=_Response(b'{"status":"not_ready"}'),
    ):
        assert healthcheck.check_health() is False
