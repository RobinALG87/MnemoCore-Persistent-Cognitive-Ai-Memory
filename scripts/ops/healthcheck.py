#!/usr/bin/env python3
"""
MnemoCore Healthcheck Script
============================
Performs HTTP GET to /health endpoint and returns appropriate exit code.
Designed to be used as Docker healthcheck.

Exit codes:
  0 - Service is healthy
  1 - Service is unhealthy or unreachable
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
import json

# Configuration from environment or defaults
PORT = os.environ.get("PORT", "8100")
TIMEOUT_SECONDS = 5


def check_health(mode: str | None = None) -> bool:
    """
    Perform health check against the /health endpoint.

    Returns:
        bool: True if healthy, False otherwise
    """
    selected_mode = mode or os.environ.get("MNEMOCORE_HEALTHCHECK_MODE", "liveness")
    path = "/ready" if selected_mode == "readiness" else "/health"
    endpoint = f"http://127.0.0.1:{PORT}{path}"
    expected_statuses = {"ready"} if selected_mode == "readiness" else {"healthy", "degraded"}

    try:
        request = urllib.request.Request(
            endpoint,
            method="GET",
            headers={"Accept": "application/json"}
        )

        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                print(f"Health check failed: HTTP {response.status}", file=sys.stderr)
                return False

            data = json.loads(response.read().decode("utf-8"))

            status = data.get("status", "")
            if status in expected_statuses:
                print(f"Health check passed: {status}")
                return True
            print(f"Health check failed: unexpected status '{status}'", file=sys.stderr)
            return False

    except urllib.error.HTTPError as e:
        print(f"Health check failed: HTTP {e.code} - {e.reason}", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"Health check failed: connection error - {e.reason}", file=sys.stderr)
        return False
    except json.JSONDecodeError as e:
        print(f"Health check failed: invalid JSON response - {e}", file=sys.stderr)
        return False
    except TimeoutError:
        print(f"Health check failed: timeout after {TIMEOUT_SECONDS}s", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Health check failed: unexpected error - {e}", file=sys.stderr)
        return False


def main():
    """Main entry point for healthcheck script."""
    parser = argparse.ArgumentParser(description="Check MnemoCore liveness or readiness")
    parser.add_argument(
        "--readiness",
        action="store_true",
        help="check /ready instead of the default /health endpoint",
    )
    args = parser.parse_args()
    is_healthy = check_health("readiness" if args.readiness else None)
    exit_code = 0 if is_healthy else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
