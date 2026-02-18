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

import os
import sys
import urllib.request
import urllib.error
import json

# Configuration from environment or defaults
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = os.environ.get("PORT", "8100")
HEALTH_ENDPOINT = f"http://{HOST}:{PORT}/health"
TIMEOUT_SECONDS = 5


def check_health() -> bool:
    """
    Perform health check against the /health endpoint.

    Returns:
        bool: True if healthy, False otherwise
    """
    try:
        request = urllib.request.Request(
            HEALTH_ENDPOINT,
            method="GET",
            headers={"Accept": "application/json"}
        )

        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                print(f"Health check failed: HTTP {response.status}", file=sys.stderr)
                return False

            data = json.loads(response.read().decode("utf-8"))

            # Check if status is "healthy"
            status = data.get("status", "")
            if status == "healthy":
                print(f"Health check passed: {status}")
                return True
            elif status == "degraded":
                # Degraded is still operational, consider it healthy
                print(f"Health check passed (degraded): {data}")
                return True
            else:
                print(f"Health check failed: unexpected status '{status}'", file=sys.stderr)
                return False

    except urllib.error.URLError as e:
        print(f"Health check failed: connection error - {e.reason}", file=sys.stderr)
        return False
    except urllib.error.HTTPError as e:
        print(f"Health check failed: HTTP {e.code} - {e.reason}", file=sys.stderr)
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
    is_healthy = check_health()
    exit_code = 0 if is_healthy else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
