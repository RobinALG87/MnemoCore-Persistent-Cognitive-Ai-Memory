#!/bin/bash
# MnemoCore Docker Entrypoint Script
# ===================================
# Validates required environment variables before starting the application

set -e

# Validate required environment variables
if [ -z "$HAIM_API_KEY" ]; then
  echo "ERROR: HAIM_API_KEY must be set" >&2
  exit 1
fi

# Optional: Warn if using default/insecure API key
if [ "$HAIM_API_KEY" = "changeme" ] || [ "$HAIM_API_KEY" = "ci-test-key-not-for-production" ]; then
  echo "WARNING: Using insecure HAIM_API_KEY. This should not be used in production!" >&2
fi

# Execute the main command
exec "$@"
