# MnemoCore Scripts

This directory contains utility scripts organized by purpose.

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `debug/`  | Debugging and troubleshooting scripts |
| `ops/`    | Operational and verification scripts |

## Scripts Overview

### Debug Scripts (`debug/`)

| Script | Description | Usage |
|--------|-------------|-------|
| `bisect_import.py` | Step-by-step import debugging for identifying import issues | `python scripts/debug/bisect_import.py` |
| `debug_async.py` | Debug async storage with mock client | `python scripts/debug/debug_async.py` |
| `debug_imports.py` | Test import of all core modules | `python scripts/debug/debug_imports.py` |
| `debug_qdrant.py` | Debug Qdrant client initialization and collections | `python scripts/debug/debug_qdrant.py` |

### Ops Scripts (`ops/`)

| Script | Description | Usage |
|--------|-------------|-------|
| `healthcheck.py` | Docker healthcheck script for /health endpoint | `python scripts/ops/healthcheck.py` |
| `verify_id.py` | Verify UUID format and memory retrieval functionality | `python scripts/ops/verify_id.py` |

## Usage Notes

- All scripts should be run from the project root directory
- Debug scripts are intended for development troubleshooting
- Ops scripts are intended for operational verification and maintenance
