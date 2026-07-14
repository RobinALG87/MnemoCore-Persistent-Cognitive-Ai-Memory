# MnemoCore prototype deployment

This guide describes the currently verified single-node prototype. It is a
working deployment baseline, not a production-readiness claim. AgentMemory can
also be used directly as a local SQLite-backed Python library without running
the REST service, Redis, or Qdrant.

## Local API

Install the project, provide a non-default API key, and bind to loopback while
developing:

```bash
pip install -e .
export HAIM_API_KEY="replace-with-a-long-random-value"
uvicorn mnemocore.api.main:app --host 127.0.0.1 --port 8100
```

The supported runtime endpoints are:

- `GET /health` — liveness and dependency diagnostics. A local-only runtime can
  report `degraded` when Redis is unavailable and still remain live.
- `GET /ready` — readiness of the initialized local runtime. It returns
  `ready` when the engine/container are initialized; Redis is diagnostic rather
  than mandatory for this prototype.
- `GET /metrics/` — Prometheus exposition on the API listener, port 8100. The
  trailing slash is canonical.

## v3 scoped API composition

The v3 memory routes are deliberately **not** enabled by the legacy `app` by
default. A global API key is not a sufficient authorization model for
scope-isolated memory: callers must be authorized for the full requested
`MemoryScope` before a database is opened.

Deploy the standalone v3 application only through an explicit composition root
that supplies a `ScopeAuthorizer` mapping authenticated credentials to allowed
complete scopes:

```python
from mnemocore.api.v3_app import create_v3_app

app = create_v3_app(
    sqlite_path="/app/data/agent-memory.sqlite3",
    scope_authorizer=my_scope_authorizer,
)
```

If no authorizer is supplied, v3 memory requests fail closed. The library does
not ship a credential-to-scope mapping because that mapping belongs to the
deploying application's identity system. Do not expose a v3 application until
that mapping, TLS, and the usual reverse-proxy controls are in place.

## Docker

The image builds a wheel, installs it into the runtime stage, runs as an
unprivileged user, and starts `mnemocore.api.main:app`:

```bash
docker build -t mnemocore:prototype .
docker run --rm \
  --publish 127.0.0.1:8100:8100 \
  --env HAIM_API_KEY="replace-with-a-long-random-value" \
  --volume mnemocore-data:/app/data \
  mnemocore:prototype
```

Smoke check:

```bash
curl --fail http://127.0.0.1:8100/health
curl --fail http://127.0.0.1:8100/ready
curl --fail http://127.0.0.1:8100/metrics/
```

`HAIM_API_KEY` is required by the container entrypoint. Do not expose port 8100
directly to the internet; terminate TLS and enforce network policy at a trusted
reverse proxy or ingress.

## Docker Compose

Compose starts MnemoCore, Redis, and Qdrant. It fails during interpolation
unless all three secrets are explicitly supplied:

```bash
cp .env.example .env
# Set HAIM_API_KEY, REDIS_PASSWORD, and QDRANT_API_KEY in .env.
docker compose config
docker compose up --build -d
```

Only the API is published, on `127.0.0.1:8100`. Redis and Qdrant stay on the
internal Compose network. Metrics use `http://127.0.0.1:8100/metrics/`; there is
no separate port 9090 listener.

Operational commands:

```bash
docker compose ps
docker compose logs --follow mnemocore
docker compose down
```

Avoid `docker compose down -v` unless irreversible deletion of local prototype
volumes is intended.

## Kubernetes / Helm

The checked-in Helm defaults describe a single-node prototype:

- one MnemoCore replica;
- API and metrics share container/service port 8100;
- liveness uses `/health`, readiness uses `/ready`;
- ServiceMonitor scrapes `/metrics/`;
- bundled Redis and Qdrant are single-replica dependencies.

Render and validate before applying:

```bash
helm dependency build helm/mnemocore
helm lint helm/mnemocore
helm template mnemo helm/mnemocore --namespace mnemocore > rendered.yaml
```

Use an externally managed Kubernetes Secret or the chart's explicit secret
inputs. Never commit real secret values. Multi-replica durability, coordinated
background work, external backing services, disruption budgets, backup/restore,
and disaster recovery are not yet verified and must not be inferred from the
chart.

## Persistence and operations

- Mount `/app/data` on durable storage for local files.
- Back up data only while following a SQLite-aware procedure that includes WAL
  state; a copy of the main database file alone is not a proven restore.
- Exercise restore and rollback in the target environment before relying on
  backups.
- Alert separately on liveness, readiness, and dependency degradation.
- Treat logs and metrics as potentially sensitive operational metadata.

AgentMemory's physical erase prototype rewrites the SQLite database under a
cooperative cross-process sidecar lock. It verifies exact scope ownership,
requires an explicit cascade for connected supersession streams, removes
dependent rows, checks foreign keys/integrity, and returns a content-free
receipt. Every process must use `SQLiteMemoryStore` and its lock contract during
erasure; uncoordinated raw SQLite connections are unsafe. Current tests do not
yet prove arbitrary power-loss points, all filesystem failure modes, backup
purging, or deletion from external derived artifacts.

Persistent webhook configuration must use an opaque `secret_ref`. The secret is
resolved at delivery time and is not written to the webhook JSON file. Inline
secrets, legacy plaintext documents, and all persistent custom headers are
rejected. Persistence mutations are serialized and use atomic replacement; load
and write failures fail closed or report that replacement already committed.

## Remaining production gates

The prototype is intentionally not labelled production-ready until all of the
following are evidenced:

- security and dependency findings are resolved or explicitly accepted;
- physical erasure is extended from the current coordinated SQLite prototype to
  cover exhaustive failure injection, power loss, backups, and external derived
  artifacts;
- the quarantined lifecycle test and legacy lane are migrated or retired with
  an explicit compatibility decision;
- real Redis/Qdrant service tests, backup/restore, upgrade/rollback, load, and
  soak tests pass;
- Helm dependencies, secret delivery, storage classes, and recovery behavior
  are validated on the target cluster.

See [the release checklist](../RELEASE_CHECKLIST.md) and
[the current platform status](status/2026-07-12-platform-baseline.md) for the
authoritative public gates.
