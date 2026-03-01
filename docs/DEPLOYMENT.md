# MnemoCore Deployment Guide

> **Version**: 5.1.0

This guide covers deploying MnemoCore in development, Docker, and Kubernetes environments.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Docker Compose (Recommended)](#docker-compose-recommended)
- [Kubernetes / Helm](#kubernetes--helm)
- [Environment Variables](#environment-variables)
- [Health Checks](#health-checks)
- [Monitoring & Observability](#monitoring--observability)
- [Production Checklist](#production-checklist)

---

## Prerequisites

| Component | Required Version | Purpose |
|-----------|-----------------|---------|
| Python | 3.11+ | Runtime |
| Redis | 7.x | Warm tier storage, streams |
| Qdrant | 1.12+ | Vector search, cold tier |
| Docker | 24+ | Container runtime (optional) |
| Helm | 3.x | Kubernetes deployment (optional) |

---

## Local Development

### Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/mnemocore.git
cd mnemocore

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing

# Copy and edit config
cp config.yaml config.local.yaml
# Edit config.local.yaml with your settings

# Run the API server
uvicorn src.mnemocore.api.main:app --host 127.0.0.1 --port 8100 --reload
```

### Running Without External Services

MnemoCore works in standalone mode without Redis or Qdrant. It uses in-memory storage for the hot tier and filesystem for warm/cold:

```yaml
# config.yaml for standalone mode
tiers:
  hot:
    max_memories: 2000
    storage_backend: memory
  warm:
    storage_backend: mmap
  cold:
    storage_backend: filesystem
```

### Running Tests

```bash
# Full test suite
pytest tests/ -q --timeout=30

# Specific test modules
pytest tests/test_cognitive_services.py -v

# With coverage
pytest tests/ --cov=src/mnemocore --cov-report=html
```

---

## Docker Deployment

### Building the Image

```bash
docker build -t mnemocore:latest .
```

The Dockerfile uses a multi-stage build:

1. **Builder stage**: `python:3.11.8-slim-bookworm` — installs dependencies in a venv
2. **Production stage**: Copies only the venv and source, runs as non-root user (`mnemocore:1000`)

### Running Standalone

```bash
docker run -d \
  --name mnemocore-api \
  -p 8100:8100 \
  -p 9090:9090 \
  -v mnemocore-data:/app/data \
  -e HAIM_API_KEY="your-secret-key" \
  -e LOG_LEVEL=INFO \
  mnemocore:latest
```

### Image Details

| Property | Value |
|----------|-------|
| Base Image | `python:3.11.8-slim-bookworm` |
| User | `mnemocore` (UID 1000) |
| Working Dir | `/app` |
| Ports | `8100` (API), `9090` (metrics) |
| Entrypoint | `uvicorn src.api.main:app` |
| Healthcheck | `python /app/scripts/ops/healthcheck.py` |
| Memory Limit | 2G (recommended) |

---

## Docker Compose (Recommended)

The included `docker-compose.yml` deploys MnemoCore with all dependencies:

### Services

| Service | Image | Purpose | Ports |
|---------|-------|---------|-------|
| `mnemocore` | Custom build | API server | `8100`, `9090` |
| `redis` | `redis:7.2-alpine` | Caching, streams | `6379` (internal) |
| `qdrant` | `qdrant/qdrant:v1.12.1` | Vector search | `6333` (internal) |

### Setup

1. **Create an `.env` file**:

```env
HAIM_API_KEY=your-secure-api-key
REDIS_PASSWORD=a-strong-redis-password
QDRANT_API_KEY=a-strong-qdrant-key
LOG_LEVEL=INFO
```

2. **Start all services**:

```bash
docker compose up -d
```

3. **Verify**:

```bash
# Check health
curl http://localhost:8100/health

# Check logs
docker compose logs mnemocore -f
```

4. **Stop**:

```bash
docker compose down          # Stop services
docker compose down -v       # Stop and remove volumes
```

### Resource Limits

| Service | Memory Limit | Memory Reserved |
|---------|-------------|-----------------|
| MnemoCore | 2 GB | 512 MB |
| Redis | 512 MB | — |
| Qdrant | 4 GB | — |

### Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `mnemocore-data` | `/app/data` | Memory files, dream reports, audit logs |
| `mnemocore-redis-data` | `/data` (Redis) | Redis persistence |
| `mnemocore-qdrant-storage` | `/qdrant/storage` | Qdrant vector data |

---

## Kubernetes / Helm

A Helm chart is provided in `helm/mnemocore/`.

### Chart Structure

```
helm/mnemocore/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml          # MnemoCore API
│   ├── deployment-redis.yaml    # Redis
│   ├── deployment-qdrant.yaml   # Qdrant
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── pvc.yaml
│   ├── hpa.yaml                 # Horizontal Pod Autoscaler
│   ├── pdb.yaml                 # Pod Disruption Budget
│   ├── networkpolicy.yaml
│   ├── serviceaccount.yaml
│   ├── servicemonitor.yaml      # Prometheus ServiceMonitor
│   └── _helpers.tpl
```

### Installation

```bash
# Install with defaults
helm install mnemocore ./helm/mnemocore

# Install with custom values
helm install mnemocore ./helm/mnemocore \
  --set api.replicaCount=3 \
  --set api.image.tag=5.1.0 \
  --set secrets.apiKey="your-key" \
  --set qdrant.storage.size=50Gi

# Upgrade
helm upgrade mnemocore ./helm/mnemocore -f custom-values.yaml

# Uninstall
helm uninstall mnemocore
```

### Key Values

```yaml
# values.yaml highlights
api:
  replicaCount: 2
  image:
    repository: mnemocore
    tag: "5.1.0"
  resources:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 2Gi
      cpu: "1"

redis:
  enabled: true
  storage:
    size: 5Gi

qdrant:
  enabled: true
  storage:
    size: 50Gi

ingress:
  enabled: false
  className: nginx
  hosts:
    - host: mnemocore.example.com
      paths:
        - path: /
          pathType: Prefix

autoscaling:
  enabled: false
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilization: 70
```

### Features

- **HPA**: Auto-scale API pods based on CPU/memory
- **PDB**: Ensures minimum available pods during disruptions
- **NetworkPolicy**: Restricts traffic between components
- **ServiceMonitor**: Prometheus auto-discovery for metrics
- **TLS**: Ingress TLS termination support

---

## Environment Variables

All sensitive configuration should be passed via environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HAIM_API_KEY` | Yes (prod) | — | API authentication key |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection string |
| `REDIS_PASSWORD` | No | — | Redis password |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant URL |
| `QDRANT_API_KEY` | No | — | Qdrant API key |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `HOST` | No | `0.0.0.0` | API bind host |
| `PORT` | No | `8100` | API bind port |

See [CONFIGURATION.md](CONFIGURATION.md) for the complete variable list.

---

## Health Checks

### HTTP Health Endpoint

```bash
curl http://localhost:8100/health
```

Returns `200 OK` with:

```json
{
  "status": "healthy",
  "redis_connected": true,
  "qdrant_circuit_breaker": "closed",
  "engine_ready": true
}
```

### Docker Healthcheck

Built into the Dockerfile:

```
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3
    CMD python /app/scripts/ops/healthcheck.py || exit 1
```

### Kubernetes Probes

Configure in the Helm values:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8100
  initialDelaySeconds: 40
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8100
  initialDelaySeconds: 10
  periodSeconds: 10
```

---

## Monitoring & Observability

### Prometheus Metrics

Exposed at `GET /metrics` (port 8100) and optionally at port `9090`.

Key metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `haim_store_total` | Counter | Total store operations |
| `haim_query_total` | Counter | Total query operations |
| `haim_store_latency_seconds` | Histogram | Store latency |
| `haim_query_latency_seconds` | Histogram | Query latency |
| `haim_tier_count` | Gauge | Memories per tier |

### Grafana Dashboard

Import the included `grafana-dashboard.json` into Grafana for a pre-built dashboard with:

- Request rate and latency panels
- Tier distribution
- Dream cycle activity
- Error rates

### Structured Logging

Enable structured JSON logging:

```yaml
observability:
  structured_logging: true
  log_level: INFO
```

Logs use `loguru` for services and stdlib `logging` for the pulse loop and API layer.

---

## Production Checklist

Before deploying to production:

- [ ] Set `HAIM_API_KEY` to a strong, unique value
- [ ] Set `REDIS_PASSWORD` and `QDRANT_API_KEY`
- [ ] Configure CORS origins for your domain
- [ ] Enable rate limiting (`security.rate_limit_enabled: true`)
- [ ] Set `subconscious_ai.dry_run: false` only after validating behavior
- [ ] Configure backup snapshots (`backup.auto_snapshot_enabled: true`)
- [ ] Set appropriate resource limits (memory, CPU)
- [ ] Configure monitoring (Prometheus + Grafana)
- [ ] Enable structured logging for log aggregation
- [ ] Review [SECURITY.md](../SECURITY.md) for security considerations
- [ ] Test health endpoint from your monitoring system
- [ ] Set up alerting for degraded health status

---

*See [CONFIGURATION.md](CONFIGURATION.md) for all config options. See [ARCHITECTURE.md](ARCHITECTURE.md) for system design.*
