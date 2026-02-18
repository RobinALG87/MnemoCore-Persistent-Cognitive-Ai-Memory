# MnemoCore Kubernetes Deployment

This directory contains Kubernetes manifests and Helm charts for deploying MnemoCore to a Kubernetes cluster.

## Overview

MnemoCore is a cognitive memory infrastructure that uses Hyperdimensional Computing (HDC) to provide persistent, scalable memory for AI systems. The Kubernetes deployment includes:

- **MnemoCore API** - Main API service with health checks and metrics
- **Redis** - In-memory data store for hot tier and caching
- **Qdrant** - Vector database for similarity search

## Prerequisites

- Kubernetes 1.25+
- Helm 3.8+
- kubectl configured to access your cluster
- (Optional) Prometheus Operator for metrics scraping
- (Optional) cert-manager for TLS certificates

## Quick Start

### 1. Install using Helm

```bash
# Add required Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm repo update

# Install MnemoCore with default values
helm install mnemocore ./helm/mnemocore \
  --namespace mnemocore \
  --create-namespace \
  --set mnemocore.apiKey.value="your-secure-api-key"
```

### 2. Install with custom values

```bash
# Create a values file
cat > values-prod.yaml << EOF
mnemocore:
  replicaCount: 3
  apiKey:
    existingSecret: mnemocore-api-key
  resources:
    limits:
      cpu: "4"
      memory: "4Gi"
    requests:
      cpu: "1"
      memory: "1Gi"
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 60

redis:
  persistence:
    size: 20Gi

qdrant:
  persistence:
    size: 100Gi

global:
  storageClass: "fast-ssd"
EOF

helm install mnemocore ./helm/mnemocore \
  --namespace mnemocore \
  --create-namespace \
  -f values-prod.yaml
```

### 3. Verify the installation

```bash
# Check pod status
kubectl get pods -n mnemocore

# Check services
kubectl get svc -n mnemocore

# Check HPA status
kubectl get hpa -n mnemocore

# Port-forward for local testing
kubectl port-forward svc/mnemocore 8100:8100 -n mnemocore

# Test the API
curl http://localhost:8100/health
```

## Configuration

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mnemocore.replicaCount` | Number of API replicas | `2` |
| `mnemocore.image.repository` | Container image repository | `mnemocore` |
| `mnemocore.image.tag` | Container image tag | `latest` |
| `mnemocore.resources.limits.cpu` | CPU limit | `2` |
| `mnemocore.resources.limits.memory` | Memory limit | `2Gi` |
| `mnemocore.autoscaling.enabled` | Enable HPA | `true` |
| `mnemocore.autoscaling.minReplicas` | Minimum replicas | `2` |
| `mnemocore.autoscaling.maxReplicas` | Maximum replicas | `10` |
| `mnemocore.apiKey.existingSecret` | Existing secret for API key | `""` |
| `redis.enabled` | Deploy Redis | `true` |
| `qdrant.enabled` | Deploy Qdrant | `true` |

### Resource Limits

| Component | CPU Limit | Memory Limit | CPU Request | Memory Request |
|-----------|-----------|--------------|-------------|----------------|
| MnemoCore | 2 | 2Gi | 500m | 512Mi |
| Redis | 1 | 512Mi | 100m | 128Mi |
| Qdrant | 2 | 4Gi | 500m | 1Gi |

### Probe Configuration

| Probe | Initial Delay | Period | Timeout | Failure Threshold |
|-------|---------------|--------|---------|-------------------|
| Liveness | 40s | 30s | 10s | 3 |
| Readiness | 20s | 10s | 5s | 3 |
| Startup | 10s | 10s | 5s | 30 |

## Production Deployment

### 1. Create Secrets

```bash
# Create API key secret
kubectl create secret generic mnemocore-api-key \
  --from-literal=HAIM_API_KEY='your-secure-api-key' \
  -n mnemocore

# Or use sealed-secrets/external-secrets for GitOps
```

### 2. Configure Storage

```bash
# Ensure you have a storage class configured
kubectl get storageclass

# For production, use fast SSD storage
helm install mnemocore ./helm/mnemocore \
  --namespace mnemocore \
  --set global.storageClass=fast-ssd \
  --set mnemocore.persistence.size=50Gi \
  --set redis.persistence.size=20Gi \
  --set qdrant.persistence.size=200Gi
```

### 3. Enable Ingress

```bash
helm install mnemocore ./helm/mnemocore \
  --namespace mnemocore \
  --set mnemocore.ingress.enabled=true \
  --set mnemocore.ingress.className=nginx \
  --set 'mnemocore.ingress.hosts[0].host=mnemocore.yourdomain.com' \
  --set 'mnemocore.ingress.hosts[0].paths[0].path=/' \
  --set 'mnemocore.ingress.hosts[0].paths[0].pathType=Prefix' \
  --set 'mnemocore.ingress.tls[0].secretName=mnemocore-tls' \
  --set 'mnemocore.ingress.tls[0].hosts[0]=mnemocore.yourdomain.com'
```

### 4. Enable Network Policies

```bash
helm install mnemocore ./helm/mnemocore \
  --namespace mnemocore \
  --set networkPolicy.enabled=true
```

## Monitoring

### Prometheus Integration

```bash
# Enable ServiceMonitor for Prometheus Operator
helm install mnemocore ./helm/mnemocore \
  --namespace mnemocore \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.labels.release=prometheus
```

### Available Metrics

MnemoCore exposes the following metrics on port 9090:

- `mnemocore_memory_count_total` - Total number of memories stored
- `mnemocore_memory_tier_hot` - Number of memories in hot tier
- `mnemocore_memory_tier_warm` - Number of memories in warm tier
- `mnemocore_memory_tier_cold` - Number of memories in cold tier
- `mnemocore_query_duration_seconds` - Query latency histogram
- `mnemocore_ltp_avg` - Average LTP score
- `mnemocore_api_requests_total` - Total API requests
- `mnemocore_api_request_duration_seconds` - API request latency

### Grafana Dashboard

Import the provided `grafana-dashboard.json` to visualize MnemoCore metrics.

## Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment mnemocore --replicas=5 -n mnemocore
```

### Autoscaling

HPA is enabled by default. Customize scaling behavior:

```bash
helm upgrade mnemocore ./helm/mnemocore \
  --namespace mnemocore \
  --set mnemocore.autoscaling.minReplicas=3 \
  --set mnemocore.autoscaling.maxReplicas=50 \
  --set mnemocore.autoscaling.targetCPUUtilizationPercentage=50
```

## Upgrading

```bash
# Upgrade to a new version
helm upgrade mnemocore ./helm/mnemocore \
  --namespace mnemocore \
  --set mnemocore.image.tag=v3.6.0

# Rollback if needed
helm rollback mnemocore -n mnemocore
```

## Troubleshooting

### Check Logs

```bash
# MnemoCore logs
kubectl logs -l app.kubernetes.io/name=mnemocore -n mnemocore -f

# Redis logs
kubectl logs -l app.kubernetes.io/component=redis -n mnemocore -f

# Qdrant logs
kubectl logs -l app.kubernetes.io/component=qdrant -n mnemocore -f
```

### Common Issues

1. **Pod stuck in Pending**
   - Check storage class availability
   - Check resource requests vs node capacity

2. **Health check failing**
   - Check Redis and Qdrant connectivity
   - Verify environment variables

3. **High memory usage**
   - Reduce `mnemocore.config.tiers.hot.max_memories`
   - Enable GPU for faster encoding

### Debug Mode

```bash
# Run with debug logging
helm upgrade mnemocore ./helm/mnemocore \
  --namespace mnemocore \
  --set mnemocore.env.logLevel=DEBUG
```

## Uninstalling

```bash
# Remove the Helm release
helm uninstall mnemocore -n mnemocore

# Remove the namespace (optional)
kubectl delete namespace mnemocore

# Remove PVCs (caution: data loss)
kubectl delete pvc -n mnemocore --all
```

## Architecture

```
                    ┌─────────────────┐
                    │     Ingress     │
                    │   (Optional)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  MnemoCore API  │
                    │   (HPA: 2-10)   │
                    │  Port: 8100     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐    │    ┌────────▼────────┐
     │      Redis      │    │    │     Qdrant      │
     │  Port: 6379     │    │    │ Port: 6333/6334 │
     │  Hot Tier Cache │    │    │ Vector Storage  │
     └─────────────────┘    │    └─────────────────┘
                            │
                    ┌───────▼───────┐
                    │  Persistent   │
                    │    Storage    │
                    └───────────────┘
```

## License

MIT License - See LICENSE file for details.
