# MnemoCore Performance Documentation

## Performance Targets (SLOs)

| Metric | Target | Description |
|--------|--------|-------------|
| `store()` P99 latency | < 100ms | Store a single memory |
| `query()` P99 latency | < 50ms | Query for similar memories |
| Throughput | > 1000 req/s | Sustained request rate |
| Memory overhead | < 100MB per 100k memories | RAM usage for storage |

## Baseline Measurements

### BinaryHDV Operations (1024 dimensions)

| Operation | Time (us) | Notes |
|-----------|-----------|-------|
| `xor_bind()` | ~5 | XOR binding of two vectors |
| `permute()` | ~5 | Cyclic permutation |
| `hamming_distance()` | ~3 | Distance calculation |
| `similarity()` | ~4 | Normalized similarity |

### permute() Benchmark Results

`BinaryHDV.permute()` now uses one production path (`unpackbits` + `roll` + `packbits`) across all dimensions.

| Dimension | permute() (us) | Notes |
|-----------|----------------|-------|
| 512 | ~5.2 | Production path |
| 4096 | ~5.5 | Production path |
| 16384 | ~6.8 | Production path |
| 32768 | ~8.2 | Production path |
| 65536 | ~11.3 | Production path |
| 131072 | ~17.7 | Production path |

Run `python benchmarks/bench_permute.py` for machine-specific current numbers.

## Load Testing

### Using Locust

```bash
# Install locust
pip install locust

# Run load test
cd tests/load
locust -f locustfile.py --host http://localhost:8100
```

### Using the Benchmark Script

```bash
# Run 100k memory benchmark
python benchmarks/bench_100k_memories.py
```

## Performance Optimization Tips

1. Use BinaryHDV instead of float HDV.
2. Use batch operations for bulk work.
3. Keep Redis connection pools right-sized.
4. Enable Qdrant binary quantization for faster search.

## Monitoring

Prometheus metrics are exposed at `/metrics` endpoint:
- `mnemocore_store_duration_seconds` - Store operation latency
- `mnemocore_query_duration_seconds` - Query operation latency
- `mnemocore_memory_count_total` - Total memories per tier
- `mnemocore_queue_length` - Subconscious queue length
