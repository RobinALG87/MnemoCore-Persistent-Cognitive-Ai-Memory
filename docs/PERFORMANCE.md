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

| Operation | Time (μs) | Notes |
|-----------|-----------|-------|
| `xor_bind()` | ~5 | XOR binding of two vectors |
| `permute()` | ~5 | Cyclic permutation |
| `hamming_distance()` | ~3 | Distance calculation |
| `similarity()` | ~4 | Normalized similarity |

### permute() Benchmark Results

| Dimension | Unpackbits (μs) | Bitwise (μs) | Faster |
|-----------|-----------------|--------------|--------|
| 512 | 5.17 | 8.10 | unpackbits |
| 4096 | 5.54 | 9.36 | unpackbits |
| 16384 | 6.79 | 12.83 | unpackbits |
| 32768 | 8.21 | 16.22 | unpackbits |
| 65536 | 11.28 | 24.94 | unpackbits |
| 131072 | 17.71 | 43.48 | unpackbits |

**Recommendation:** Use `unpackbits` implementation for all dimensions (56-145% faster).

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

1. **Use BinaryHDV instead of float HDV** - 10x faster operations
2. **Batch operations** - Use batch methods for bulk inserts
3. **Connection pooling** - Ensure Redis connection pool is sized appropriately
4. **Qdrant quantization** - Enable binary quantization for faster search

## Monitoring

Prometheus metrics are exposed at `/metrics` endpoint:
- `mnemocore_store_duration_seconds` - Store operation latency
- `mnemocore_query_duration_seconds` - Query operation latency
- `mnemocore_memory_count_total` - Total memories per tier
- `mnemocore_queue_length` - Subconscious queue length
