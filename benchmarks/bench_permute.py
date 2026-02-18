"""
Benchmark for BinaryHDV.permute() operation.

Compares two implementation strategies:
1. unpackbits/roll/packbits (currently used for dim < 32768)
2. bitwise byte/bit shift (currently used for dim >= 32768)

Usage:
    .venv\Scripts\python.exe benchmarks/bench_permute.py
"""

import timeit
from typing import Callable, Tuple

import numpy as np

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.binary_hdv import BinaryHDV


def permute_unpackbits(data: np.ndarray, dimension: int, shift: int) -> np.ndarray:
    """
    Permute using unpackbits/roll/packbits approach.
    Faster for smaller dimensions due to NumPy optimization of these ops.
    """
    if shift == 0:
        return data.copy()

    shift = shift % dimension
    bits = np.unpackbits(data)
    bits = np.roll(bits, shift)
    return np.packbits(bits)


def permute_bitwise(data: np.ndarray, dimension: int, shift: int) -> np.ndarray:
    """
    Permute using bitwise byte/bit shift approach.
    Faster for larger dimensions as it avoids unpacking overhead.
    """
    if shift == 0:
        return data.copy()

    shift = shift % dimension
    byte_shift = shift // 8
    bit_shift = shift % 8

    # 1. Byte level shift (circular)
    result = np.roll(data, byte_shift)

    # 2. Bit level shift (if needed)
    if bit_shift > 0:
        r_shift = 8 - bit_shift

        # Calculate bits that wrap from previous byte
        low_part = np.empty_like(result)
        low_part[0] = result[-1] << r_shift
        low_part[1:] = result[:-1] << r_shift

        # Shift current byte bits right
        result >>= bit_shift
        # Combine
        result |= low_part

    return result


def benchmark_dimension(dimension: int, n_iterations: int = 100, n_warmup: int = 10) -> dict:
    """
    Benchmark both permute implementations for a given dimension.

    Returns dict with timing results in microseconds.
    """
    # Create test vector
    hdv = BinaryHDV.random(dimension)
    shift = 1

    # Warmup
    for _ in range(n_warmup):
        permute_unpackbits(hdv.data, dimension, shift)
        permute_bitwise(hdv.data, dimension, shift)

    # Create globals dict for timeit
    globs = {
        'permute_unpackbits': permute_unpackbits,
        'permute_bitwise': permute_bitwise,
        'hdv': hdv,
        'dimension': dimension,
        'shift': shift,
    }

    # Benchmark unpackbits approach
    unpackbits_times = timeit.repeat(
        stmt='permute_unpackbits(hdv.data, dimension, shift)',
        globals=globs,
        number=n_iterations,
        repeat=5
    )

    # Benchmark bitwise approach
    bitwise_times = timeit.repeat(
        stmt='permute_bitwise(hdv.data, dimension, shift)',
        globals=globs,
        number=n_iterations,
        repeat=5
    )

    # Calculate statistics (take minimum to reduce noise)
    unpackbits_best = min(unpackbits_times) / n_iterations * 1_000_000  # microseconds
    bitwise_best = min(bitwise_times) / n_iterations * 1_000_000  # microseconds

    # Verify both implementations produce same result
    result_unpackbits = permute_unpackbits(hdv.data, dimension, shift)
    result_bitwise = permute_bitwise(hdv.data, dimension, shift)
    assert np.array_equal(result_unpackbits, result_bitwise), \
        f"Implementation mismatch at dimension {dimension}"

    return {
        'dimension': dimension,
        'unpackbits_us': unpackbits_best,
        'bitwise_us': bitwise_best,
        'faster': 'unpackbits' if unpackbits_best < bitwise_best else 'bitwise',
        'ratio': max(unpackbits_best, bitwise_best) / min(unpackbits_best, bitwise_best)
    }


def main():
    """Run benchmark across specified dimensions and output results."""
    dimensions = [512, 4096, 16384, 32768, 65536, 131072]

    print("=" * 80)
    print("BinaryHDV.permute() Performance Benchmark")
    print("=" * 80)
    print()
    print(f"{'Dimension':>10} | {'Unpackbits (us)':>16} | {'Bitwise (us)':>14} | {'Faster':>10} | {'Ratio':>6}")
    print("-" * 80)

    results = []
    for dim in dimensions:
        # Use more iterations for smaller dimensions for better precision
        n_iter = 1000 if dim <= 4096 else (500 if dim <= 16384 else 100)
        result = benchmark_dimension(dim, n_iterations=n_iter)
        results.append(result)

        print(f"{result['dimension']:>10} | {result['unpackbits_us']:>14.2f} | {result['bitwise_us']:>12.2f} | {result['faster']:>10} | {result['ratio']:>6.2f}x")

    print("-" * 80)
    print()

    # Analysis
    print("Analysis:")
    print("-" * 40)

    # Find crossover point
    crossover_dim = None
    for i, result in enumerate(results):
        if result['faster'] == 'bitwise':
            crossover_dim = result['dimension']
            break

    if crossover_dim:
        print(f"- Crossover point: bitwise becomes faster at dimension >= {crossover_dim}")
    else:
        print("- Unpackbits is faster across all tested dimensions")

    # Calculate percentage differences
    print("\nPercentage difference (negative = unpackbits faster):")
    for result in results:
        diff_pct = ((result['bitwise_us'] - result['unpackbits_us']) / result['unpackbits_us']) * 100
        print(f"  dim={result['dimension']:>6}: {diff_pct:+.1f}%")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("-" * 80)

    # Check if difference is <5% at current threshold (32768)
    threshold_result = next((r for r in results if r['dimension'] == 32768), None)
    if threshold_result:
        diff_at_threshold = abs(threshold_result['bitwise_us'] - threshold_result['unpackbits_us'])
        pct_diff_at_threshold = (diff_at_threshold / min(threshold_result['bitwise_us'], threshold_result['unpackbits_us'])) * 100

        if pct_diff_at_threshold < 5:
            # Choose one implementation
            faster = threshold_result['faster']
            print(f"Difference at current threshold (32768) is {pct_diff_at_threshold:.1f}% (< 5%)")
            print(f"RECOMMENDATION: Use single implementation ({faster}) for all dimensions")
        else:
            print(f"Difference at current threshold (32768) is {pct_diff_at_threshold:.1f}% (>= 5%)")
            print(f"RECOMMENDATION: Keep dual implementation with threshold at 32768")
            print(f"  - Use unpackbits for dimension < 32768")
            print(f"  - Use bitwise for dimension >= 32768")

    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
