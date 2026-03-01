"""
Benchmark BinaryHDV.permute() using the production implementation.
"""

import sys
import timeit
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path so 'mnemocore' is importable
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from mnemocore.core.binary_hdv import BinaryHDV


def permute_reference(data: np.ndarray, shift: int) -> np.ndarray:
    bits = np.unpackbits(data)
    bits = np.roll(bits, shift)
    return np.packbits(bits)


def benchmark_dimension(dimension: int, shift: int = 13) -> Dict[str, float]:
    hdv = BinaryHDV.random(dimension)

    # Correctness check against golden reference
    expected = permute_reference(hdv.data, shift)
    actual = hdv.permute(shift).data
    assert np.array_equal(actual, expected), "permute() mismatch vs reference"

    t = min(
        timeit.repeat(
            stmt="hdv.permute(shift)",
            globals={"hdv": hdv, "shift": shift},
            repeat=5,
            number=500,
        )
    )
    us = (t / 500) * 1_000_000
    return {"dimension": float(dimension), "permute_us": us}


def main() -> None:
    dimensions: List[int] = [512, 4096, 16384, 32768, 65536, 131072]
    print("BinaryHDV.permute() benchmark (production path)")
    print(f"{'Dimension':>10} | {'permute(us)':>12}")
    print("-" * 27)
    for dim in dimensions:
        result = benchmark_dimension(dim)
        print(f"{int(result['dimension']):>10} | {result['permute_us']:>12.2f}")


if __name__ == "__main__":
    main()
