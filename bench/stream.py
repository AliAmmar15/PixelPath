"""STREAM-style memory bandwidth test (copy and triad).

Measures the bandwidth this machine actually delivers, which is the real
ceiling for memory-bound kernels such as grayscale. Arrays are far larger
than the L3 cache so the data must come from DRAM.

Byte counting follows STREAM: copy moves 2 arrays, triad moves 3
(write-allocate traffic is not counted, so true traffic is somewhat higher).

Run:  python -m bench.stream
"""

import numba
import numpy as np
from numba import njit, prange

from bench.common import RESULTS, ensure_dirs, save_json

GIB = 1024 ** 3
N = 2 ** 26  # 64 Mi float32 = 256 MiB per array


@njit(parallel=True)
def copy(a, c):
    for i in prange(a.shape[0]):
        c[i] = a[i]


@njit(parallel=True)
def triad(a, b, c, s):
    for i in prange(a.shape[0]):
        a[i] = b[i] + s * c[i]


def best_time(fn, reps=10):
    import time
    fn()  # warm-up / compile
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ensure_dirs()
    a = np.ones(N, np.float32)
    b = np.full(N, 2.0, np.float32)
    c = np.full(N, 0.5, np.float32)
    nbytes = N * 4
    results = {"array_mib": nbytes / 1024 ** 2, "threads": {}}

    thread_counts = sorted({1, 2, 4, numba.config.NUMBA_NUM_THREADS // 2, numba.config.NUMBA_NUM_THREADS})
    for t in thread_counts:
        numba.set_num_threads(t)
        tc = best_time(lambda: copy(a, c))
        tt = best_time(lambda: triad(a, b, c, np.float32(3.0)))
        results["threads"][str(t)] = {
            "copy_gibs": 2 * nbytes / tc / GIB,
            "triad_gibs": 3 * nbytes / tt / GIB,
        }
        print(f"threads={t:2d}  copy {2 * nbytes / tc / GIB:6.2f} GiB/s   triad {3 * nbytes / tt / GIB:6.2f} GiB/s")
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    tn = best_time(lambda: np.copyto(c, a))
    results["numpy_copy_gibs"] = 2 * nbytes / tn / GIB
    print(f"numpy copy (1 thread)  {results['numpy_copy_gibs']:6.2f} GiB/s")

    results["best_gibs"] = max(max(v.values()) for v in results["threads"].values())
    print(f"best measured          {results['best_gibs']:6.2f} GiB/s")
    save_json(RESULTS / "stream.json", results)


if __name__ == "__main__":
    main()
