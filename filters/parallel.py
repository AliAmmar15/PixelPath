"""Stage 4: Numba parallel execution across CPU cores.

Same loop kernels as Stage 3, compiled with @njit(parallel=True), so the
outer `prange` loop is split across threads.

Controls for the experiments:
  * set_threads(n)    number of worker threads (numba.set_num_threads)
  * set_chunksize(n)  scheduling granularity (numba.set_parallel_chunksize).
                      0 = Numba default (static: iterations divided evenly
                      between threads up front). n > 0 = iterations are handed
                      out in chunks of n, so idle threads can take more work
                      (dynamic-style load balancing, at some overhead).
  * split="rows" | "pixels" on grayscale/blur_2d chooses what the parallel loop runs over.
"""

import numba

from ._compiled import CompiledStage

_stage = CompiledStage(parallel=True)

grayscale = _stage.grayscale
blur_2d = _stage.blur_2d
blur_separable = _stage.blur_separable
correlate_2d_float = _stage.correlate_2d_float
separable_float = _stage.separable_float


def max_threads():
    return numba.config.NUMBA_NUM_THREADS


def set_threads(n):
    numba.set_num_threads(int(n))


def get_threads():
    return numba.get_num_threads()


def set_chunksize(n):
    """Returns the previous chunk size. Needs Numba >= 0.57."""
    return numba.set_parallel_chunksize(int(n))
