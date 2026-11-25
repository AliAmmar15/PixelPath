"""Stage 3: Numba JIT compilation, single core.

The loop kernels in _kernels.py are compiled to machine code with @njit.
`prange` acts like `range` here, so everything runs on one thread.

The first call to each function includes compile time; benchmarks always do
one warm-up call before timing.
"""

from ._compiled import CompiledStage

_stage = CompiledStage(parallel=False)

grayscale = _stage.grayscale
blur_2d = _stage.blur_2d
blur_separable = _stage.blur_separable
correlate_2d_float = _stage.correlate_2d_float
separable_float = _stage.separable_float
