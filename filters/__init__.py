"""Image filters at each stage of the optimization path.

    serial      Stage 1  pure Python loops
    vectorized  Stage 2  NumPy whole-array operations
    jit         Stage 3  Numba @njit, single core
    parallel    Stage 4  Numba @njit(parallel=True), multi-core
    reference   OpenCV, for correctness and as the performance target

The registries below map a stage name to a callable so the tests and
benchmarks can loop over every stage the same way.
"""

from . import jit, parallel, reference, serial, vectorized

GRAY_STAGES = {
    "serial": serial.grayscale,
    "numpy": vectorized.grayscale,
    "jit": jit.grayscale,
    "parallel": parallel.grayscale,
    "opencv": reference.grayscale,
}

# name -> (function(gray, ksize, sigma), variant used by the cost model)
BLUR_STAGES = {
    "serial_2d": (serial.blur_2d, "direct"),
    "serial_sep": (serial.blur_separable, "separable"),
    "numpy_2d": (vectorized.blur_2d, "direct"),
    "numpy_sep": (vectorized.blur_separable, "separable"),
    "jit_2d": (jit.blur_2d, "direct"),
    "jit_sep": (jit.blur_separable, "separable"),
    "parallel_2d": (parallel.blur_2d, "direct"),
    "parallel_sep": (parallel.blur_separable, "separable"),
    "opencv": (reference.blur_2d, "separable"),  # OpenCV uses a separable filter for Gaussians
}

PARALLEL_STAGES = {"parallel", "parallel_2d", "parallel_sep", "opencv"}
