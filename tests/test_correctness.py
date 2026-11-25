"""Correctness tests. Every stage must pass these before it is merged.

Pass condition: max absolute error <= 1 grey level against both the serial
baseline and OpenCV (rounding differences between float32/float64 and
OpenCV's fixed-point arithmetic can move a pixel by one level).
"""

import numpy as np
import pytest

from filters import BLUR_STAGES, GRAY_STAGES, lowrank, parallel, reference, serial, vectorized
from filters.kernel import gaussian_kernel_1d, gaussian_kernel_2d, pad_reflect101
from filters.metrics import max_abs_error, rel_frobenius_error

TOL = 1.0
# Odd, non-square sizes catch off-by-one and row/column mix-ups.
SIZES = [(37, 53), (64, 48), (17, 17)]


def rgb(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def gray_img(h, w, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


# ------------------------------------------------------------------ kernel

@pytest.mark.parametrize("ksize", [1, 3, 5, 9, 15])
def test_kernel_sums_to_one(ksize):
    assert abs(float(gaussian_kernel_1d(ksize).sum()) - 1.0) < 1e-6
    assert abs(float(gaussian_kernel_2d(ksize).sum()) - 1.0) < 1e-5


@pytest.mark.parametrize("ksize", [3, 9, 15])
def test_gaussian_kernel_is_rank_one(ksize):
    """Shaska & Shaska Sec. 8.4: a Gaussian kernel has exactly one nonzero
    singular value, which is why the separable blur is exact."""
    assert lowrank.numerical_rank(gaussian_kernel_2d(ksize)) == 1


def test_even_ksize_rejected():
    with pytest.raises(ValueError):
        gaussian_kernel_1d(4)


def test_reflect101_matches_definition():
    a = np.arange(5, dtype=np.uint8).reshape(1, 5)
    padded = pad_reflect101(np.repeat(a, 3, axis=0), 1, 2)
    assert padded[1].tolist() == [2, 1, 0, 1, 2, 3, 4, 3, 2]


def test_too_small_image_rejected():
    with pytest.raises(ValueError):
        pad_reflect101(np.zeros((3, 3), np.uint8), 3, 3)


# ------------------------------------------------------------------ grayscale

@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("stage", list(GRAY_STAGES))
def test_grayscale_matches_serial_and_opencv(stage, h, w):
    img = rgb(h, w)
    out = GRAY_STAGES[stage](img)
    assert out.shape == (h, w) and out.dtype == np.uint8
    assert max_abs_error(out, serial.grayscale(img)) <= TOL
    assert max_abs_error(out, reference.grayscale(img)) <= TOL


def test_grayscale_split_variants_identical():
    img = rgb(41, 29)
    assert np.array_equal(parallel.grayscale(img, split="rows"),
                          parallel.grayscale(img, split="pixels"))


# ------------------------------------------------------------------ blur

@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("ksize", [3, 5, 9])
@pytest.mark.parametrize("stage", list(BLUR_STAGES))
def test_blur_matches_serial_and_opencv(stage, ksize, h, w):
    g = gray_img(h, w)
    fn, _ = BLUR_STAGES[stage]
    out = fn(g, ksize)
    assert out.shape == (h, w) and out.dtype == np.uint8
    assert max_abs_error(out, serial.blur_2d(g, ksize)) <= TOL
    assert max_abs_error(out, reference.blur_2d(g, ksize)) <= TOL


def test_blur_large_kernel_vs_opencv():
    g = gray_img(120, 90)
    for name in ["numpy_2d", "numpy_sep", "jit_2d", "jit_sep", "parallel_2d", "parallel_sep"]:
        out = BLUR_STAGES[name][0](g, 15)
        assert max_abs_error(out, reference.blur_2d(g, 15)) <= TOL, name


def test_blur_2d_split_variants_identical():
    g = gray_img(33, 61)
    assert np.array_equal(parallel.blur_2d(g, 5, split="rows"),
                          parallel.blur_2d(g, 5, split="pixels"))


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_parallel_result_independent_of_thread_count(threads):
    g = gray_img(97, 83)
    img = rgb(97, 83)
    prev = parallel.get_threads()
    try:
        parallel.set_threads(min(threads, parallel.max_threads()))
        b = parallel.blur_2d(g, 9)
        s = parallel.blur_separable(g, 9)
        gr = parallel.grayscale(img)
    finally:
        parallel.set_threads(prev)
    assert np.array_equal(b, parallel.blur_2d(g, 9))
    assert np.array_equal(s, parallel.blur_separable(g, 9))
    assert np.array_equal(gr, parallel.grayscale(img))


def test_constant_image_unchanged():
    g = np.full((30, 40), 123, np.uint8)
    for name, (fn, _) in BLUR_STAGES.items():
        assert np.all(fn(g, 5) == 123), name


# ------------------------------------------------------------------ low-rank (Chapter 8)

def test_lowrank_full_rank_matches_direct():
    g = gray_img(50, 70)
    K = lowrank.log_kernel(9, 1.4)
    assert lowrank.numerical_rank(K) == 2  # LoG = g''(x)g(y) + g(x)g''(y)
    direct = reference.correlate_2d_float(g, K)
    approx = lowrank.correlate_lowrank(g, K, 2, parallel)
    assert np.max(np.abs(approx - direct)) < 1e-2


def test_lowrank_error_decreases_with_rank():
    g = gray_img(60, 60)
    K = lowrank.disk_kernel(5)
    exact = vectorized.correlate_2d_float(g, K)
    errs = [rel_frobenius_error(lowrank.correlate_lowrank(g, K, r, parallel), exact)
            for r in range(1, lowrank.numerical_rank(K) + 1)]
    assert errs[-1] < 1e-5
    assert all(b <= a + 1e-9 for a, b in zip(errs, errs[1:]))


def test_theorem_8_7_operator_norm_error():
    """||K - K_r||_2 equals the (r+1)-th singular value."""
    K = lowrank.disk_kernel(6)
    S = lowrank.kernel_svd(K)[1]
    for r in range(1, 4):
        err = np.linalg.norm(K.astype(np.float64) - lowrank.rank_r_kernel(K, r), 2)
        assert abs(err - S[r]) < 1e-9
