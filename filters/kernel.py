"""Shared building blocks used by every stage.

Every implementation (serial, NumPy, JIT, parallel) imports its constants from
here, so the stages differ only in *how* they compute, never in *what*.

Conventions (agreed in Phase 0):
  * Input colour images are uint8 arrays of shape (H, W, 3) in RGB order.
  * Grayscale weights are ITU-R BT.601: 0.299 R + 0.587 G + 0.114 B
    (the same weights OpenCV uses, so results can be compared).
  * Filters are correlations (the kernel is not flipped). Gaussian kernels are
    symmetric, so this is identical to convolution for the blur.
  * Borders use reflect-101 (OpenCV's default): ...c b | a b c d | c b...
  * Arithmetic is float32; results are rounded half-up and clipped to uint8.
"""

import numpy as np

GRAY_WEIGHTS = (0.299, 0.587, 0.114)
GRAY_WEIGHTS_F32 = np.array(GRAY_WEIGHTS, dtype=np.float32)


def default_sigma(ksize):
    """OpenCV's rule for choosing sigma from the kernel size."""
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


def _check_ksize(ksize):
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}")


def gaussian_kernel_1d(ksize, sigma=None):
    """Normalised 1-D Gaussian of length ksize (float32, sums to 1)."""
    _check_ksize(ksize)
    if sigma is None or sigma <= 0:
        sigma = default_sigma(ksize)
    x = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g /= g.sum()
    return g.astype(np.float32)


def gaussian_kernel_2d(ksize, sigma=None):
    """2-D Gaussian kernel K = g g^T.

    Because K is an outer product it has rank 1 (one nonzero singular value,
    Shaska & Shaska, Sec. 8.4). That is exactly why the separable blur, which
    applies g along rows and then along columns, gives the same answer.
    """
    g = gaussian_kernel_1d(ksize, sigma).astype(np.float64)
    return np.outer(g, g).astype(np.float32)


def pad_reflect101(img, ry, rx):
    """Pad a 2-D array by ry rows and rx columns using reflect-101 borders.

    NumPy's mode="reflect" does not repeat the edge pixel, which is exactly
    OpenCV's BORDER_REFLECT_101.
    """
    h, w = img.shape[:2]
    if h <= ry or w <= rx:
        raise ValueError(
            f"image {h}x{w} is too small for a kernel radius of ({ry}, {rx}); "
            "reflect-101 padding needs each side to be larger than the radius"
        )
    return np.pad(img, ((ry, ry), (rx, rx)), mode="reflect")


def round_to_uint8(x):
    """Round half up and clip to [0, 255]. Used by the NumPy stage."""
    return np.clip(np.floor(x + 0.5), 0, 255).astype(np.uint8)
