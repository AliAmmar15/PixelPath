"""Stage 2: NumPy vectorization.

The per-pixel Python loops disappear. Grayscale becomes one matrix-vector
product per pixel ((H*W) x 3 times 3). The blur loops only over the k*k kernel
taps; each tap is a whole-image multiply-add on shifted slices of the padded
image. Every tap reads and writes a full image, so this stage moves far more
memory than the cost model's minimum, which is a point worth making in the report.
"""

import numpy as np

from .kernel import (
    GRAY_WEIGHTS_F32,
    gaussian_kernel_1d,
    gaussian_kernel_2d,
    pad_reflect101,
    round_to_uint8,
)


def grayscale(img):
    return round_to_uint8(img.astype(np.float32) @ GRAY_WEIGHTS_F32)


def correlate_2d_float(gray, K):
    K = np.asarray(K, dtype=np.float32)
    kh, kw = K.shape
    h, w = gray.shape
    P = pad_reflect101(gray, kh // 2, kw // 2).astype(np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    for dy in range(kh):
        for dx in range(kw):
            out += K[dy, dx] * P[dy:dy + h, dx:dx + w]
    return out


def separable_float(gray, kcol, krow):
    kcol = np.asarray(kcol, dtype=np.float32)
    krow = np.asarray(krow, dtype=np.float32)
    ry, rx = len(kcol) // 2, len(krow) // 2
    h, w = gray.shape
    P = pad_reflect101(gray, ry, rx).astype(np.float32)
    tmp = np.zeros((h + 2 * ry, w), dtype=np.float32)
    for dx in range(len(krow)):
        tmp += krow[dx] * P[:, dx:dx + w]
    out = np.zeros((h, w), dtype=np.float32)
    for dy in range(len(kcol)):
        out += kcol[dy] * tmp[dy:dy + h, :]
    return out


def blur_2d(gray, ksize, sigma=None):
    return round_to_uint8(correlate_2d_float(gray, gaussian_kernel_2d(ksize, sigma)))


def blur_separable(gray, ksize, sigma=None):
    g = gaussian_kernel_1d(ksize, sigma)
    return round_to_uint8(separable_float(gray, g, g))
