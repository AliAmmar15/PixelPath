"""Builds the public API (grayscale, blur_2d, ...) around compiled loop kernels.

jit.py and parallel.py both call `build(parallel=...)`; the only difference is
the Numba `parallel` flag.
"""

import numpy as np
from numba import njit

from . import _kernels as k
from .kernel import GRAY_WEIGHTS_F32, gaussian_kernel_1d, gaussian_kernel_2d, pad_reflect101


class CompiledStage:
    def __init__(self, parallel):
        jit = njit(parallel=parallel, fastmath=False)
        self.parallel = parallel
        self._gray_rows = jit(k.gray_rows)
        self._gray_pixels = jit(k.gray_pixels)
        self._corr2d_f32 = jit(k.corr2d_f32)
        self._corr2d_u8 = jit(k.corr2d_u8)
        self._corr2d_u8_pixels = jit(k.corr2d_u8_pixels)
        self._sep_h = jit(k.sep_h)
        self._sep_v_f32 = jit(k.sep_v_f32)
        self._sep_v_u8 = jit(k.sep_v_u8)

    # ---- grayscale
    def grayscale(self, img, split="rows"):
        img = np.ascontiguousarray(img)
        out = np.empty(img.shape[:2], dtype=np.uint8)
        fn = self._gray_rows if split == "rows" else self._gray_pixels
        fn(img, GRAY_WEIGHTS_F32, out)
        return out

    # ---- direct 2-D
    def correlate_2d_float(self, gray, K):
        K = np.ascontiguousarray(K, dtype=np.float32)
        P = pad_reflect101(gray, K.shape[0] // 2, K.shape[1] // 2)
        out = np.empty(gray.shape, dtype=np.float32)
        self._corr2d_f32(P, K, out)
        return out

    def blur_2d(self, gray, ksize, sigma=None, split="rows"):
        K = gaussian_kernel_2d(ksize, sigma)
        r = ksize // 2
        P = pad_reflect101(gray, r, r)
        out = np.empty(gray.shape, dtype=np.uint8)
        fn = self._corr2d_u8 if split == "rows" else self._corr2d_u8_pixels
        fn(P, K, out)
        return out

    # ---- separable
    def _horizontal(self, gray, kcol, krow):
        kcol = np.ascontiguousarray(kcol, dtype=np.float32)
        krow = np.ascontiguousarray(krow, dtype=np.float32)
        ry, rx = kcol.shape[0] // 2, krow.shape[0] // 2
        P = pad_reflect101(gray, ry, rx)
        tmp = np.empty((gray.shape[0] + 2 * ry, gray.shape[1]), dtype=np.float32)
        self._sep_h(P, krow, tmp)
        return tmp, kcol

    def separable_float(self, gray, kcol, krow):
        tmp, kcol = self._horizontal(gray, kcol, krow)
        out = np.empty(gray.shape, dtype=np.float32)
        self._sep_v_f32(tmp, kcol, out)
        return out

    def blur_separable(self, gray, ksize, sigma=None):
        g = gaussian_kernel_1d(ksize, sigma)
        tmp, g = self._horizontal(gray, g, g)
        out = np.empty(gray.shape, dtype=np.uint8)
        self._sep_v_u8(tmp, g, out)
        return out
