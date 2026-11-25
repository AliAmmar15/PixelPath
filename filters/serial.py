"""Stage 1: pure-Python serial baseline.

Pixels are processed one at a time with ordinary Python loops over Python
lists (the array is converted with .tolist() first, which is much faster than
indexing a NumPy array element by element and is what "plain Python" means).
Padding uses the shared helper so the border rule matches every other stage.

This stage is very slow on purpose: it is the baseline every speedup is
measured against. Large images are timed on a strip of rows and scaled up
(see bench/run_benchmarks.py).
"""

import math

import numpy as np

from .kernel import GRAY_WEIGHTS, gaussian_kernel_1d, gaussian_kernel_2d, pad_reflect101


def _to_u8(v):
    r = math.floor(v + 0.5)
    if r < 0:
        return 0
    if r > 255:
        return 255
    return r


def grayscale(img):
    wr, wg, wb = GRAY_WEIGHTS
    out = []
    for row in img.tolist():
        orow = []
        for r, g, b in row:
            orow.append(_to_u8(wr * r + wg * g + wb * b))
        out.append(orow)
    return np.array(out, dtype=np.uint8)


def correlate_2d_float(gray, K):
    """Direct 2-D correlation, k*k multiply-adds per pixel. Returns float64."""
    K = np.asarray(K, dtype=np.float64).tolist()
    kh, kw = len(K), len(K[0])
    h, w = gray.shape
    P = pad_reflect101(gray, kh // 2, kw // 2).tolist()
    out = []
    for y in range(h):
        orow = []
        for x in range(w):
            acc = 0.0
            for dy in range(kh):
                prow = P[y + dy]
                krow = K[dy]
                for dx in range(kw):
                    acc += krow[dx] * prow[x + dx]
            orow.append(acc)
        out.append(orow)
    return np.array(out, dtype=np.float64)


def separable_float(gray, kcol, krow):
    """Row pass with krow, then column pass with kcol. Returns float64."""
    kcol = [float(v) for v in kcol]
    krow = [float(v) for v in krow]
    ry, rx = len(kcol) // 2, len(krow) // 2
    h, w = gray.shape
    P = pad_reflect101(gray, ry, rx).tolist()

    tmp = []  # (h + 2*ry) x w, horizontal pass over every padded row
    for prow in P:
        trow = []
        for x in range(w):
            acc = 0.0
            for dx in range(len(krow)):
                acc += krow[dx] * prow[x + dx]
            trow.append(acc)
        tmp.append(trow)

    out = []
    for y in range(h):
        orow = []
        for x in range(w):
            acc = 0.0
            for dy in range(len(kcol)):
                acc += kcol[dy] * tmp[y + dy][x]
            orow.append(acc)
        out.append(orow)
    return np.array(out, dtype=np.float64)


def _round_array(f):
    return np.array([[_to_u8(v) for v in row] for row in f.tolist()], dtype=np.uint8)


def blur_2d(gray, ksize, sigma=None):
    return _round_array(correlate_2d_float(gray, gaussian_kernel_2d(ksize, sigma)))


def blur_separable(gray, ksize, sigma=None):
    g = gaussian_kernel_1d(ksize, sigma)
    return _round_array(separable_float(gray, g, g))
