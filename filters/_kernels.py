"""Loop kernels shared by Stage 3 (JIT) and Stage 4 (parallel).

Each function below is plain Python that uses `prange` for its outer loop.
jit.py compiles them with @njit (prange then behaves like range, one core),
and parallel.py compiles the *same source* with @njit(parallel=True), which
splits the prange loop across threads. Stages 3 and 4 therefore differ only
in threading, so the speedup between them is purely from parallelism.

Inputs are the padded uint8 image P; outputs are written into a preallocated
array. All arithmetic is float32.
"""

import math

import numpy as np
from numba import njit, prange


@njit(inline="always")
def round_u8(v):
    r = math.floor(v + 0.5)
    if r < 0:
        r = 0
    elif r > 255:
        r = 255
    return np.uint8(r)


# ---------------------------------------------------------------- grayscale

def gray_rows(img, w, out):
    """Split work by rows: each thread gets a block of whole rows."""
    h, wd = out.shape
    w0, w1, w2 = w[0], w[1], w[2]
    for y in prange(h):
        for x in range(wd):
            v = (w0 * np.float32(img[y, x, 0])
                 + w1 * np.float32(img[y, x, 1])
                 + w2 * np.float32(img[y, x, 2]))
            out[y, x] = round_u8(v)


def gray_pixels(img, w, out):
    """Split work by flattened pixel index instead of by row."""
    h, wd = out.shape
    w0, w1, w2 = w[0], w[1], w[2]
    for i in prange(h * wd):
        y = i // wd
        x = i - y * wd
        v = (w0 * np.float32(img[y, x, 0])
             + w1 * np.float32(img[y, x, 1])
             + w2 * np.float32(img[y, x, 2]))
        out[y, x] = round_u8(v)


# ---------------------------------------------------------------- direct 2-D correlation

def corr2d_f32(P, K, out):
    h, wd = out.shape
    kh, kw = K.shape
    for y in prange(h):
        for x in range(wd):
            acc = np.float32(0.0)
            for dy in range(kh):
                for dx in range(kw):
                    acc += K[dy, dx] * np.float32(P[y + dy, x + dx])
            out[y, x] = acc


def corr2d_u8(P, K, out):
    h, wd = out.shape
    kh, kw = K.shape
    for y in prange(h):
        for x in range(wd):
            acc = np.float32(0.0)
            for dy in range(kh):
                for dx in range(kw):
                    acc += K[dy, dx] * np.float32(P[y + dy, x + dx])
            out[y, x] = round_u8(acc)


def corr2d_u8_pixels(P, K, out):
    """Same as corr2d_u8, but the parallel loop runs over flattened pixels."""
    h, wd = out.shape
    kh, kw = K.shape
    for i in prange(h * wd):
        y = i // wd
        x = i - y * wd
        acc = np.float32(0.0)
        for dy in range(kh):
            for dx in range(kw):
                acc += K[dy, dx] * np.float32(P[y + dy, x + dx])
        out[y, x] = round_u8(acc)


# ---------------------------------------------------------------- separable passes

def sep_h(P, krow, tmp):
    """Horizontal pass over every padded row: tmp[y, x] = sum krow[dx] * P[y, x+dx]."""
    hp, wd = tmp.shape
    kw = krow.shape[0]
    for y in prange(hp):
        for x in range(wd):
            acc = np.float32(0.0)
            for dx in range(kw):
                acc += krow[dx] * np.float32(P[y, x + dx])
            tmp[y, x] = acc


def sep_v_f32(tmp, kcol, out):
    """Vertical pass. Accumulates a whole output row at a time so the inner
    loop walks memory contiguously (and can use SIMD) instead of striding
    down a column."""
    h, wd = out.shape
    kh = kcol.shape[0]
    for y in prange(h):
        row = np.zeros(wd, dtype=np.float32)
        for dy in range(kh):
            c = kcol[dy]
            for x in range(wd):
                row[x] += c * tmp[y + dy, x]
        for x in range(wd):
            out[y, x] = row[x]


def sep_v_u8(tmp, kcol, out):
    h, wd = out.shape
    kh = kcol.shape[0]
    for y in prange(h):
        row = np.zeros(wd, dtype=np.float32)
        for dy in range(kh):
            c = kcol[dy]
            for x in range(wd):
                row[x] += c * tmp[y + dy, x]
        for x in range(wd):
            out[y, x] = round_u8(row[x])
