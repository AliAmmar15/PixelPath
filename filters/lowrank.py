"""Low-rank (SVD) kernel decomposition, from Shaska & Shaska, Sec. 8.4.

Any k x k kernel can be written as a sum of rank-one terms (Eq. 8.1):

    K = sum_i sigma_i u_i v_i^T

Each rank-one term sigma_i u_i v_i^T is a *separable* filter: a column pass
with sqrt(sigma_i) u_i and a row pass with sqrt(sigma_i) v_i. So a kernel of
rank r can be applied as r separable passes added together, costing about
4kr FLOP per pixel instead of 2k^2.

  * Gaussian kernels have rank 1, so one separable pass is exact.
  * Other kernels (e.g. a disk / bokeh blur) have higher rank. Keeping only
    the first r terms is the best rank-r approximation (Theorem 8.7), with
    operator-norm error ||K - K_r|| = sigma_{r+1}.
"""

import numpy as np


def kernel_svd(K):
    U, S, Vt = np.linalg.svd(np.asarray(K, dtype=np.float64))
    return U, S, Vt


def numerical_rank(K, rtol=1e-6):
    S = kernel_svd(K)[1]
    return int(np.sum(S > rtol * S[0]))


def rank_r_terms(K, r):
    """Return r (kcol, krow) pairs such that sum_i outer(kcol_i, krow_i) = K_r."""
    U, S, Vt = kernel_svd(K)
    terms = []
    for i in range(r):
        s = np.sqrt(S[i])
        terms.append(((s * U[:, i]).astype(np.float32), (s * Vt[i, :]).astype(np.float32)))
    return terms


def rank_r_kernel(K, r):
    U, S, Vt = kernel_svd(K)
    return (U[:, :r] * S[:r]) @ Vt[:r, :]


def correlate_lowrank(gray, K, r, stage):
    """Apply the best rank-r approximation of K as r separable passes.

    `stage` is any stage module with separable_float (e.g. filters.parallel).
    Returns float32.
    """
    acc = None
    for kcol, krow in rank_r_terms(K, r):
        part = stage.separable_float(gray, kcol, krow)
        acc = part if acc is None else acc + part
    return acc


def disk_kernel(radius):
    """Normalised circular (bokeh-style) averaging kernel. Not separable."""
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    K = (x * x + y * y <= radius * radius).astype(np.float64)
    return (K / K.sum()).astype(np.float32)


def log_kernel(ksize, sigma):
    """Laplacian-of-Gaussian kernel (edge detection), up to scale.

    (q - 1) e^{-q} with q = (x^2 + y^2) / (2 sigma^2) equals
    (x^2/2s^2 - 1) g(x) g(y) + g(x) (y^2/2s^2) g(y): a sum of two outer
    products, so it has rank 2. (Subtracting the mean, a common trick to make
    it sum to zero, would add a constant rank-one matrix and give rank 3.)
    """
    r = ksize // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    q = (x * x + y * y) / (2 * sigma * sigma)
    K = (q - 1) * np.exp(-q)
    return K.astype(np.float32)
