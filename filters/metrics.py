"""Correctness metrics."""

import numpy as np


def max_abs_error(a, b):
    """Largest per-pixel difference (in grey levels for uint8 images)."""
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def rel_frobenius_error(a, ref):
    """||a - ref||_F / ||ref||_F, the image error measure used in
    Shaska & Shaska, Fig. 8.3."""
    a = a.astype(np.float64)
    ref = ref.astype(np.float64)
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(a - ref) / denom) if denom else 0.0


def psnr(a, ref, peak=255.0):
    mse = np.mean((a.astype(np.float64) - ref.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(peak * peak / mse))
