"""Helpers shared by the benchmark scripts: paths, test images, timing."""

import json
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
IMAGES = ROOT / "images"

SIZES = {
    "256": (256, 256),
    "512": (512, 512),
    "1080p": (1080, 1920),
    "4K": (2160, 3840),
    "8K": (4320, 7680),
}


def ensure_dirs():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)


def synthetic_rgb(h, w, seed=0):
    """Deterministic photo-like test image: smooth gradients, edges, texture, noise.

    A mix of flat and busy regions so the filters do realistic work.
    """
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    img = np.empty((h, w, 3), np.float32)
    img[..., 0] = 200 * x + 30 * np.sin(12 * np.pi * y)
    img[..., 1] = 180 * y + 40 * np.cos(9 * np.pi * x * y)
    img[..., 2] = 120 + 90 * np.sin(5 * np.pi * (x + y))
    for _ in range(12):  # hard-edged rectangles
        y0, x0 = rng.integers(0, h), rng.integers(0, w)
        y1, x1 = min(h, y0 + rng.integers(h // 20 + 1, h // 4 + 2)), min(w, x0 + rng.integers(w // 20 + 1, w // 4 + 2))
        img[y0:y1, x0:x1] = rng.integers(0, 256, 3)
    img += rng.normal(0, 12, (h, w, 3)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def load_image(h, w):
    """Use images/source.(jpg|png) resized to (w, h) if present, else a synthetic image."""
    for ext in ("jpg", "jpeg", "png"):
        src = IMAGES / f"source.{ext}"
        if src.exists():
            from PIL import Image
            with Image.open(src) as im:
                return np.asarray(im.convert("RGB").resize((w, h), Image.LANCZOS))
    return synthetic_rgb(h, w)


def time_call(fn, repeats=5, warmup=1):
    """Run fn warmup times untimed, then `repeats` timed runs.
    Returns (median seconds, min seconds, last output)."""
    out = None
    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), out


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))
