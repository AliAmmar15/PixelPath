# PixelPath: Parallel Image Filtering from Python to Multicore

CSI 4650 Parallel and Distributed Computing, course project.
Matthew Mankiewicz, Dalya Karem, Ali Ammar.

Two image filters, grayscale conversion and Gaussian blur, taken along an
optimization path. Each stage is checked against the previous one and against
OpenCV, and timed:

| Stage | Module | What changes |
|---|---|---|
| 1. Serial | `filters/serial.py` | Pure Python loops over pixels (the baseline) |
| 2. NumPy | `filters/vectorized.py` | Whole-array operations, no per-pixel Python |
| 3. Numba JIT | `filters/jit.py` | Loops compiled to machine code, one core |
| 4. Numba parallel | `filters/parallel.py` | Same loops split across cores with `prange` |
| Reference | `filters/reference.py` | OpenCV, for correctness and as the speed target |

The blur has a **direct 2-D** version (k² multiply-adds per pixel) and a
**separable** version (two 1-D passes, 2k per pixel). The separable blur is exact because the
Gaussian kernel has rank 1 (Shaska & Shaska, *Linear Algebra with Applications
in AI*, §8.4); `filters/lowrank.py` extends this to non-separable kernels
using a truncated SVD.

## Setup

Python 3.12 (Windows, macOS or Linux).

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Running

```bash
python run_all.py --quick
```

That runs the tests and a small benchmark sweep in about 3 minutes. For the final report:

```bash
python run_all.py
```

Individual steps:

```bash
python -m pytest -q                      # correctness tests
python -m bench.hardware                 # theoretical peaks from bench/hardware.json
python -m bench.stream                   # measured memory bandwidth
python -m bench.run_benchmarks --quick   # main timing sweep -> results/results.csv
python -m bench.workload                 # rows vs pixels, chunk size -> results/workload.csv
python -m bench.lowrank_bench            # Chapter 8 experiment -> results/lowrank.csv
python -m bench.plots                    # figures -> results/figures/
```

To use a real photo instead of the synthetic test image, put it at
`images/source.jpg`; it is resized to each test size.

**Before the final run:** edit `bench/hardware.json` for the machine you
benchmark on (CPU, cores, clock, memory speed and channels). The values in it
now are for an AMD Ryzen 5 5500 with dual-channel DDR4-3200.

## Layout

```
filters/
  kernel.py        shared rules: Gaussian kernel, reflect-101 padding, rounding
  serial.py        Stage 1
  vectorized.py    Stage 2
  _kernels.py      loop kernels shared by Stages 3 and 4 (same source, different compile flags)
  _compiled.py     wraps the kernels into the public API
  jit.py           Stage 3
  parallel.py      Stage 4 (+ thread count and chunk size controls)
  reference.py     OpenCV
  lowrank.py       SVD / low-rank kernel decomposition (Chapter 8)
  cost_model.py    FLOP and byte counts per pixel (for GFLOP/s, GiB/s, Roofline)
  metrics.py       max abs error, relative Frobenius error, PSNR
tests/             correctness tests (pytest)
bench/             benchmarks, STREAM, experiments, plots, hardware.json
results/           CSVs and figures (generated)
report/report.md   report skeleton
run_all.py         one command for everything
```

## Shared rules (all stages)

- Input: uint8 RGB `(H, W, 3)`. Grayscale: `0.299 R + 0.587 G + 0.114 B`.
- Blur: correlation with a normalised Gaussian; σ from OpenCV's rule if not given.
- Borders: reflect-101 (OpenCV default).
- Math in float32; round half up and clip to uint8.
- Pass condition: max absolute error ≤ 1 grey level vs serial and vs OpenCV.
- Timing: 1 warm-up call (excludes JIT compile), then the median of 5 runs.
- Units: GiB/s = bytes / 1024³ / s. FLOP/byte counts in `filters/cost_model.py`.

## Roles

Fill in names once the group decides.

| Role | Member | Owns |
|---|---|---|
| A: Baseline, correctness, hardware | Ali Ammar | `serial.py`, `kernel.py`, `tests/`, `bench/hardware.*`, `bench/stream.py` |
| B: Single-core optimization | Matthew Mankiewicz | `vectorized.py`, `jit.py`, `_kernels.py`, `cost_model.py`, `lowrank.py` |
| C: Parallelism and benchmarks | Dalya Karem | `parallel.py`, `bench/run_benchmarks.py`, `bench/workload.py`, `bench/plots.py`, `run_all.py` |

## Open work (headroom to explore)

The code works end to end, but OpenCV is still well ahead, which is where most
of the report's analysis comes from. Ideas, roughly by expected payoff:

- **JIT 2-D blur is scalar.** The inner loop converts uint8 to float per tap and
  does not vectorize. Try converting the padded image to float32 once, swapping
  loop order to accumulate whole rows (as `sep_v_*` already does), or
  `fastmath=True`.
- **Separable pass memory traffic.** The float32 temp buffer doubles the bytes moved.
  Try processing in horizontal strips that fit in L2 (cache blocking), so the
  temp buffer never leaves cache.
- **Threads beyond 6.** Scaling flattens past the 6 physical cores (SMT shares
  FP units) and grayscale hits the bandwidth ceiling. Explain this with the Roofline.
- **Stretch:** a Numba CUDA kernel, or a Sobel edge filter.
