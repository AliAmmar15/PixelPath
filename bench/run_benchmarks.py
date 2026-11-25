"""Main benchmark: every stage x image size x kernel size x thread count.

Writes results/results.csv (one row per configuration) and saves sample
output images to results/samples/.

Timing: 1 warm-up call (so Numba compile time is excluded), then the median
of --repeats timed calls with time.perf_counter.

The pure-Python serial stage would take minutes or hours on large images, so
when a configuration exceeds --serial-budget (pixels x taps) it is timed on a
strip of rows and scaled up by H / strip_rows. Those rows are marked
estimated=1 in the CSV; say so in the report.

Run:  python -m bench.run_benchmarks            (full, ~10-20 min)
      python -m bench.run_benchmarks --quick    (~2 min)
"""

import argparse
import csv
import os

import numpy as np

from bench.common import RESULTS, SIZES, ensure_dirs, load_image, time_call
from filters import BLUR_STAGES, GRAY_STAGES, PARALLEL_STAGES, cost_model, parallel, reference
from filters.metrics import max_abs_error, rel_frobenius_error

FIELDS = ["filter", "stage", "size", "height", "width", "pixels", "ksize", "threads",
          "time_s", "time_min_s", "mpix_s", "gflops", "gibs", "intensity",
          "max_err_vs_opencv", "rel_frob_vs_opencv", "estimated"]


def thread_counts():
    n = parallel.max_threads()
    return sorted({1, 2, 4, max(1, n // 2), n} & set(range(1, n + 1)))


def set_all_threads(n):
    parallel.set_threads(n)
    reference.set_threads(n)


def serial_strip(img, taps, budget, min_rows):
    """Rows of img to time so that rows * width * taps ~= budget."""
    h, w = img.shape[:2]
    if h * w * taps <= budget:
        return h
    return min(h, max(min_rows, budget // (w * taps)))


def make_row(filt, stage, size_name, h, w, ksize, threads, tmed, tmin, variant, out, ref, estimated):
    px = h * w
    flop = cost_model.flops_per_pixel(filt, variant, ksize) * px
    byts = cost_model.bytes_per_pixel(filt, variant, ksize) * px
    return {
        "filter": filt, "stage": stage, "size": size_name, "height": h, "width": w, "pixels": px,
        "ksize": ksize if ksize else "", "threads": threads,
        "time_s": f"{tmed:.6g}", "time_min_s": f"{tmin:.6g}",
        "mpix_s": f"{px / tmed / 1e6:.6g}",
        "gflops": f"{flop / tmed / 1e9:.6g}",
        "gibs": f"{byts / tmed / cost_model.GIB:.6g}",
        "intensity": f"{cost_model.intensity(filt, variant, ksize):.4g}",
        "max_err_vs_opencv": "" if out is None else f"{max_abs_error(out, ref):.0f}",
        "rel_frob_vs_opencv": "" if out is None else f"{rel_frobenius_error(out, ref):.3g}",
        "estimated": int(estimated),
    }


def bench_one(fn, img_full, taps, args, is_serial, min_rows):
    """Returns (median, min, output-or-None, estimated)."""
    if not is_serial:
        tmed, tmin, out = time_call(lambda: fn(img_full), args.repeats)
        return tmed, tmin, out, False
    rows = serial_strip(img_full, taps, args.serial_budget, min_rows)
    strip = img_full[:rows]
    tmed, tmin, out = time_call(lambda: fn(strip), min(args.repeats, 3), warmup=0)
    if rows == img_full.shape[0]:
        return tmed, tmin, out, False
    scale = img_full.shape[0] / rows
    return tmed * scale, tmin * scale, None, True


def save_samples(img, gray, ksize):
    from PIL import Image
    d = RESULTS / "samples"
    d.mkdir(exist_ok=True)
    Image.fromarray(img).save(d / "input.png")
    Image.fromarray(gray).save(d / "grayscale.png")
    Image.fromarray(parallel.blur_separable(gray, ksize)).save(d / f"blur_k{ksize}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small sweep for a fast check")
    ap.add_argument("--sizes", nargs="+", choices=list(SIZES), default=None)
    ap.add_argument("--ksizes", nargs="+", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--serial-budget", type=int, default=3_000_000,
                    help="max pixels*taps run by the pure-Python stage before switching to strip timing")
    ap.add_argument("--out", default=str(RESULTS / "results.csv"))
    args = ap.parse_args()

    sizes = args.sizes or (["256", "512", "1080p"] if args.quick else list(SIZES))
    ksizes = args.ksizes or ([3, 9] if args.quick else [3, 9, 15])
    args.repeats = args.repeats or (3 if args.quick else 5)
    threads = thread_counts()

    ensure_dirs()
    print(f"sizes={sizes} ksizes={ksizes} threads={threads} repeats={args.repeats}")
    rows = []

    for size_name in sizes:
        h, w = SIZES[size_name]
        img = load_image(h, w)
        set_all_threads(parallel.max_threads())
        gray = reference.grayscale(img)
        if size_name == "512":
            save_samples(img, gray, 9)

        # ---------------- grayscale
        ref = reference.grayscale(img)
        for stage, fn in GRAY_STAGES.items():
            for t in (threads if stage in PARALLEL_STAGES else [1]):
                set_all_threads(t)
                tmed, tmin, out, est = bench_one(fn, img, 3, args, stage == "serial", 1)
                ref_part = ref if out is None or out.shape == ref.shape else ref[:out.shape[0]]
                rows.append(make_row("grayscale", stage, size_name, h, w, None, t, tmed, tmin,
                                     "direct", out, ref_part, est))
                print(f"  gray  {size_name:>5} {stage:<12} t={t:<2} {tmed * 1e3:10.2f} ms" + (" (est)" if est else ""))

        # ---------------- blur
        for k in ksizes:
            set_all_threads(parallel.max_threads())
            ref = reference.blur_2d(gray, k)
            for stage, (fn, variant) in BLUR_STAGES.items():
                taps = k * k if variant == "direct" else 2 * k
                for t in (threads if stage in PARALLEL_STAGES else [1]):
                    set_all_threads(t)
                    tmed, tmin, out, est = bench_one(lambda g, fn=fn, k=k: fn(g, k), gray, taps, args,
                                                     stage.startswith("serial"), k)
                    # a strip's last rows see a different border than the full image, so compare
                    # only full-image results
                    rows.append(make_row("blur", stage, size_name, h, w, k, t, tmed, tmin, variant,
                                         out if out is not None and out.shape == ref.shape else None, ref, est))
                    print(f"  blur  {size_name:>5} k={k:<2} {stage:<12} t={t:<2} {tmed * 1e3:10.2f} ms" + (" (est)" if est else ""))

    set_all_threads(parallel.max_threads())
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {os.path.relpath(args.out)}")


if __name__ == "__main__":
    main()
