"""Workload-distribution experiments for the parallel stage.

  * split:     parallel loop over rows vs over flattened pixel indices
  * chunksize: 0 = Numba default (static: iterations divided evenly up front);
               n > 0 = iterations handed out n at a time (dynamic-style,
               better load balance, more scheduling overhead)
  * threads:   1 .. max

Grayscale is memory-bound and the direct blur is compute-bound, so the two
react differently. Writes results/workload.csv.

Run:  python -m bench.workload [--size 4K] [--ksize 9]
"""

import argparse
import csv

from bench.common import RESULTS, SIZES, ensure_dirs, load_image, time_call
from bench.run_benchmarks import thread_counts
from filters import parallel, reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4K", choices=list(SIZES))
    ap.add_argument("--ksize", type=int, default=9)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--chunksizes", nargs="+", type=int, default=[0, 1, 4, 16, 64, 256])
    args = ap.parse_args()
    ensure_dirs()

    h, w = SIZES[args.size]
    img = load_image(h, w)
    gray = reference.grayscale(img)
    jobs = {
        "grayscale": lambda split: parallel.grayscale(img, split=split),
        "blur_2d": lambda split: parallel.blur_2d(gray, args.ksize, split=split),
    }
    rows = []
    for filt, run in jobs.items():
        for split in ("rows", "pixels"):
            for chunk in args.chunksizes:
                # chunks of single pixels are absurdly fine-grained; scale them up
                eff_chunk = chunk * w if (split == "pixels" and chunk > 0) else chunk
                for t in thread_counts():
                    parallel.set_threads(t)
                    old = parallel.set_chunksize(eff_chunk)
                    try:
                        tmed, tmin, _ = time_call(lambda: run(split), args.repeats)
                    finally:
                        parallel.set_chunksize(old)
                    rows.append({"filter": filt, "split": split, "chunk_rows": chunk,
                                 "chunk_iterations": eff_chunk, "threads": t,
                                 "time_s": f"{tmed:.6g}", "time_min_s": f"{tmin:.6g}",
                                 "mpix_s": f"{h * w / tmed / 1e6:.6g}"})
                    print(f"{filt:<9} split={split:<6} chunk={chunk:<4} t={t:<2} {tmed * 1e3:9.2f} ms")
    parallel.set_threads(parallel.max_threads())

    with open(RESULTS / "workload.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to results/workload.csv")


if __name__ == "__main__":
    main()
