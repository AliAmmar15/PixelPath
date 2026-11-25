"""Chapter 8 experiment: speed vs accuracy of low-rank kernel approximation.

For a non-separable kernel (a disk / bokeh blur), apply its best rank-r
approximation as r separable passes (filters/lowrank.py) for r = 1..rank and
compare against the direct 2-D correlation. Records time, speedup, relative
Frobenius error of the output, and sigma_{r+1}/sigma_1 (Theorem 8.7).

Writes results/lowrank.csv.

Run:  python -m bench.lowrank_bench [--size 1080p] [--radius 7]
"""

import argparse
import csv

from bench.common import RESULTS, SIZES, ensure_dirs, load_image, time_call
from filters import lowrank, parallel, reference
from filters.metrics import rel_frobenius_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1080p", choices=list(SIZES))
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    ensure_dirs()

    h, w = SIZES[args.size]
    gray = reference.grayscale(load_image(h, w))
    K = lowrank.disk_kernel(args.radius)
    S = lowrank.kernel_svd(K)[1]
    rank = lowrank.numerical_rank(K)
    k = K.shape[0]
    print(f"disk kernel {k}x{k}, numerical rank {rank}")
    print("singular values / sigma_1:", ", ".join(f"{s / S[0]:.3g}" for s in S[:rank + 1]))

    t_direct, _, exact = time_call(lambda: parallel.correlate_2d_float(gray, K), args.repeats)
    rows = [{"method": "direct_2d", "rank": k, "time_s": f"{t_direct:.6g}", "speedup_vs_direct": "1",
             "rel_frob_err": "0", "sigma_next_rel": "0", "flop_per_px": 2 * k * k}]
    print(f"direct 2-D       {t_direct * 1e3:8.2f} ms")

    for r in range(1, rank + 1):
        t, _, approx = time_call(lambda: lowrank.correlate_lowrank(gray, K, r, parallel), args.repeats)
        err = rel_frobenius_error(approx, exact)
        nxt = S[r] / S[0] if r < len(S) else 0.0
        rows.append({"method": "lowrank", "rank": r, "time_s": f"{t:.6g}",
                     "speedup_vs_direct": f"{t_direct / t:.4g}", "rel_frob_err": f"{err:.4g}",
                     "sigma_next_rel": f"{nxt:.4g}", "flop_per_px": 4 * k * r})
        print(f"rank {r:2d}          {t * 1e3:8.2f} ms   speedup {t_direct / t:5.2f}x   rel. error {err:.2e}")

    with open(RESULTS / "lowrank.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote results/lowrank.csv")


if __name__ == "__main__":
    main()
