"""Build/run everything: tests -> hardware peaks -> STREAM -> benchmarks -> experiments -> figures.

    python run_all.py            full run for the final report (~15-30 min)
    python run_all.py --quick    small sweep to check everything works (~3 min)
    python run_all.py --plots    only redraw figures from existing results

Stops at the first failing step. Run the final benchmarks on one machine,
plugged in, with other programs closed.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(step, args):
    print(f"\n=== {step} ===", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"step '{step}' failed (exit code {result.returncode})")
    print(f"--- {step}: {time.perf_counter() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()

    if a.plots:
        run("figures", ["-m", "bench.plots"])
        return

    q = ["--quick"] if a.quick else []
    run("correctness tests", ["-m", "pytest", "-q"])
    run("hardware peaks", ["-m", "bench.hardware"])
    run("STREAM bandwidth", ["-m", "bench.stream"])
    run("benchmarks", ["-m", "bench.run_benchmarks", *q])
    run("workload distribution", ["-m", "bench.workload",
                                  *(["--size", "1080p", "--repeats", "3", "--chunksizes", "0", "1", "16"] if a.quick else [])])
    run("low-rank (Chapter 8)", ["-m", "bench.lowrank_bench", *(["--repeats", "3"] if a.quick else [])])
    run("figures", ["-m", "bench.plots"])
    print("\nDone. Results in results/, figures in results/figures/.")


if __name__ == "__main__":
    main()
