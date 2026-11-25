"""Make the report figures from results/*.csv and results/stream.json.

  optimization_path.png   time per stage (the figure the course guide requires)
  roofline.png            each stage vs the compute and bandwidth ceilings
  thread_scaling.png      speedup and parallel efficiency vs threads
  throughput_vs_size.png  Mpixel/s vs image size
  workload.png            row vs pixel split, chunk size   (if workload.csv exists)
  lowrank.png             Chapter 8 rank-r approximation  (if lowrank.csv exists)

Colour follows the stage, never the rank: serial = blue, NumPy = orange,
JIT = aqua, parallel = yellow, OpenCV = neutral grey (it's the reference, not
one of ours). Separable variants use the same hue with a hatch, so the
direct/separable difference doesn't rely on colour alone.

Run:  python -m bench.plots [--size 4K] [--ksize 9]
"""

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bench.common import FIGURES, RESULTS, SIZES, ensure_dirs, load_json  # noqa: E402
from bench.hardware import load_hardware  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FAMILY_COLOR = {"serial": "#2a78d6", "numpy": "#eb6834", "jit": "#1baf7a",
                "parallel": "#eda100", "opencv": MUTED}
FAMILY_LABEL = {"serial": "Serial (Python)", "numpy": "NumPy", "jit": "Numba JIT",
                "parallel": "Numba parallel", "opencv": "OpenCV"}
STAGE_LABEL = {
    "serial": "Serial", "numpy": "NumPy", "jit": "JIT", "parallel": "Parallel", "opencv": "OpenCV",
    "serial_2d": "Serial 2-D", "serial_sep": "Serial sep.", "numpy_2d": "NumPy 2-D",
    "numpy_sep": "NumPy sep.", "jit_2d": "JIT 2-D", "jit_sep": "JIT sep.",
    "parallel_2d": "Parallel 2-D", "parallel_sep": "Parallel sep.",
}


def family(stage):
    return stage.split("_")[0]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
        "lines.linewidth": 2, "lines.markersize": 6,
    })


def read_csv(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(r, key):
    return float(r[key]) if r.get(key, "") != "" else float("nan")


def pick(rows, **kw):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]


def best_threads(rows):
    """For each stage keep the fastest thread count."""
    by = {}
    for r in rows:
        s = r["stage"]
        if s not in by or num(r, "time_s") < num(by[s], "time_s"):
            by[s] = r
    return by


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


# ------------------------------------------------------------------ 1. optimization path

def plot_optimization_path(rows, size, k):
    panels = [("Grayscale", pick(rows, filter="grayscale", size=size),
               ["serial", "numpy", "jit", "parallel", "opencv"]),
              (f"Gaussian blur, k = {k}", pick(rows, filter="blur", size=size, ksize=k),
               ["serial_2d", "serial_sep", "numpy_2d", "numpy_sep", "jit_2d", "jit_sep",
                "parallel_2d", "parallel_sep", "opencv"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [5, 9]})
    for ax, (title, sub, order) in zip(axes, panels):
        best = best_threads(sub)
        order = [s for s in order if s in best]
        if not order:
            continue
        times = [num(best[s], "time_s") * 1e3 for s in order]
        base = times[0]

        def fmt_ms(v):
            return f"{v:,.0f} ms" if v >= 100 else f"{v:.3g} ms"
        y = np.arange(len(order))[::-1]
        for yi, s, t in zip(y, order, times):
            hatch = "////" if s.endswith("_sep") else None
            ax.barh(yi, t, height=0.7, color=FAMILY_COLOR[family(s)], edgecolor=SURFACE,
                    linewidth=2, hatch=hatch)
            est = best[s]["estimated"] == "1"
            thr = best[s]["threads"]
            label = f"{fmt_ms(t)}  ({base / t:,.0f}x)" if s != order[0] else f"{fmt_ms(t)}  (baseline)"
            if est:
                label += " est."
            if family(s) in ("parallel", "opencv"):
                label += f"  [{thr} thr]"
            ax.text(t * 1.15, yi, label, va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y, [STAGE_LABEL[s] for s in order])
        ax.set_xscale("log")
        ax.set_xlim(min(times) / 2, max(times) * 60)
        ax.set_xlabel("time per image (ms, log scale), speedup vs serial in brackets")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    h, w = SIZES[size]
    fig.suptitle(f"Optimization path, {size} image ({w}x{h})", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "optimization_path.png")


# ------------------------------------------------------------------ 2. roofline

def plot_roofline(rows, size, k, hw, stream):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    I = np.logspace(-1, 3, 200)
    bw_bytes = hw["peak_gibs"] * 1024 ** 3 / 1e9  # GB/s in 1e9 bytes for GFLOP/s units
    ax.plot(I, np.minimum(hw["peak_gflops"], I * bw_bytes), color=INK2, lw=1.5)
    ax.plot(I, np.minimum(hw["peak_gflops_1core"], I * bw_bytes), color=INK2, lw=1, ls=":")
    ax.text(I[-1], hw["peak_gflops"] * 1.08, f"peak fp32, {hw['cores']} cores: {hw['peak_gflops']:.0f} GFLOP/s",
            ha="right", fontsize=7.5, color=INK2)
    ax.text(I[-1], hw["peak_gflops_1core"] * 1.08, f"1 core: {hw['peak_gflops_1core']:.0f} GFLOP/s",
            ha="right", fontsize=7.5, color=INK2)
    ax.text(0.12, 0.12 * bw_bytes * 1.5, f"spec bandwidth {hw['peak_gibs']:.1f} GiB/s",
            rotation=33, fontsize=7.5, color=INK2)
    if stream:
        mb = stream["best_gibs"] * 1024 ** 3 / 1e9
        ax.plot(I, np.minimum(hw["peak_gflops"], I * mb), color=MUTED, lw=1, ls="--")
        ax.text(0.3, 0.3 * mb / 3.2, f"measured (STREAM) {stream['best_gibs']:.1f} GiB/s",
                rotation=33, fontsize=7.5, color=MUTED)

    markers = {"grayscale": "o", "direct": "s", "separable": "^"}
    seen = set()
    for filt, sub in (("grayscale", pick(rows, filter="grayscale", size=size)),
                      ("blur", pick(rows, filter="blur", size=size, ksize=k))):
        for s, r in best_threads(sub).items():
            if r["estimated"] == "1" and filt == "blur" and False:
                continue
            kind = "grayscale" if filt == "grayscale" else ("separable" if s.endswith("_sep") or s == "opencv" else "direct")
            fam = family(s)
            label = FAMILY_LABEL[fam] if fam not in seen else None
            seen.add(fam)
            ax.scatter(num(r, "intensity"), num(r, "gflops"), s=55, marker=markers[kind],
                       color=FAMILY_COLOR[fam], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label)
    for kind, mk, txt in (("grayscale", "o", "grayscale"), ("direct", "s", f"blur 2-D (k={k})"),
                          ("separable", "^", f"blur separable (k={k})")):
        ax.scatter([], [], marker=mk, color=INK2, label=txt)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte, compulsory traffic)")
    ax.set_ylabel("attained GFLOP/s")
    ax.set_title(f"Roofline, {size} image, best thread count per stage", loc="left")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, "roofline.png")


# ------------------------------------------------------------------ 3. thread scaling

def plot_thread_scaling(rows, size, k, hw):
    series = [("Grayscale, parallel", pick(rows, filter="grayscale", size=size, stage="parallel"), "parallel", None),
              ("Grayscale, OpenCV", pick(rows, filter="grayscale", size=size, stage="opencv"), "opencv", None),
              (f"Blur 2-D, parallel", pick(rows, filter="blur", size=size, ksize=k, stage="parallel_2d"), "parallel", "-"),
              (f"Blur sep., parallel", pick(rows, filter="blur", size=size, ksize=k, stage="parallel_sep"), "parallel", "--"),
              (f"Blur, OpenCV", pick(rows, filter="blur", size=size, ksize=k, stage="opencv"), "opencv", "--")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    maxt = 1
    for name, sub, fam, ls in series:
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: int(r["threads"]))
        t = np.array([int(r["threads"]) for r in sub])
        tm = np.array([num(r, "time_s") for r in sub])
        sp = tm[0] / tm
        maxt = max(maxt, t.max())
        marker = "o" if name.startswith("Gray") else ("s" if "2-D" in name else "^")
        for ax, yv in ((a1, sp), (a2, sp / t)):
            ax.plot(t, yv, color=FAMILY_COLOR[fam], ls=ls or ":", marker=marker, label=name)
    ideal = np.arange(1, maxt + 1)
    a1.plot(ideal, ideal, color=AXIS, lw=1, ls="--", label="ideal (linear)")
    a2.axhline(1, color=AXIS, lw=1, ls="--")
    for ax in (a1, a2):
        ax.axvline(hw["cores"], color=GRID, lw=1.5)
        ax.text(hw["cores"], ax.get_ylim()[1] * 0.97 if ax is a2 else maxt * 0.95,
                f" {hw['cores']} physical cores", fontsize=7, color=MUTED, va="top")
        ax.set_xlabel("threads")
    a1.set_ylabel("speedup vs 1 thread")
    a1.set_title("Strong scaling", loc="left")
    a2.set_ylabel("parallel efficiency (speedup / threads)")
    a2.set_ylim(0, 1.15)
    a2.set_title("Parallel efficiency", loc="left")
    a1.legend(loc="upper left")
    a1.set_xlim(0.5, maxt + 0.5)
    fig.suptitle(f"Thread scaling, {size} image, blur k = {k}", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "thread_scaling.png")


# ------------------------------------------------------------------ 4. throughput vs size

def plot_throughput(rows, k):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, filt, stages, title in (
            (a1, "grayscale", ["serial", "numpy", "jit", "parallel", "opencv"], "Grayscale"),
            (a2, "blur", ["serial_2d", "numpy_2d", "jit_2d", "parallel_2d", "parallel_sep", "opencv"],
             f"Gaussian blur, k = {k}")):
        for s in stages:
            sub = pick(rows, filter=filt, stage=s) if filt == "grayscale" else pick(rows, filter=filt, stage=s, ksize=k)
            by_size = defaultdict(list)
            for r in sub:
                by_size[int(r["pixels"])].append(r)
            if not by_size:
                continue
            px = sorted(by_size)
            best = [min(by_size[p], key=lambda r: num(r, "time_s")) for p in px]
            mp = [num(r, "mpix_s") for r in best]
            ls = "--" if s.endswith("_sep") else "-"
            ax.plot(px, mp, color=FAMILY_COLOR[family(s)], ls=ls, marker="o", label=STAGE_LABEL[s])
            est = [(p, m) for p, m, r in zip(px, mp, best) if r["estimated"] == "1"]
            if est:
                ax.scatter(*zip(*est), s=60, facecolor=SURFACE, edgecolor=FAMILY_COLOR[family(s)],
                           linewidth=2, zorder=3)
            ax.annotate(STAGE_LABEL[s], (px[-1], mp[-1]), xytext=(5, 0), textcoords="offset points",
                        fontsize=7, color=INK2, va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("image size (pixels)")
        ax.set_ylabel("throughput (Mpixel/s, best thread count)")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7)
    fig.text(0.01, -0.1, "Hollow markers: serial time estimated from a strip of rows.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, "throughput_vs_size.png")


# ------------------------------------------------------------------ 5. workload distribution

def plot_workload(wl):
    if not wl:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    maxt = max(int(r["threads"]) for r in wl)
    for ax, filt, title in ((axes[0], "grayscale", "Grayscale (memory-bound)"),
                            (axes[1], "blur_2d", "Blur 2-D (compute-bound)")):
        sub = pick(wl, filter=filt, threads=maxt)
        chunks = sorted({int(r["chunk_rows"]) for r in sub})
        x = np.arange(len(chunks))
        for i, (split, color) in enumerate((("rows", FAMILY_COLOR["parallel"]), ("pixels", FAMILY_COLOR["jit"]))):
            ms = [num(pick(sub, split=split, chunk_rows=c)[0], "time_s") * 1e3 for c in chunks]
            bars = ax.bar(x + (i - 0.5) * 0.38, ms, width=0.36, color=color, edgecolor=SURFACE,
                          linewidth=2, label=f"split by {split}")
            ax.bar_label(bars, fmt="%.3g", fontsize=6.5, color=INK2, padding=2)
        ax.set_xticks(x, ["default\n(static)" if c == 0 else f"{c} row{'s' if c > 1 else ''}" for c in chunks])
        ax.set_xlabel("chunk size (dynamic scheduling granularity)")
        ax.set_ylabel("time (ms)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.legend(loc="upper left", ncol=2)
    fig.suptitle(f"Workload distribution at {maxt} threads", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "workload.png")


# ------------------------------------------------------------------ 6. low-rank

def plot_lowrank(lr):
    if not lr:
        return
    low = [r for r in lr if r["method"] == "lowrank"]
    ranks = [int(r["rank"]) for r in low]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    err = [max(num(r, "rel_frob_err"), 1e-9) for r in low]
    sig = [max(num(r, "sigma_next_rel"), 1e-9) for r in low]
    a1.plot(ranks, err, color=FAMILY_COLOR["serial"], marker="o", label="output error (relative Frobenius)")
    a1.plot(ranks, sig, color=FAMILY_COLOR["numpy"], marker="s", ls="--",
            label=r"$\sigma_{r+1}/\sigma_1$ (Theorem 8.7 bound on kernel error)")
    a1.set_yscale("log")
    a1.set_xlabel("rank r (number of separable passes)")
    a1.set_ylabel("relative error")
    a1.set_title("Accuracy vs rank", loc="left")
    a1.legend(fontsize=7)
    sp = [num(r, "speedup_vs_direct") for r in low]
    bars = a2.bar(ranks, sp, color=FAMILY_COLOR["parallel"], edgecolor=SURFACE, linewidth=2)
    a2.bar_label(bars, fmt="%.2gx", fontsize=7, color=INK2, padding=2)
    a2.axhline(1, color=INK2, lw=1, ls="--", label="direct 2-D (1x)")
    a2.legend(loc="upper right")
    a2.set_xlabel("rank r")
    a2.set_ylabel("speedup vs direct 2-D (parallel)")
    a2.set_title("Speed vs rank", loc="left")
    a2.grid(axis="x", visible=False)
    k = int([r for r in lr if r["method"] == "direct_2d"][0]["rank"])
    fig.suptitle(f"Low-rank (SVD) approximation of a {k}x{k} disk kernel", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "lowrank.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default=None, help="image size for single-size plots (default: largest run)")
    ap.add_argument("--ksize", type=int, default=None, help="blur kernel size (default: 9 if present)")
    args = ap.parse_args()
    ensure_dirs()
    style()

    rows = read_csv("results.csv")
    if not rows:
        raise SystemExit("results/results.csv not found: run python -m bench.run_benchmarks first")
    sizes_run = [s for s in SIZES if pick(rows, size=s)]
    size = args.size or sizes_run[-1]
    ks = sorted({int(r["ksize"]) for r in rows if r["ksize"]})
    k = args.ksize or (9 if 9 in ks else ks[len(ks) // 2])
    hw = load_hardware()
    stream = load_json(RESULTS / "stream.json")

    plot_optimization_path(rows, size, k)
    plot_roofline(rows, size, k, hw, stream)
    plot_thread_scaling(rows, size, k, hw)
    plot_throughput(rows, k)
    plot_workload(read_csv("workload.csv"))
    plot_lowrank(read_csv("lowrank.csv"))


if __name__ == "__main__":
    main()
