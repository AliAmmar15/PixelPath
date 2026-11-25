"""Theoretical hardware peaks for the Roofline (from bench/hardware.json).

    peak compute   = cores x clock x FLOP/cycle/core          (GFLOP/s)
    peak bandwidth = MT/s x 10^6 x (bus bits / 8) x channels / 1024^3   (GiB/s)

The clock uses the boost frequency, which is an upper bound; all-core
sustained clocks are usually a bit lower.

Run:  python -m bench.hardware
"""

from bench.common import ROOT, load_json

GIB = 1024 ** 3


def load_hardware():
    hw = load_json(ROOT / "bench" / "hardware.json")
    fpc = hw["fp32_flop_per_cycle_per_core"]
    hw["peak_gflops_1core"] = hw["clock_ghz_boost"] * fpc
    hw["peak_gflops"] = hw["cores"] * hw["clock_ghz_boost"] * fpc
    hw["peak_gibs"] = hw["mem_mts"] * 1e6 * (hw["mem_bus_bits"] / 8) * hw["mem_channels"] / GIB
    hw["ridge_intensity"] = hw["peak_gflops"] * 1e9 / (hw["peak_gibs"] * GIB)
    return hw


def main():
    hw = load_hardware()
    print(f"CPU                     {hw['cpu']}, {hw['cores']} cores / {hw['threads']} threads")
    print(f"Peak fp32, 1 core       {hw['peak_gflops_1core']:.1f} GFLOP/s")
    print(f"Peak fp32, all cores    {hw['peak_gflops']:.1f} GFLOP/s")
    print(f"Peak memory bandwidth   {hw['peak_gibs']:.2f} GiB/s  ({hw['memory']})")
    print(f"Ridge point             {hw['ridge_intensity']:.1f} FLOP/byte "
          "(kernels below this are memory-bound)")


if __name__ == "__main__":
    main()
