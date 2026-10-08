"""Roofline figures distinguish analytical graph timing from measured DRAM profiling."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FP32 = ["scalar", "shared tiling", "register tiling", "warp tiling + layout", "vector global",
        "vector shared + stores", "shared pipeline", "register pipeline"]
BF16 = ["Tensor Core", "vector global", "XOR swizzle", "async copy", "async pipeline"]


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def roofline(rows, metadata, output, measured=False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8))
    for ax, dtype, names in zip(axes, ["fp32", "bf16"], [FP32, BF16]):
        selected = [r for r in rows if r["dtype"] == dtype]
        peak, bw = metadata[f"{dtype}_peak_tflops"], metadata["memory_peak_gbps"]
        xs = [float(r["intensity"]) for r in selected]
        ys = [float(r["tflops"]) for r in selected]
        ridge = peak * 1000 / bw
        x = np.geomspace(min(min(xs), ridge) / 4, max(max(xs), ridge) * 1.6, 300)
        ax.loglog(x, np.minimum(peak, bw * x / 1000), color="black", linewidth=2,
                  label=f"Theoretical roof: {peak:.1f} TFLOP/s, {bw:.0f} GB/s")
        stages = [f'{"F" if dtype == "fp32" else "T"}{i}' for i in range(len(names))] + ["cuBLAS"]
        colors = plt.get_cmap("tab10")
        for i, stage in enumerate(stages):
            points = sorted([r for r in selected if r["stage"] == stage], key=lambda r: int(r["size"]))
            if not points: continue
            label = f"{stage}: {names[i]}" if stage != "cuBLAS" else "cuBLAS"
            ax.plot([float(r["intensity"]) for r in points], [float(r["tflops"]) for r in points],
                    color="black" if stage == "cuBLAS" else colors(i), marker="*" if stage == "cuBLAS" else "o",
                    markersize=10 if stage == "cuBLAS" else 5, linewidth=1.2, label=label)
            if stage == "cuBLAS":
                for point in points:
                    label_offset = (5, 7)
                    crowded = measured and dtype == "fp32"
                    if crowded:
                        label_offset = {512: (-18, -16), 1024: (18, -24), 4096: (-36, 36), 8192: (-44, 16)}.get(int(point["size"]), (5, 7))
                    ax.annotate(str(point["size"]), (float(point["intensity"]), float(point["tflops"])),
                                xytext=label_offset, textcoords="offset points", fontsize=8,
                                arrowprops=dict(arrowstyle="-", color="gray", linewidth=.6) if crowded and int(point["size"]) >= 1024 else None,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=.5))
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(min(ys) * .6, max(peak, max(ys)) * 1.6)
        ax.set_title("FP32 CUDA Cores" if dtype == "fp32" else "BF16 Tensor Cores / FP32 accumulation")
        ax.set_xlabel("Measured DRAM arithmetic intensity (FLOP/byte)" if measured else "Analytical arithmetic intensity (FLOP/byte)")
        ax.set_ylabel("Achieved performance (TFLOP/s)")
        ax.grid(True, which="both", alpha=.2)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f'{metadata["gpu"]}: GEMM ablation roofline', fontsize=14)
    caption = ("NCU cold-cache duration and DRAM bytes from the same replay reports; each line follows increasing matrix size."
               if measured else "Measured CUDA Graph runtime; X uses minimum algorithmic IO (not DRAM counters). Labels on cuBLAS show matrix size.")
    fig.text(.5, .035, caption, ha="center", fontsize=9)
    fig.text(.5, .01, f'Theoretical compute roofs at sampled active clock {metadata["clock_assumption_mhz"]:.0f} MHz; dense BF16 throughput.', ha="center", fontsize=9)
    fig.subplots_adjust(top=.86, bottom=.38, wspace=.23)
    save(fig, output, "roofline_measured_dram" if measured else "roofline_analytical")


def plot_results(rows, measured, metadata, output):
    roofline(rows, metadata, output)
    if measured: roofline(measured, metadata, output, True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, dtype, names in zip(axes, ["fp32", "bf16"], [FP32, BF16]):
        selected = [r for r in rows if r["dtype"] == dtype]
        n = max(int(r["size"]) for r in selected)
        points = [r for r in selected if int(r["size"]) == n]
        ax.bar([r["stage"] for r in points], [float(r["tflops"]) for r in points],
               color=["#333333" if r["stage"] == "cuBLAS" else plt.get_cmap("tab10")(i) for i, r in enumerate(points)])
        for i, point in enumerate(points):
            ax.annotate(f'{float(point["tflops"]):.2f}', (i, float(point["tflops"])), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(f"{dtype.upper()} / M=N=K={n}")
        ax.set_ylabel("Achieved performance (TFLOP/s)")
        ax.set_ylim(0, max(float(r["tflops"]) for r in points) * 1.18)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Incremental GEMM techniques versus cuBLAS")
    fig.tight_layout()
    save(fig, output, "stage_performance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with (args.directory / "timings.csv").open() as file: rows = list(csv.DictReader(file))
    measured_path = args.directory / "measured_dram.csv"
    with measured_path.open() if measured_path.exists() else open("/dev/null") as file:
        measured = list(csv.DictReader(file))
    metadata = json.loads((args.directory / "environment.json").read_text())
    plot_results(rows, measured, metadata, args.directory)
