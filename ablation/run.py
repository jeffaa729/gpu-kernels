"""Check and time SM89 or Hopper GEMM stages and plot their rooflines."""
import argparse
import csv
import ctypes
import io
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from common import Operation, measure
from plot import plot_results

FP32 = ["scalar", "shared tiling", "register tiling", "warp tiling + layout",
        "vector global", "vector shared + stores", "shared pipeline", "register pipeline"]
BF16 = ["Tensor Core", "vector global", "XOR swizzle", "async copy", "async pipeline"]
HOPPER = ["mma.sync baseline", "TMA + WGMMA (serial)", "warp-specialized pipeline",
          "register redistribution", "persistent grid", "L2-friendly scheduling",
          "wider 128x256 tile", "first WGMMA overwrite", "TMA output store"]


class ClockSampler:
    def __init__(self):
        self.rows = []
        self.process = subprocess.Popen([
            "nvidia-smi", "--query-gpu=clocks.current.sm,clocks.current.memory,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits", "--loop-ms=200"], stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        for line in self.process.stdout:
            try:
                sm, memory, utilization, temperature = map(float, line.strip().split(","))
                self.rows.append(dict(time=time.time(), sm_mhz=sm, memory_mhz=memory,
                                      utilization_pct=utilization, temperature_c=temperature))
            except ValueError:
                continue

    def stop(self):
        self.process.terminate()
        self.process.wait()
        self.thread.join()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--sample-ms", type=float, default=20)
    parser.add_argument("--warmup-ms", type=float, default=150)
    parser.add_argument("--ncu", choices=["auto", "off", "required"], default="auto")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if any(n <= 0 or n % 128 for n in args.sizes):
        parser.error("sizes must be positive multiples of 128")
    return args


def save_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows):
    headers = ["size", "dtype", "stage", "custom us", "cuBLAS us", "cuBLAS %", "vs prev"]
    cells = []
    for row in rows:
        if row["stage"] == "cuBLAS":
            continue
        baseline = next(r for r in rows if r["dtype"] == row["dtype"] and r["size"] == row["size"] and r["stage"] == "cuBLAS")
        cells.append([str(row["size"]), row["dtype"], row["stage"], f'{row["time_us"]:.2f}',
                      f'{baseline["time_us"]:.2f}', f'{100 * baseline["time_us"] / row["time_us"]:.1f}',
                      "-" if row["speedup_previous"] is None else f'{row["speedup_previous"]:.2f}x'])
    widths = [max(len(r[i]) for r in [headers, *cells]) for i in range(len(headers))]
    def line(r):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(headers), line(["-" * w for w in widths]), *map(line, cells)]) + "\n"


def profile(rows, args):
    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    if not Path(ncu).exists():
        if args.ncu == "required":
            raise RuntimeError("Nsight Compute not found")
        return [], "Nsight Compute not found; analytical traffic only."
    reports = args.output_dir / "ncu"
    reports.mkdir(exist_ok=True)
    measured = []
    for index, row in enumerate(rows):
        name = f'{row["dtype"]}_{row["stage"]}_{row["size"]}'
        raw = reports / f"{name}.csv"
        result = subprocess.run([
            ncu, "--profile-from-start", "off", "--cache-control", "all", "--clock-control", "none",
            "--metrics", "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum",
            "--csv", "--log-file", str(raw), "--export", str(reports / name), "--force-overwrite",
            str(ROOT / "build/ablation/profile_gemm"), str(int(row["dtype"] == "bf16")),
            str(row["stage_index"]), str(row["size"])], capture_output=True, text=True)
        log = raw.read_text() if raw.exists() else ""
        if result.returncode or "ERR_NVGPUCTRPERM" in log:
            (reports / "failure.log").write_text(log + result.stdout + result.stderr)
            if args.ncu == "required" or measured:
                raise RuntimeError(f"Nsight profiling failed; see {reports / 'failure.log'}")
            return [], "Nsight counters unavailable; see ncu/failure.log. Analytical traffic only."
        # Export mode saves the report without printing the metric table.
        # Read the requested metrics back from that report explicitly.
        exported = subprocess.run([
            ncu, "--import", str(reports / f"{name}.ncu-rep"), "--page", "details", "--csv",
            "--print-units", "base", "--metrics", "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum"],
            capture_output=True, text=True, check=True)
        log = exported.stdout
        raw.write_text(log)
        start = log.find('"ID"')
        if start < 0:
            raise RuntimeError(f"Missing Nsight CSV header: {raw}")
        metrics = list(csv.DictReader(io.StringIO(log[start:])))
        duration = read = write = 0.
        for metric in metrics:
            key, value, unit = metric.get("Metric Name"), metric.get("Metric Value", ""), metric.get("Metric Unit", "")
            if not value:
                continue
            value = float(value.replace(",", ""))
            if key == "gpu__time_duration.sum":
                duration += value * {"nsecond": .001, "usecond": 1, "msecond": 1000,
                                     "second": 1e6, "ns": .001, "us": 1, "ms": 1000}.get(unit, 1)
            elif key in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
                value *= {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}.get(unit, 1)
                if key == "dram__bytes_read.sum": read += value
                else: write += value
        if duration <= 0 or read + write <= 0:
            raise RuntimeError(f"Missing time/DRAM metrics: {raw}")
        measured.append(dict(dtype=row["dtype"], size=row["size"], stage=row["stage"],
                             stage_index=row["stage_index"], time_us=duration, dram_read_bytes=read,
                             dram_write_bytes=write, intensity=row["flops"] / (read + write),
                             tflops=row["flops"] / duration / 1e6))
        print(f'Nsight {index + 1}/{len(rows)}: {name}', flush=True)
    save_csv(args.output_dir / "measured_dram.csv", measured)
    return measured, "Cold-cache Nsight kernel replay; clocks unlocked; duration and DRAM bytes from the same report."


@torch.no_grad()
def main():
    args = arguments()
    capability = torch.cuda.get_device_capability()
    if capability not in [(8, 9), (9, 0)]:
        raise RuntimeError("Ablation supports SM89 and SM90")
    hopper = capability == (9, 0)
    build = ROOT / ("build/ablation_sm90" if hopper else "build/ablation")
    args.output_dir = args.output_dir or ROOT / ("profiles/ablation_sm90" if hopper else "profiles/ablation")
    if hopper:
        args.ncu = "off"
        if any(n % 256 for n in args.sizes):
            raise ValueError("Hopper ablation sizes must be multiples of 256")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    lib = ctypes.CDLL(str(build / "libgemm_ablation.so"))
    lib.ablation_gemm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    lib.ablation_copy.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_void_p]
    lib.ablation_attributes.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ablation_error.restype = ctypes.c_char_p
    def checked(status):
        if status: raise RuntimeError(lib.ablation_error().decode())
    checked(lib.ablation_init())
    attributes = (ctypes.c_int * 4)()
    checked(lib.ablation_attributes(attributes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler = ClockSampler()
    rows, all_samples, measured = [], {}, []
    checks = 0
    try:
        suites = [(torch.bfloat16, HOPPER)] if hopper else [(torch.float32, FP32), (torch.bfloat16, BF16)]
        for dtype, names in suites:
            dtype_name = "fp32" if dtype == torch.float32 else "bf16"
            prefix = "H" if hopper else ("F" if dtype == torch.float32 else "T")
            for n in args.sizes:
                generator = torch.Generator(device="cuda").manual_seed(123 + n)
                A = torch.randn((n, n), device="cuda", dtype=dtype, generator=generator) * .1
                B = torch.randn((n, n), device="cuda", dtype=dtype, generator=generator) * .1
                expected = (A.float() @ B.float().t()).to(dtype)
                # Physical [N,M] output viewed as logical column-major [M,N].
                C = torch.full((n, n), float("nan"), device="cuda", dtype=dtype).t()
                previous = None
                for stage in [*range(len(names)), -1]:
                    label = f"{prefix}{stage}" if stage >= 0 else "cuBLAS"
                    if n >= 4096:
                        print(f"Timing {dtype_name} {n}: {label}", flush=True)
                    def call(stage=stage):
                        checked(lib.ablation_gemm(C.data_ptr(), A.data_ptr(), B.data_ptr(), n, n, n,
                                                 int(dtype == torch.bfloat16), stage, torch.cuda.current_stream().cuda_stream))
                        return C
                    tolerance = .02 if dtype == torch.bfloat16 else 2e-4
                    operation = Operation(f"M={n},N={n},K={n}", dtype_name, label,
                                          {"custom": call}, (expected,), tolerance, tolerance)
                    C.fill_(float("nan"))
                    operation.check(call())
                    checks += 1
                    if args.test_only:
                        continue
                    # Avoid huge graphs for the slow scalar baseline; use more nodes for fast kernels.
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(); call(); end.record(); end.synchronize()
                    args.graph_operations = max(1, min(16, int(1 / max(begin.elapsed_time(end), .001))))
                    samples = measure(operation, args)["custom"]
                    elapsed = statistics.median(samples)
                    flops = 2 * n ** 3
                    minimum_bytes = 3 * n ** 2 * A.element_size()
                    rows.append(dict(dtype=dtype_name, size=n, stage=label, stage_index=stage,
                                     technique=names[stage] if stage >= 0 else "cuBLAS TN",
                                     time_us=elapsed, flops=flops, minimum_bytes=minimum_bytes,
                                     intensity=flops / minimum_bytes, tflops=flops / elapsed / 1e6,
                                     speedup_previous=previous / elapsed if stage > 0 else None))
                    all_samples[f"{dtype_name}/{n}/{label}"] = dict(graph_operations=args.graph_operations, samples_us=samples)
                    previous = elapsed
                if not args.test_only:
                    print(f"Finished {dtype_name} {n}: {len(names)} stages + cuBLAS", flush=True)
        if args.test_only:
            print(f"Passed {checks} GEMM checks")
            return

        del A, B, C, expected, operation
        torch.cuda.empty_cache()

        copy_gbps = None
        if not hopper:
            # 128 MiB per buffer exceeds L2 capacity; count both read and write.
            src = torch.randn(32 * 1024 * 1024, device="cuda")
            dst = torch.empty_like(src)
            def copy():
                checked(lib.ablation_copy(dst.data_ptr(), src.data_ptr(), src.numel() // 4, torch.cuda.current_stream().cuda_stream))
                return dst
            args.graph_operations = 1
            copy_us = statistics.median(measure(Operation("128 MiB", "fp32", "copy", {"custom": copy}, (src,), 0, 0), args)["custom"])
            copy_gbps = 2 * src.numel() * src.element_size() / copy_us / 1000
            del src, dst
            torch.cuda.empty_cache()
        save_csv(args.output_dir / "timings.csv", rows)
        report = table(rows)
        (args.output_dir / "timings.md").write_text(report)
        print(report, end="", flush=True)
        ncu_status = "Profiling disabled; analytical traffic only."
        if args.ncu != "off":
            measured, ncu_status = profile(rows, args)
    finally:
        sampler.stop()
        lib.ablation_destroy()

    active = [r["sm_mhz"] for r in sampler.rows if r["utilization_pct"] >= 50]
    if not active:
        raise RuntimeError("No active GPU clock samples were captured; rerun with longer warmup")
    clock = max(active)
    sm_count, reported_clock, memory_clock, bus = attributes
    metadata = dict(gpu=torch.cuda.get_device_name(), sm_count=sm_count, torch=torch.__version__, cuda=torch.version.cuda,
                    clock_assumption_mhz=clock, reported_clock_khz=reported_clock,
                    clock_basis="Maximum sampled SM clock while GPU utilization >=50%; theoretical ceiling at that clock, not sustained measured compute.",
                    fp32_peak_tflops=sm_count * 128 * 2 * clock / 1e6,
                    bf16_peak_tflops=sm_count * (4096 if hopper else 512) * clock / 1e6,
                    stage_names={"bf16": HOPPER} if hopper else {"fp32": FP32, "bf16": BF16},
                    stage_prefixes={"bf16": "H"} if hopper else {"fp32": "F", "bf16": "T"},
                    memory_clock_khz=memory_clock, memory_bus_bits=bus,
                    memory_peak_gbps=2 * memory_clock * 1000 * bus / 8 / 1e9,
                    measured_copy_gbps=copy_gbps, correctness_checks=checks, profiling=ncu_status,
                    layout="A[M,K] and physical B[N,K] are row-major; C[M,N] is column-major; logical C=A@B.T, cuBLAS TN.",
                    precision="FP32 CUDA Core arithmetic (TF32 disabled); BF16 inputs/output with FP32 accumulation.",
                    timing="CUDA Graph replay; median; warmup, correctness, allocations and graph capture excluded.",
                    intensity="2MNK / (element_bytes*(MK+NK+MN)); minimum algorithmic IO, not measured DRAM traffic.",
                    limitations="F1->F2 changes block size along with register tiling; F2->F3 changes warp mapping and shared layout; F5 changes shared loads and stores together. Later stages hold tile sizes fixed. For K>2048, F0 uses 1M-element launch chunks and F1 uses 128-tile-row chunks for watchdog safety; graph timing includes all chunks, and Nsight counters/durations sum the cold-cache kernel replays. DRAM writes can remain in L2 at kernel completion, particularly for small matrices.",
                    sources=(["https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/",
                              "https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog",
                              "https://github.com/pranjalssh/fast.cu/tree/main/h100/matmul"] if hopper else
                             ["https://www.nvidia.com/en-us/geforce/laptops/compare/",
                              "https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf"]),
                    clocks=sampler.rows, samples=all_samples)
    if hopper:
        metadata["precision"] = "BF16 inputs/output; FP32 accumulation; TF32 disabled for FP32 correctness oracle."
        metadata["limitations"] = ("H0->H1 jointly changes instruction family, copies, shared layout and BK (32->64); "
            "H1->H2 adds warp specialization and a five-stage queue; H5->H6 changes BN (128->256) and stages (5->3). "
            "Other transitions retain tile/queue dimensions. These are engineering stages, not one-variable causal experiments. "
            "Arithmetic intensity uses ideal minimum IO, not hardware counters; it cannot establish actual memory traffic, L2 hits or bottlenecks. "
            "Warm graph replays can reuse L2 data. The compute roof is a clock-scaled theoretical dense BF16 ceiling, not measured sustained throughput.")
    (args.output_dir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(rows, measured, metadata, args.output_dir)
    print(f"Passed {checks} GEMM checks. Plots: {args.output_dir}")
    print(ncu_status)


if __name__ == "__main__":
    # CUDA Graph capture requires a non-default stream on this PyTorch build.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        main()
    stream.synchronize()
