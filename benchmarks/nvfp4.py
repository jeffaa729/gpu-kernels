"""B200 NVFP4 GEMM and source-matched R0-R6 ablations against cuBLASLt."""

import csv
import ctypes
import json
import math
import subprocess
import sys

from common import I, P, ROOT, Operation, bind, checked, library, pointers, save_report, stream, torch
from reference.python.nvfp4 import dequantize, quantize

STAGES = ("main", "r0", "r1", "r2", "r3", "r4", "r5", "r6")


class NVFP4Operation(Operation):
    def check(self, result):
        super().check(result)
        checked(self.lib, "nvfp4", self.validate(self.context))


def cases(args, family):
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("The NVFP4 suite requires a B200/SM100 GPU.")
    if args.reference not in (None, "cublas", "cublaslt"):
        raise ValueError("NVFP4 GEMM reference: cublaslt")
    lib = library("nvfp4")
    create = bind(lib, "gpu_kernels_nvfp4_plan_create",
                  [ctypes.POINTER(P)] + [I] * 4 + [P] * 5 + [ctypes.c_size_t, P])
    destroy = bind(lib, "gpu_kernels_nvfp4_plan_destroy", [P], None)
    reference = bind(lib, "gpu_kernels_nvfp4_cublas", [P] * 5)
    custom = bind(lib, "gpu_kernels_nvfp4_custom", [P] * 3)
    validate = bind(lib, "gpu_kernels_nvfp4_validate", [P])
    l2_bytes = bind(lib, "gpu_kernels_nvfp4_l2_bytes", [])()
    stage = -1 if args.stage == "main" else int(args.stage[1:])
    if args.test:
        # The second case exercises ring wraps and persistent accumulator reuse.
        shapes = ((256, 256, 256), (2048, 4096, 2048))
    else:
        sizes = (8192,) if args.suite == "quick" else (8192, 16384)
        shapes = tuple((n, n, n) for n in sizes)
    graph_operations = args.graph_operations

    for m, n, k in shapes:
        torch.manual_seed(2026 + m + n + k)
        A, SFA = quantize(torch.randn((m, k), device="cuda") * .1)
        B, SFB = quantize(torch.randn((n, k), device="cuda") * .1)
        expected = (dequantize(A, SFA) @ dequantize(B, SFB).T).half()
        operands = (A, B, SFA, SFB)
        input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in operands)
        rotations = 1 if args.test else max(2, math.ceil(2 * l2_bytes / input_bytes))
        # A full rotation exceeds L2. Both graphs traverse the identical allocation set.
        args.graph_operations = math.ceil(graph_operations / rotations) * rotations
        workspace = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        custom_output = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
        reference_output = torch.empty_like(custom_output)
        slots, cursor = [], [0, 0]
        try:
            for rotation in range(rotations):
                inputs = operands if rotation == 0 else tuple(tensor.clone() for tensor in operands)
                context = P()
                checked(lib, "nvfp4", create(ctypes.byref(context), m, n, k, stage,
                    *pointers(inputs), workspace.data_ptr(), workspace.numel(), stream()))
                slots.append((context, inputs))

            def run(use_reference):
                context, inputs = slots[cursor[use_reference] % rotations]
                cursor[use_reference] += 1
                if use_reference:
                    checked(lib, "nvfp4", reference(context, *pointers((reference_output, *inputs[:2])), stream()))
                    return reference_output
                checked(lib, "nvfp4", custom(context, custom_output.data_ptr(), stream()))
                return custom_output

            functions = {"custom": lambda: run(0), "cuBLASLt": lambda: run(1)}
            operation = NVFP4Operation(f"M={m},N={n},K={k}", "nvfp4/fp16", f"TN/{args.stage}",
                                      functions, (expected,), 2e-3, 2e-3)
            operation.lib, operation.validate, operation.context = lib, validate, slots[0][0]
            operation.rotations = rotations
            operation.backend_settle_ms = 100
            yield operation
        finally:
            torch.cuda.current_stream().synchronize()
            for context, _ in slots:
                destroy(context)


def batch(args):
    """One rental session: isolate GPU faults by running each stage in its own process."""
    destination = ROOT / "profiles/runtime/nvfp4_batch"
    destination.mkdir(parents=True, exist_ok=True)
    rows, samples, failures = [], [], []
    jobs = [("r0", True), *((stage, False) for stage in (*STAGES[1:], "main"))]
    for stage, test in jobs:
        name = f"{stage}_test" if test else stage
        command = [sys.executable, str(ROOT / "benchmarks/run.py"), "nvfp4", "--stage", stage,
                   "--suite", "b200" if stage == "main" else "quick", "--reference", "cublaslt",
                   "--graph-operations", str(args.graph_operations), "--warmup-ms", str(args.warmup_ms),
                   "--sample-ms", str(args.sample_ms), "--trials", str(args.trials)]
        if test:
            command.append("--test")
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            log = result.stdout + result.stderr
            failed = result.returncode != 0
        except subprocess.TimeoutExpired as error:
            log = "Stage exceeded 180 seconds.\n"
            for output in (error.stdout, error.stderr):
                if output:
                    log += output.decode() if isinstance(output, bytes) else output
            failed = True
        (destination / f"{name}.log").write_text(log)
        print(f"{name}: {'FAIL' if failed else 'PASS'}", flush=True)
        if failed:
            failures.append(name)
            if test:
                raise SystemExit(f"Baseline correctness failed; stopped to avoid wasting GPU time. Log: {destination / (name + '.log')}")
            continue
        if not test:
            report = "nvfp4" if stage == "main" else f"nvfp4_{stage}"
            with (ROOT / f"profiles/runtime/{report}.csv").open() as file:
                rows.extend({**row, "custom_us": float(row["custom_us"]),
                             "reference_us": float(row["reference_us"]),
                             "reference_pct": float(row["reference_pct"])} for row in csv.DictReader(file))
            with (ROOT / f"profiles/runtime/{report}_samples.json").open() as file:
                samples.append(dict(stage=stage, **json.load(file)))
    if rows:
        save_report("nvfp4_batch", rows, samples)
    if failures:
        raise SystemExit(f"Failed stages: {', '.join(failures)}. Logs: {destination}")
