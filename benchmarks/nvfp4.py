"""B200 NVFP4 GEMM against cuBLASLt."""

import ctypes
import math

from common import I, P, Operation, bind, checked, library, pointers, stream, torch
from reference.python.nvfp4 import dequantize, quantize


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
                  [ctypes.POINTER(P)] + [I] * 3 + [P] * 5 + [ctypes.c_size_t, P])
    destroy = bind(lib, "gpu_kernels_nvfp4_plan_destroy", [P], None)
    reference = bind(lib, "gpu_kernels_nvfp4_cublas", [P] * 5)
    custom = bind(lib, "gpu_kernels_nvfp4_custom", [P] * 3)
    validate = bind(lib, "gpu_kernels_nvfp4_validate", [P])
    l2_bytes = bind(lib, "gpu_kernels_nvfp4_l2_bytes", [])()
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
                checked(lib, "nvfp4", create(ctypes.byref(context), m, n, k,
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
            operation = NVFP4Operation(f"M={m},N={n},K={k}", "nvfp4/fp16", "TN",
                                      functions, (expected,), 2e-3, 2e-3)
            operation.lib, operation.validate, operation.context = lib, validate, slots[0][0]
            operation.rotations = rotations
            operation.backend_settle_ms = 100
            yield operation
        finally:
            torch.cuda.current_stream().synchronize()
            for context, _ in slots:
                destroy(context)
