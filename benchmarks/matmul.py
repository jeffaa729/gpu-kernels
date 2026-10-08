"""K-contiguous A/B and column-major C, matching fast.cu and cuBLAS TN."""

from common import I, P, Operation, bind, checked, library, pointers, stream, torch


def cases(args, family):
    is_sm90 = torch.cuda.get_device_capability() == (9, 0)
    if args.suite == "h100" and not is_sm90:
        raise ValueError("The H100 GEMM suite requires an SM90 GPU.")
    reference = args.reference or "cublas"
    if reference != "cublas":
        raise ValueError("GEMM reference: cublas")

    lib = library("operator")
    checked(lib, "operator", bind(lib, "gpu_kernels_cublas_init", [])())
    gemm = bind(lib, "gpu_kernels_gemm", [P] * 3 + [I] * 5 + [P])
    dtypes = (torch.bfloat16,) if is_sm90 else (torch.float32, torch.bfloat16)

    try:
        for dtype in dtypes:
            if args.test and dtype == torch.bfloat16 and is_sm90:
                # Exercise persistent reuse and cross-tile queue wraps.
                shapes = ((2048, 2048, 64), (2048, 4096, 192),
                          (4096, 2048, 256))
            elif args.test:
                shapes = ((128, 256, 64), (640, 128, 128), (1152, 128, 64),
                          (256, 512, 192))
                if dtype == torch.float32:
                    shapes += ((17, 33, 65),)
            else:
                sizes = (2048,) if args.suite == "quick" else (2048, 4096, 8192)
                shapes = tuple((n, n, n) for n in sizes)

            for m, n, k in shapes:
                left = torch.randn((m, k), device="cuda", dtype=dtype) * .1
                right = torch.randn((n, k), device="cuda", dtype=dtype) * .1
                right = right.t()  # Logical B[K,N], physical [N,K].
                expected = (left.float() @ right.float()).to(dtype)
                # NaN initialization catches unwritten output elements before timing.
                custom_output = torch.full((n, m), float("nan"), device="cuda", dtype=dtype)
                custom_output = custom_output.t()  # Column-major C[M,N].

                def native(output, use_reference):
                    checked(lib, "operator", gemm(
                        *pointers((output, left, right)),
                        m, n, k, int(dtype == torch.bfloat16),
                        use_reference, stream()))
                    return output

                functions = {
                    "custom": lambda output=custom_output: native(output, 0)
                }
                cublas_output = torch.empty_like(custom_output)
                functions["cuBLAS"] = (
                    lambda output=cublas_output: native(output, 1))

                tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-4
                yield Operation(
                    f"M={m},N={n},K={k}",
                    "bf16" if dtype == torch.bfloat16 else "fp32",
                    "TN", functions, (expected,), tolerance, tolerance)
    finally:
        bind(lib, "gpu_kernels_cublas_destroy", [], None)()
