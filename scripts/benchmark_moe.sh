#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GPU_KERNELS_PYTHON:-$repo_root/.venv/bin/python}"
gpus="${1:-2}"
(( $# == 0 )) || shift
if [[ "$gpus" != 2 && "$gpus" != 4 && "$gpus" != 8 ]]; then
    echo "usage: bash scripts/benchmark_moe.sh [2|4|8] [--check] [--tokens ...]" >&2
    exit 1
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Set GPU_KERNELS_PYTHON to the Python environment with torch, DeepEP and DeepGEMM." >&2
    exit 1
fi
if ! "$python_bin" -c 'import torch, deep_ep, deep_gemm'; then
    echo "Install official DeepEP and DeepGEMM into $python_bin before this H100 run." >&2
    exit 1
fi
# DeepEP's supported single-node NVLink-only mode needs no RDMA/GIN device.
export EP_DISABLE_GIN="${EP_DISABLE_GIN:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
# DeepGEMM's wheel includes the CUTLASS/CuTe headers used by its JIT kernels.
export CUTLASS_DIR="${CUTLASS_DIR:-$("$python_bin" -c 'import pathlib, deep_gemm; print(pathlib.Path(deep_gemm.__file__).parent)')}"
mkdir -p "$repo_root/build"
if ! GPU_KERNELS_CUDA_ARCH=90 bash "$repo_root/scripts/build.sh" gpu_kernels_moe_swiglu_bench gpu_kernels_fused_moe_bench \
        >"$repo_root/build/moe_build.log" 2>&1; then
    cat "$repo_root/build/moe_build.log" >&2
    exit 1
fi
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$gpus" \
    benchmarks/moe_pipeline.py "$@"
