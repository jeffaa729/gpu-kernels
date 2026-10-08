#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GPU_KERNELS_PYTHON:-$repo_root/.venv/bin/python}"
gpus="${1:-2}"
(( $# == 0 )) || shift
if [[ "$gpus" != 2 && "$gpus" != 4 ]]; then
    echo "usage: bash scripts/benchmark_moe.sh [2|4] [--check] [--tokens ...]" >&2
    exit 1
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Set GPU_KERNELS_PYTHON to the Python environment with torch, DeepEP and DeepGEMM." >&2
    exit 1
fi
if ! "$python_bin" -c 'import torch; v = torch.cuda.nccl.version(); assert v >= (2, 30, 4), f"DeepEP V2 needs NCCL >= 2.30.4; found {v}"'; then
    exit 1
fi
if ! "$python_bin" -c 'import torch, deep_ep, deep_gemm'; then
    echo "Install official DeepEP and DeepGEMM into $python_bin before this H100 run." >&2
    exit 1
fi
mkdir -p "$repo_root/build"
if ! GPU_KERNELS_CUDA_ARCH=90 bash "$repo_root/scripts/build.sh" gpu_kernels_moe_swiglu_bench \
        >"$repo_root/build/moe_build.log" 2>&1; then
    cat "$repo_root/build/moe_build.log" >&2
    exit 1
fi
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$gpus" \
    benchmarks/moe_pipeline.py "$@"
