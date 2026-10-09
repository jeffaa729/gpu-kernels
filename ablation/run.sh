#!/usr/bin/env bash
set -euo pipefail
ablation_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$ablation_root")"
python_bin="${GPU_KERNELS_PYTHON:-$repo_root/.venv/bin/python}"
architecture="${GPU_KERNELS_CUDA_ARCH:-89}"
build_dir="$repo_root/build/ablation"
if [[ "$architecture" == 90 || "$architecture" == 90a ]]; then
    architecture=90a
    build_dir="$repo_root/build/ablation_sm90"
fi
mkdir -p "$build_dir"
cmake -S "$ablation_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DGPU_KERNELS_ABLATION_ARCH="$architecture" \
    -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}" >"$build_dir/build.log" 2>&1
if ! cmake --build "$build_dir" -j "${GPU_KERNELS_BUILD_JOBS:-4}" >>"$build_dir/build.log" 2>&1; then
    tail -n 100 "$build_dir/build.log" >&2
    exit 1
fi
uv pip install --python "$python_bin" -r "$ablation_root/pyproject.toml" --index-url https://pypi.org/simple >"$build_dir/plot_dependencies.log" 2>&1
"$python_bin" "$ablation_root/run.py" "$@"
