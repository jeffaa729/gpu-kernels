# gpu-kernels

Hand-written CUDA kernels benchmarked against established libraries. Times are in microseconds; **reference % = reference time / custom time × 100**. Results below are from the latest matching runs recorded in this project.

## GEMM — SM89 (RTX 4060 Laptop)

[`kernels/gemm/matmul/sm89.cu`](kernels/gemm/matmul/sm89.cu) implements FP32 CUDA-core and BF16 Tensor Core GEMM (TN layout).

| M=N=K | dtype | custom µs | cuBLAS µs | reference % |
| ---: | :--- | ---: | ---: | ---: |
| 2048 | FP32 | 2600.45 | 2546.98 | 97.9 |
| 2048 | BF16 | 755.92 | 644.02 | 85.2 |

- FP32: block/warp/thread tiling, register accumulation, `float4` loads, shared-memory double buffering.
- BF16: `mma.sync`, `ldmatrix`, XOR swizzling, two-stage `cp.async` buffering.

## GEMM — SM90 (H100 SXM)

[`kernels/gemm/matmul/sm90.cu`](kernels/gemm/matmul/sm90.cu) implements BF16 persistent WGMMA GEMM. These are the latest reported results for the current Kernel 10 version.

| M=N=K | custom µs | cuBLAS µs | reference % |
| ---: | ---: | ---: | ---: |
| 2048 | 30.53 | 27.59 | 90.4 |
| 4096 | 227.58 | 219.14 | 96.3 |
| 8192 | 1794.82 | 1807.82 | 100.7 |

- TMA loads/stores, WGMMA, three-stage shared-memory pipeline, persistent tile scheduling, shared-memory output epilogue.

## GEMM ablation: analytical rooflines

Each curve compares incremental kernel optimizations with cuBLAS. Throughput comes
from measured CUDA Graph runtimes; arithmetic intensity uses minimum algorithmic
IO, not measured DRAM traffic. The roofs use theoretical bandwidth and dense
compute throughput at the sampled active GPU clock. Matrix sizes label cuBLAS
points; these plots do not establish actual memory traffic or bottlenecks.

### SM89: RTX 4060 Laptop

FP32 stages F0-F7 and BF16 stages T0-T4 cover tiling, vectorized access, swizzling,
and asynchronous pipelines, with square matrices from 256 to 8192. Stage files
are in [`ablation/sm89_fp32/`](ablation/sm89_fp32/) and
[`ablation/sm89_bf16/`](ablation/sm89_bf16/).

![RTX 4060 Laptop analytical GEMM rooflines, FP32 and BF16 versus cuBLAS](assets/roofline/sm89_analytical.png)

### SM90: H100

BF16 stages H0-H8 cover TMA/WGMMA, warp specialization, register redistribution,
persistent scheduling, L2-friendly scheduling, wider tiles, accumulator overwrite,
and TMA output stores, with square matrices from 512 to 8192. Stage files are in
[`ablation/sm90_bf16/`](ablation/sm90_bf16/). Some transitions change tile size and
queue depth together; improvements are not necessarily monotonic.

![H100 analytical BF16 GEMM roofline, incremental optimization stages versus cuBLAS](assets/roofline/sm90_analytical.png)

## FlashAttention forward

[`kernels/attention/flash_attention/`](kernels/attention/flash_attention/) contains causal BF16, head-dimension-128 MHA/GQA/MQA kernels for SM89 and SM90. Latest reported RTX 4060 run: `B=1, T=512, Hq=8, D=128`.

| Hkv | custom µs | FlashAttention-2 µs | reference % |
| ---: | ---: | ---: | ---: |
| 8 (MHA) | 32.06 | 34.16 | 106.5 |
| 2 (GQA) | 32.05 | 34.26 | 106.9 |
| 1 (MQA) | 32.68 | 34.60 | 105.9 |

- SM89: fused QK/online-softmax/PV, causal-tile specialization, `cp.async` prefetch, `ldmatrix`.
- SM90: TMA, mbarriers, WGMMA, staged K/V tiles. H100 timing is not yet verified.

## MegaMoE

The current code is the **unfused BF16 baseline**, using DeepEP dispatch/combine,
two DeepGEMM expert GEMMs, and a custom CUDA SwiGLU kernel. The fused persistent
MegaMoE kernel is not implemented yet. Distributed runs require two or four H100
GPUs on one NVLink-connected node with DeepEP and DeepGEMM installed.

## Run

```bash
uv sync --locked
bash scripts/test.sh all
bash scripts/benchmark.sh matmul quick
bash scripts/benchmark.sh flash_attention quick
GPU_KERNELS_CUDA_ARCH=90 bash scripts/benchmark.sh matmul h100
GPU_KERNELS_CUDA_ARCH=90 bash scripts/benchmark.sh flash_attention h100 --reference fa3
bash scripts/benchmark_moe.sh 2 --check
```

`all` covers the single-GPU GEMM and FlashAttention suites. The distributed MoE
baseline has its own launcher. Runtime reports are overwritten in
`profiles/runtime/`; Nsight reports use `bash scripts/profile.sh FAMILY quick`.
