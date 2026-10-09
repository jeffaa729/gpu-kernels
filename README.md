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
- SM90: raw CUDA/PTX TMA/WGMMA forward; the optimized H100 results are below.

### Optimized H100 against official FlashAttention

Measured on the same NVIDIA H100 80GB HBM3, using the unchanged 30-shape suite:
`B ∈ {1,4}`, `T ∈ {128,256,512,1024,2048}`, `Hq=8`,
`Hkv ∈ {8,2,1}`, `D=128`. All backends use causal BF16 inputs/output
and FP32 natural-log LSE. CUDA Graph replay excludes setup and Python API latency;
results are medians of 15 trials, after 1 second of warmup.

The original kernel took 43.65 µs at `B=1,T=512,Hkv=8` and 980.52 µs at
`B=4,T=2048,Hkv=8`. The optimized kernel takes 10.06 µs and 79.72 µs:
approximately 4.3× and 12.3× faster on this H100.

| size | dtype | operation | custom µs | reference (FA3) µs | reference % |
| :--- | :--- | :--- | ---: | ---: | ---: |
| B=1,T=512,Hkv=8 | BF16 | forward | 10.06 | 11.03 | 109.6 |
| B=1,T=512,Hkv=2 | BF16 | forward | 9.90 | 10.90 | 110.1 |
| B=1,T=512,Hkv=1 | BF16 | forward | 9.94 | 10.96 | 110.3 |
| B=4,T=2048,Hkv=8 | BF16 | forward | 79.72 | 68.71 | 86.2 |
| B=4,T=2048,Hkv=2 | BF16 | forward | 74.80 | 64.14 | 85.7 |
| B=4,T=2048,Hkv=1 | BF16 | forward | 71.77 | 64.58 | 90.0 |

The ≥95% FA3 throughput target is reached on **24/30 shapes**, not all shapes.
The six `B=4,T≥1024` cases remain at 85.7–94.9%. Against FA2, all 30 cases
exceed 100%; FA2 is not the performance target for the Hopper kernel.

FA4's Hopper implementation is a separate reference, not its Blackwell kernel:

| size | dtype | operation | custom µs | reference (FA4) µs | reference % |
| :--- | :--- | :--- | ---: | ---: | ---: |
| B=1,T=512,Hkv=8 | BF16 | forward | 10.06 | 9.37 | 93.1 |
| B=1,T=512,Hkv=2 | BF16 | forward | 9.90 | 9.26 | 93.5 |
| B=1,T=512,Hkv=1 | BF16 | forward | 9.94 | 9.28 | 93.4 |
| B=4,T=2048,Hkv=8 | BF16 | forward | 79.72 | 83.00 | 104.1 |
| B=4,T=2048,Hkv=2 | BF16 | forward | 74.80 | 77.89 | 104.1 |
| B=4,T=2048,Hkv=1 | BF16 | forward | 71.77 | 77.49 | 108.0 |

FA4 throughput ratios span 92.2–108.0%; 13/30 cases reach ≥95%. These results
do not establish uniform parity with either Hopper reference.

### Source-traced Hopper implementation

[`sm90_forward.cu`](kernels/attention/flash_attention/sm90_forward.cu) retains
one fixed D128 kernel. Comments above each optimization identify its official
file/function at commit
[`94e22c9`](https://github.com/Dao-AILab/flash-attention/tree/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c).

- FA3 `tile_size.h`: 128×128 tiling; two consumer warp groups share K/V.
- FA3 `mainloop_fwd_sm90_tma_gmma_ws.hpp`: independent two-stage K/V TMA
  pipelines, QK shared/shared WGMMA, register/shared PV, and no BF16 V transpose.
- FA3 `flash_fwd_kernel_sm90.h`: producer/consumer register redistribution
  and descriptor prefetch. `mainloop::mma` supplies ping-pong scheduling and
  the source-level QK(next)/PV(current)/softmax pipeline.
- FA3 `softmax.h` and `utils.h`: base-2 online softmax, final-only row-sum
  reduction, causal-mask specialization, and register-layout BF16 packing.
- FA3 `epilogue_fwd.hpp`: warp-cooperative `stmatrix` shared-memory stores
  followed by a TMA output store.
- FA4 `cute/tile_scheduler.py::SingleTileLPTScheduler`: longest causal query
  tiles first, with batch/head locality.

The [FA3 paper](https://arxiv.org/abs/2407.08608) describes warp specialization,
ping-pong scheduling and intra-warpgroup pipelining. The
[FA4 paper](https://arxiv.org/abs/2603.05451) and official
`cute/flash_fwd_sm90.py` are used only where applicable to Hopper; Blackwell
TMEM, `tcgen05`, and two-CTA MMA are not ported.

All 30 shapes pass output/LSE checks against the FP32 PyTorch oracle. The four
existing smoke-test shapes also pass with all references. Native MHA memcheck
at `T=64` and `B=4,T=2048` reports zero errors; racecheck at the latter shape
reports zero hazards. No Nsight-counter or measured-overlap claims are made.

The saved run is in the generated, Git-ignored
[`profiles/flash_attention_optimization/`](profiles/flash_attention_optimization/):
`final.csv`, timing samples, exact baseline/final source snapshots, environment
metadata, and sanitizer logs. GEMM, SM89 and MoE kernels are unchanged.

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
GPU_KERNELS_CUDA_ARCH=90 bash scripts/benchmark.sh flash_attention h100 --reference fa2
GPU_KERNELS_CUDA_ARCH=90 bash scripts/benchmark.sh flash_attention h100 --reference fa3
GPU_KERNELS_CUDA_ARCH=90 bash scripts/benchmark.sh flash_attention h100 --reference all --trials 15 --sample-ms 40
bash scripts/benchmark_moe.sh 2 --check
```

`all` covers the single-GPU GEMM and FlashAttention suites. The distributed MoE
baseline has its own launcher. Runtime reports are overwritten in
`profiles/runtime/`; Nsight reports use `bash scripts/profile.sh FAMILY quick`.
