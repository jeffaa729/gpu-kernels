# gpu-kernels

CUDA/CuTe kernels for GEMM, FlashAttention forward, and distributed MoE, benchmarked against library implementations.

Times are in microseconds. **Reference % = reference time / custom time × 100.**

## Benchmarks

### RTX 4060

```bash
uv sync --locked
GPU_KERNELS_CUDA_ARCH=89 bash scripts/benchmark.sh matmul full --reference cublas
GPU_KERNELS_CUDA_ARCH=89 bash scripts/benchmark.sh flash_attention full --reference fa2
```

### H100 / H200

```bash
GPU_KERNELS_CUDA_ARCH=90 bash scripts/benchmark.sh matmul h100 --reference cublas
GPU_KERNELS_CUDA_ARCH=90 bash scripts/benchmark.sh flash_attention h100 --reference fa4
```

### B200 — NVFP4 GEMM

```bash
bash scripts/test.sh nvfp4
bash scripts/benchmark.sh nvfp4 b200 --reference cublaslt
bash scripts/benchmark.sh nvfp4 b200 --batch
```

Requires a B200 and CUDA Toolkit 13.1+. Packed E2M1 inputs, VEC16 UE4M3 scales, FP32 accumulation, and row-major FP16 output. `b200` runs 4096, 8192, and 16384; `quick` runs 8192. Both backends rotate identical operands across allocations totaling at least twice L2; preparation and a 100 ms per-backend settling interval are excluded from CUDA Graph timing. Results overwrite `profiles/runtime/nvfp4.{csv,md}`.

`--batch` first checks R0, runs R0–R6 at 8192 in separate processes, then runs the main kernel at all three sizes. A baseline failure stops the batch; later stage failures are logged without discarding other results. Summary: `profiles/runtime/nvfp4_batch.{csv,md}`; logs: `profiles/runtime/nvfp4_batch/`. Select one snapshot with `--stage r0` through `--stage r6` (default: `main`).

The complete kernels in `kernels/gemm/nvfp4/sm100.cu` and `ablation/sm100_nvfp4/` adapt [fast.cu](https://github.com/pranjalssh/fast.cu/tree/2dfe5e26aecfd9e5f27bf9d5837deea01acda24b/gb300/nvfp4) (MIT). B200 uses native K64 and relative shared descriptors: R0/R1 use K128 feeds; R2 adds K256 windows; R3 overlapping TMEM accumulators; R4 256-bit stores; R5 output cache policy; R6 exact block size. The main kernel additionally implements runtime L2-side ownership and the reference's 8×8 ordering for the 8192 cube. K96 tail stages do not apply. Existing SM89/SM90 kernels are unchanged.

### MegaMoE

```bash
export GPU_KERNELS_PYTHON=/path/to/venv/bin/python
export CUTLASS_DIR=/path/to/cutlass
bash scripts/benchmark_moe.sh 2 --trials 31
```

## GEMM — RTX 4060 Laptop (SM89)

FP32 and BF16 GEMM, TN layout, compared with cuBLAS.

| M=N=K | dtype | Custom µs | cuBLAS µs | Reference % |
| ---: | :--- | ---: | ---: | ---: |
| 2048 | FP32 | 2600.45 | 2546.98 | 97.9 |
| 2048 | BF16 | 755.92 | 644.02 | 85.2 |

## GEMM — H100 SXM (SM90)

BF16 GEMM, TN layout, compared with cuBLAS.

| M=N=K | Custom µs | cuBLAS µs | Reference % |
| ---: | ---: | ---: | ---: |
| 2048 | 30.53 | 27.59 | 90.4 |
| 4096 | 227.58 | 219.14 | 96.3 |
| 8192 | 1794.82 | 1807.82 | 100.7 |

## GEMM — B200 (SM100)

NVFP4 GEMM, TN layout, FP16 output. Median of nine CUDA Graph timing trials on 2026-10-19; CUDA Toolkit 13.1, cuBLASLt 13.2.2.2. Reference % = cuBLASLt time / custom time × 100. All stages pass numerical checks; the main kernel also passes both focused Compute Sanitizer memcheck cases with zero errors.

| M=N=K | Custom µs | cuBLASLt µs | Reference % |
| ---: | ---: | ---: | ---: |
| 4096 | 32.63 | 31.31 | 96.0 |
| 8192 | 229.56 | 225.77 | 98.3 |
| 16384 | 2236.12 | 1709.69 | 76.5 |

16384 remains untuned and noisy: custom samples ranged from 1.75 to 2.50 ms within the latest run. The table reports its median, not a stable performance claim.

### NVFP4 ablation — 8192³

Earlier 8192-only run, before enlarging the routing table for 16384. Each row adds one reference-derived technique to the previous stage. R0 already includes two-CTA MMA, persistent scheduling, warp specialization, TMA, and block-scale handling. The main kernel adds L2 ownership scheduling to R6; it does not improve this B200 run. Results are specific to this GPU and measurement protocol.

| Stage | Added technique | Custom µs | cuBLASLt µs | Reference % |
| :--- | :--- | ---: | ---: | ---: |
| R0 | K128 baseline | 439.33 | 224.77 | 51.2 |
| R1 | Direct `%laneid` | 264.45 | 227.23 | 85.9 |
| R2 | K256 operand windows | 243.94 | 226.02 | 92.7 |
| R3 | Overlapping TMEM accumulators | 228.71 | 222.87 | 97.4 |
| R4 | 256-bit output stores | 218.04 | 222.88 | 102.2 |
| R5 | Output cache hints | 216.94 | 222.13 | 102.4 |
| R6 | Exact 224-thread block size | 218.17 | 222.15 | 101.8 |
| Main | L2-side ownership and tile ordering | 218.56 | 223.17 | 102.1 |

### GEMM ablation figures

Roofline figures use measured runtimes, analytical minimum-I/O estimates, and modeled hardware ceilings—not measured DRAM traffic.

RTX 4060 Laptop: FP32 F0–F7 and BF16 T0–T4, square matrices from 256 to 8192.

<img src="assets/roofline/sm89_analytical.png" alt="RTX 4060 Laptop analytical GEMM rooflines, FP32 and BF16 versus cuBLAS" width="800">

H100: BF16 H0–H8, square matrices from 512 to 8192.

<img src="assets/roofline/sm90_analytical.png" alt="H100 analytical BF16 GEMM roofline, incremental optimization stages versus cuBLAS" width="800">

## FlashAttention forward — RTX 4060 Laptop (SM89)

Causal BF16 MHA: `B=1, T=512, Hq=Hkv=8, D=128`.

| Custom µs | FlashAttention-2 µs | Reference % |
| ---: | ---: | ---: |
| 32.06 | 34.16 | 106.5 |

## FlashAttention forward — H100 SXM (SM90)

Causal BF16 MHA: `B=4, Hq=Hkv=8, D=128`. Median of 35 CUDA Graph timing trials on 2026-10-16; reference: FlashAttention-4 `4.0.0b34`.

| T | Custom µs | FlashAttention-4 µs | Reference % |
| ---: | ---: | ---: | ---: |
| 2048 | 78.23 | 77.54 | 99.1 |
| 4096 | 265.06 | 263.87 | 99.5 |
| 8192 | 975.55 | 971.83 | 99.6 |
| 16384 | 3717.90 | 3690.64 | 99.3 |

### FlashAttention ablation figures

H100, cumulative stages A1–A8 versus FlashAttention-4: causal BF16 MHA, `B=4, H=8, D=128`, `T ∈ {512,1024,2048,4096,8192}`.

<img src="assets/attention/sm90_ablation.png" alt="H100 MHA A1-A8 ablation versus official FlashAttention-4" width="800">

Analytical minimum-I/O roofline using the same runtimes; nominal H100 SXM ceilings are not adjusted for the run's 525 W power limit.

<img src="assets/attention/sm90_roofline_analytical.png" alt="H100 SXM analytical minimum-I/O FlashAttention roofline, A1-A8 and official FA4" width="800">

## MegaMoE — 4× H200 NVL (EP4)

Four NVIDIA H200 NVL GPUs, NVLink `NV6` between every pair, 600 W power limit per GPU. BF16 inputs, weights, and outputs; FP32 accumulation. V4-Flash-shaped routed MoE layer.

| Parameter | Value |
| :--- | ---: |
| Hidden size (H) | 4096 |
| Expert intermediate size (I) | 2048 |
| Total experts (E) | 256 |
| Experts per GPU | 64 |
| Top-k experts per token | 6 |
| SwiGLU clamp | 10.0 |

Reference: DeepEP V2 dispatch/combine + DeepGEMM BF16 grouped GEMMs + weighted SwiGLU.

Measured on 2026-10-19. Complete routed MoE layer: median of 31 CUDA Graph timing trials, 10 operations per replay, using the slowest rank per sample.

| Tokens/GPU | Custom µs | DeepEP V2 + DeepGEMM µs | Reference % |
| ---: | ---: | ---: | ---: |
| 1 | 194.22 | 290.31 | 149.5 |
| 64 | 887.05 | 1068.29 | 120.4 |
| 256 | 910.07 | 1146.66 | 126.0 |
| 1024 | 1053.08 | 1440.81 | 136.8 |
| 2048 | 1852.50 | 2257.62 | 121.9 |
| 4096 | 3216.85 | 4013.21 | 124.8 |

## MegaMoE — 2× H200 NVL (EP2)

Two NVIDIA H200 NVL GPUs, NVLink `NV6`, 600 W power limit per GPU. BF16 inputs, weights, and outputs; FP32 accumulation. V4-Flash-shaped routed MoE: `H=4096, I=2048, E=256`, 128 experts/GPU, top-k=6, SwiGLU clamp=10.

Reference: DeepEP V2 dispatch/combine + DeepGEMM BF16 grouped GEMMs + weighted SwiGLU.

Measured on 2026-10-18. Complete routed MoE layer: median of 31 CUDA Graph timing trials, 10 operations per replay, using the slowest rank per sample.

| Tokens/GPU | Custom µs | DeepEP V2 + DeepGEMM µs | Reference % |
| ---: | ---: | ---: | ---: |
| 1 | 179.15 | 269.82 | 150.6 |
| 64 | 1662.81 | 2068.48 | 124.4 |
| 256 | 1739.71 | 2166.15 | 124.5 |
| 1024 | 1926.32 | 2481.00 | 128.8 |
| 2048 | 2024.23 | 2935.44 | 145.0 |
| 4096 | 3551.77 | 4588.19 | 129.2 |
