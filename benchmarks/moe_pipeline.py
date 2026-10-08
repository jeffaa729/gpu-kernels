"""Unfused BF16 expert-parallel MoE baseline on two or four H100 GPUs."""

import argparse
import ctypes
import os
import statistics
from pathlib import Path

import deep_ep
import deep_gemm
import torch
import torch.distributed as dist

from reference.python.moe import moe_forward


ROOT = Path(__file__).resolve().parents[1]


def swiglu_bridge():
    library = ctypes.CDLL(str(ROOT / "build/libgpu_kernels_moe_swiglu_bench.so"))
    launch = library.gpu_kernels_moe_swiglu
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_float, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    library.gpu_kernels_moe_last_error.restype = ctypes.c_char_p

    def call(output, gate_up, weights, valid_rows, clamp):
        status = launch(output.data_ptr(), gate_up.data_ptr(), weights.data_ptr(), valid_rows.data_ptr(),
                        gate_up.shape[0], output.shape[1], clamp, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(library.gpu_kernels_moe_last_error().decode())

    return call


def inputs(tokens, hidden, intermediate, experts, topk, rank, world):
    torch.manual_seed(2026 + rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts // world, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16) / hidden ** .5
    w2 = torch.randn(experts // world, hidden, intermediate, device="cuda", dtype=torch.bfloat16) / intermediate ** .5
    scores = torch.randn(tokens, experts, device="cuda")
    values, ids = scores.topk(topk, dim=-1, sorted=False)
    weights = values.softmax(-1)
    return x, ids, weights, w1.contiguous(), w2.contiguous()


def gather(tensor, world):
    pieces = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(pieces, tensor)
    return torch.cat(pieces)


def check_output(actual, x, ids, weights, w1, w2, rank, world, clamp):
    expected = moe_forward(gather(x, world), gather(ids, world), gather(weights, world),
                           gather(w1, world), gather(w2, world), clamp)
    expected = expected[rank * x.shape[0]:(rank + 1) * x.shape[0]]
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
    if rank == 0:
        print("MoE baseline: small distributed correctness check passed")


def run_case(args, tokens, rank, world, launch_swiglu, check):
    x, ids, weights, w1, w2 = inputs(tokens, args.hidden, args.intermediate, args.experts, args.topk, rank, world)
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    ep = deep_ep.ElasticBuffer(dist.group.WORLD, num_max_tokens_per_rank=tokens, hidden=args.hidden,
                               num_topk=args.topk, use_fp8_dispatch=False, explicitly_destroy=True,
                               allow_multiple_reduction=False, num_gpu_timeout_secs=10, num_cpu_timeout_secs=30)
    stats = torch.zeros(args.experts // world, device="cuda", dtype=torch.int32)

    scratch = None

    def pipeline():
        nonlocal scratch
        stats.zero_()
        received, _, received_weights, handle, _ = ep.dispatch(
            x, topk_idx=ids, topk_weights=weights, cumulative_local_expert_recv_stats=stats,
            num_experts=args.experts, expert_alignment=alignment,
            do_cpu_sync=False, do_handle_copy=False, do_expand=True)
        rows = received.shape[0]
        if scratch is None:
            scratch = (torch.empty(rows, 2 * args.intermediate, device="cuda", dtype=torch.bfloat16),
                       torch.empty(rows, args.intermediate, device="cuda", dtype=torch.bfloat16),
                       torch.empty(rows, args.hidden, device="cuda", dtype=torch.bfloat16))
        l1, l2_input, l2 = scratch
        psum = handle.psum_num_recv_tokens_per_expert
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(received, w1, l1, psum, compiled_dims="", use_psum_layout=True)
        launch_swiglu(l2_input, l1, received_weights, psum[-1:], args.clamp)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(l2_input, w2, l2, psum, compiled_dims="", use_psum_layout=True)
        return ep.combine(l2, handle=handle)[0]

    try:
        result = pipeline()
        torch.cuda.synchronize()
        if check:
            check_output(result, x, ids, weights, w1, w2, rank, world, args.clamp)
        if args.check_only:
            return None
        for _ in range(args.warmup):
            pipeline()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(args.trials):
            dist.barrier()
            start.record()
            pipeline()
            stop.record()
            stop.synchronize()
            elapsed = torch.tensor(start.elapsed_time(stop) * 1000, device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.item())
        return statistics.median(samples)
    finally:
        dist.barrier()
        ep.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 128, 512, 2048, 4096])
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--clamp", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--check", action="store_true", help="check the first case against the small PyTorch oracle")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world not in (2, 4) or args.experts % world or args.topk > args.experts:
        parser.error("use 2 or 4 GPUs, experts divisible by GPUs, and topk <= experts")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    launch_swiglu = swiglu_bridge()
    try:
        rows = []
        for index, tokens in enumerate(args.tokens[:1] if args.check_only else args.tokens):
            elapsed = run_case(args, tokens, rank, world, launch_swiglu,
                               check=index == 0 and (args.check or args.check_only))
            if rank == 0 and elapsed is not None:
                rows.append((tokens, elapsed))
        if rank == 0 and rows:
            print("| GPUs | tokens/rank | H | I | E | topk | unfused us |")
            print("| ---: | ----------: | ---: | ---: | ---: | ---: | ---------: |")
            for tokens, elapsed in rows:
                print(f"| {world} | {tokens} | {args.hidden} | {args.intermediate} | {args.experts} | {args.topk} | {elapsed:.2f} |")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
