"""Graph-replayed BF16 Hopper MoE: custom fusion versus DeepEP V2 + DeepGEMM."""

import argparse
import ctypes
import gc
import hashlib
from importlib.metadata import version
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

import deep_ep
import deep_gemm
import torch
import torch.distributed as dist

from reference.python.moe import moe_forward
from benchmarks.fused_moe import CustomMoE


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SOURCE = "https://github.com/deepseek-ai/DeepGEMM/pull/316"
ROUTING_SOURCE = ("https://github.com/deepseek-ai/DeepGEMM/blob/"
                  "891d57b4db1071624b5c8fa0d1e51cb317fa709f/tests/test_mega_moe.py#L58-L64")
MODEL_PROFILES = {
    "v4-flash-bf16": dict(hidden=4096, intermediate=2048, experts=256, topk=6),
    "v4-pro-bf16": dict(hidden=7168, intermediate=3072, experts=384, topk=6),
}


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


def inputs(tokens, hidden, intermediate, experts, topk, rank, world, routing_weights="softmax", official_inputs=False):
    torch.manual_seed(rank if official_inputs else 2026 + rank)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts // world, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(experts // world, hidden, intermediate, device="cuda", dtype=torch.bfloat16)
    if not official_inputs:
        w1.div_(hidden ** .5)
        w2.div_(intermediate ** .5)
    scores = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
    values, ids = scores.topk(topk, dim=-1, sorted=False)
    # The pinned official test uses top-k Gaussian scores directly, without softmax.
    weights = values if routing_weights == "raw-topk" else values.softmax(-1)
    return x, ids, weights, w1.contiguous(), w2.contiguous()


def gather(tensor, world):
    pieces = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(pieces, tensor)
    return torch.cat(pieces)


def assert_close_all_ranks(actual, expected, *, atol, rtol, context):
    """Keep ranks on the same control path when a numerical assertion fails."""
    failure = None
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        failure = error
        print(f"[rank {dist.get_rank()}] {context}: {error}", file=sys.stderr, flush=True)
    failed = torch.tensor(int(failure is not None), dtype=torch.int32, device=actual.device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise AssertionError(f"{context}: numerical check failed on at least one rank") from failure


def check_output(actual, x, ids, weights, w1, w2, rank, world, clamp):
    expected = moe_forward(gather(x, world), gather(ids, world), gather(weights, world),
                           gather(w1, world), gather(w2, world), clamp)
    expected = expected[rank * x.shape[0]:(rank + 1) * x.shape[0]]
    assert_close_all_ranks(actual, expected, atol=0.05, rtol=0.05,
                           context=f"T={x.shape[0]} baseline/oracle")
    if rank == 0:
        print("MoE baseline: small distributed correctness check passed")


def run_case(args, tokens, rank, world, launch_swiglu, check):
    x, ids, weights, w1, w2 = inputs(tokens, args.hidden, args.intermediate, args.experts, args.topk,
                                   rank, world, args.routing_weights, official_inputs=args.profile in MODEL_PROFILES)
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    ep = deep_ep.ElasticBuffer(dist.group.WORLD, num_max_tokens_per_rank=tokens, hidden=args.hidden,
                               num_topk=args.topk, use_fp8_dispatch=False, explicitly_destroy=True,
                               allow_hybrid_mode=False,
                               allow_multiple_reduction=False, num_gpu_timeout_secs=10, num_cpu_timeout_secs=30)
    stats = torch.zeros(args.experts // world, device="cuda", dtype=torch.int32)

    scratch = None
    baseline_stages = {} if args.breakdown else None

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
        if baseline_stages is not None:
            baseline_stages.update(received=received, received_weights=received_weights, handle=handle)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(received, w1, l1, psum, compiled_dims="", use_psum_layout=True)
        launch_swiglu(l2_input, l1, received_weights, psum[-1:], args.clamp)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(l2_input, w2, l2, psum, compiled_dims="", use_psum_layout=True)
        return ep.combine(l2, handle=handle)[0]

    try:
        result = pipeline()
        torch.cuda.synchronize()
        expected = result.clone()
        if check:
            check_output(result, x, ids, weights, w1, w2, rank, world, args.clamp)
        custom = CustomMoE(x, ids, weights, w1, w2, args.clamp, dist.group.WORLD)
        for iteration in range(3):
            actual = custom()
            torch.cuda.synchronize()
            assert_close_all_ranks(actual, expected, atol=.02, rtol=.02,
                                   context=f"T={tokens} custom eager check {iteration + 1}")
        # One changing-input check catches stale pool readiness and route reuse.
        if check or args.check_only:
            ids.copy_((ids + 1) % args.experts)
            x.neg_()
            expected = pipeline().clone()
            actual = custom()
            torch.cuda.synchronize()
            assert_close_all_ranks(actual, expected, atol=.02, rtol=.02,
                                   context=f"T={tokens} custom changing-input check")
        error = (actual.float() - expected.float()).abs().max()
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        if rank == 0 and (args.check or args.check_only):
            print(f"T={tokens}: baseline/custom PASS, max_abs={error.item():.6f}", flush=True)
        if args.check_only:
            return None
        for _ in range(args.warmup):
            pipeline()
            custom()
        torch.cuda.synchronize()
        graphs, graph_outputs = [], []
        for function in (pipeline, custom):
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                for _ in range(args.repeats):
                    captured_output = function()
            graphs.append(graph)
            graph_outputs.append(captured_output)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        samples = [[], []]
        for trial in range(args.trials):
            for backend in ((0, 1) if trial % 2 == 0 else (1, 0)):
                dist.barrier()
                start.record()
                graphs[backend].replay()
                stop.record()
                stop.synchronize()
                elapsed = torch.tensor(start.elapsed_time(stop) * 1000 / args.repeats, device="cuda")
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                samples[backend].append(elapsed.item())
        for backend, output in zip(("baseline", "custom"), graph_outputs):
            assert_close_all_ranks(output, expected, atol=.02, rtol=.02,
                                   context=f"T={tokens} {backend} graph check")
        times = [statistics.median(values) for values in samples]
        breakdown_us = {} if args.breakdown else None
        if args.breakdown:
            # Retain the last dispatch tensors/handle and reuse the pipeline's
            # scratch. These compute-only replays allocate no new payload buffers.
            received = baseline_stages["received"]
            received_weights = baseline_stages["received_weights"]
            handle = baseline_stages["handle"]
            psum = handle.psum_num_recv_tokens_per_expert
            l1, l2_input, l2 = scratch

            def baseline_l1():
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                    received, w1, l1, psum, compiled_dims="", use_psum_layout=True)

            def baseline_swiglu():
                launch_swiglu(l2_input, l1, received_weights, psum[-1:], args.clamp)

            def baseline_l2():
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                    l2_input, w2, l2, psum, compiled_dims="", use_psum_layout=True)

            def phase1_sample():
                custom.ready.zero_()
                custom.phase1()

            if rank == 0:
                print("  Breakdown: isolated repeated stages on fixed buffers; "
                      "non-additive timings, not evidence of overlap", flush=True)
            for name, function in (("baseline L1", baseline_l1), ("baseline SwiGLU", baseline_swiglu),
                                   ("baseline L2", baseline_l2),
                                   ("custom copies", custom.copy_inputs), ("custom barrier", custom.barrier),
                                   ("custom routing", custom.metadata), ("custom L1+SwiGLU+reset", phase1_sample),
                                   ("custom L2+scatter", custom.phase2), ("custom reduce", custom.reduce)):
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.repeats):
                        function()
                start.record()
                graph.replay()
                stop.record()
                stop.synchronize()
                elapsed = torch.tensor(start.elapsed_time(stop) * 1000 / args.repeats, device="cuda")
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                if rank == 0:
                    breakdown_us[name] = elapsed.item()
                    print(f"  {name}: {breakdown_us[name]:.2f} us", flush=True)
        return times, samples, error.item(), breakdown_us
    finally:
        # Collective cleanup is only safe when no rank is unwinding a failure.
        if sys.exc_info()[0] is None:
            dist.barrier()
            ep.destroy()


def write_report(args, world, rows, source_sha256):
    """Checkpoint completed cases atomically, outside all measured regions."""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    label = (f"{args.profile}: BF16 EP{world} adaptation; not an official speedup reproduction"
             if args.profile in MODEL_PROFILES else "token-sweep" if args.profile == "sweep" else None)
    report = dict(config=vars(args), run_id=output.stem, profile_label=label,
                  profile_source=PROFILE_SOURCE if args.profile in MODEL_PROFILES else None,
                  routing_source=ROUTING_SOURCE if args.routing_weights == "raw-topk" else None,
                  adaptation=("Official rank seed and unscaled Gaussian BF16 inputs; FP8/FP4 casts omitted"
                              if args.profile in MODEL_PROFILES else "Local rank seed and scaled BF16 expert weights"),
                  gpu=torch.cuda.get_device_name(), world=world,
                  torch=torch.__version__, cuda=torch.version.cuda, deep_ep=version("deep_ep"),
                  deep_gemm=deep_gemm.__version__,
                  kernel_sha256=source_sha256["kernels/moe/fused/sm90.cu"],
                  source_sha256=source_sha256,
                  timing="CUDA Graph, all pipeline stages, max rank per sample, median across trials",
                  memory="Peak PyTorch allocated bytes, max rank, whole case including setup/checks/both backends/graphs",
                  rows=rows)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+",
                        help="tokens per rank (default: 1 64 256 1024 2048 4096 for V4 presets; 128 512 for small checks)")
    parser.add_argument("--profile", choices=["sweep", *MODEL_PROFILES],
                        help="default: v4-flash-bf16 for benchmarks; small shapes for --check/--check-only; sweep: tokens only")
    parser.add_argument("--output", default=str(ROOT / "profiles/runtime/moe.json"),
                        help="JSON report path; use a separate filename for each ablation")
    parser.add_argument("--resume", action="store_true", help="resume compatible completed cases from --output")
    parser.add_argument("--routing-weights", choices=["softmax", "raw-topk"],
                        help="default: raw-topk for V4 presets, softmax otherwise; raw-topk matches pinned test")
    parser.add_argument("--max-memory-gib", type=float,
                        help="stop after saving a case exceeding this peak allocated GiB; not an OOM preflight")
    parser.add_argument("--hidden", type=int, help="overrides profile hidden size")
    parser.add_argument("--intermediate", type=int, help="overrides profile expert intermediate size")
    parser.add_argument("--experts", type=int, help="overrides profile total expert count")
    parser.add_argument("--topk", type=int, help="overrides profile experts selected per token")
    parser.add_argument("--clamp", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=10, help="pipeline calls per graph replay")
    parser.add_argument("--breakdown", action="store_true",
                        help="also measure isolated baseline L1/SwiGLU/L2 and custom phases (non-additive)")
    parser.add_argument("--check", action="store_true", help="check the first case against the small PyTorch oracle")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.profile is None and not (args.check or args.check_only):
        args.profile = "v4-flash-bf16"
    if args.tokens is None:
        args.tokens = [1, 64, 256, 1024, 2048, 4096] if args.profile else [128, 512]
    defaults = MODEL_PROFILES.get(args.profile, dict(hidden=1024, intermediate=512, experts=16, topk=4))
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.routing_weights is None:
        args.routing_weights = "raw-topk" if args.profile in MODEL_PROFILES else "softmax"
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world not in (2, 4, 8) or args.experts % world or args.topk > args.experts:
        parser.error("use 2, 4 or 8 GPUs, experts divisible by GPUs, and topk <= experts")
    source_sha256 = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
        "kernels/moe/fused/sm90.cu", "kernels/moe/fused/common.cuh",
        "include/fused_moe.h", "benchmarks/fused_moe.py",
        "benchmarks/moe_pipeline.py", "benchmarks/bindings/fused_moe_bridge.cu",
        "kernels/moe/swiglu.cu", "benchmarks/bindings/moe_swiglu_bridge.cu",
        "CMakeLists.txt", "scripts/build.sh", "scripts/benchmark_moe.sh",
    )}
    rows = []
    if args.resume:
        previous = json.loads(Path(args.output).read_text())
        # Token selection and the stopping limit may change when continuing a run.
        ignored = {"tokens", "output", "resume", "max_memory_gib"}
        config = {key: value for key, value in vars(args).items() if key not in ignored}
        saved = {key: value for key, value in previous["config"].items() if key not in ignored}
        if config != saved or previous["world"] != world or previous.get("source_sha256") != source_sha256:
            parser.error("--resume requires matching benchmark settings, world size and complete source manifest")
        rows = previous["rows"]
    torch.cuda.set_device(local_rank)
    # DeepEP reuses NCCL's communicator pointer; initialize it eagerly.
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    launch_swiglu = swiglu_bridge()
    try:
        if rank == 0 and args.profile == "sweep":
            print(f"Profile: token-sweep (not an official model preset), tokens/rank={args.tokens}", flush=True)
        elif rank == 0 and args.profile in MODEL_PROFILES:
            print(f"Profile: {args.profile}, BF16 EP{world} adaptation, not an official speedup reproduction; "
                  f"H={args.hidden} I={args.intermediate} E={args.experts} topk={args.topk}, "
                  f"routing={args.routing_weights}, tokens/rank={args.tokens}", flush=True)
        completed = {row["tokens"] for row in rows}
        for index, tokens in enumerate(args.tokens[:1] if args.check_only else args.tokens):
            if tokens in completed and not args.check_only:
                if rank == 0:
                    print(f"T={tokens}: already saved, skipping", flush=True)
                continue
            if rank == 0:
                print(f"T={tokens}: starting", flush=True)
            torch.cuda.reset_peak_memory_stats()
            elapsed = run_case(args, tokens, rank, world, launch_swiglu,
                               check=index == 0 and (args.check or args.check_only))
            # Measure the whole case, including setup, validation and both graphs.
            # These statistics cover PyTorch allocations, not total device usage.
            torch.cuda.synchronize()
            peak = torch.tensor(torch.cuda.max_memory_allocated(), dtype=torch.int64, device="cuda")
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            peak_allocated_bytes = peak.item()
            del peak
            # run_case has returned: graph/scratch locals can now be reclaimed.
            gc.collect()
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"T={tokens}: peak allocated {peak_allocated_bytes / 2**30:.3f} GiB "
                      "(max rank, whole case)", flush=True)
            if rank == 0 and elapsed is not None:
                times, samples, error, breakdown_us = elapsed
                rows.append(dict(tokens=tokens, baseline_us=times[0], custom_us=times[1],
                                 reference_percent=100 * times[0] / times[1], samples=samples, max_abs=error,
                                 peak_allocated_bytes=peak_allocated_bytes))
                if breakdown_us is not None:
                    rows[-1].update(breakdown_us=breakdown_us,
                                    breakdown_label="Isolated repeated stages; non-additive, not evidence of overlap")
                write_report(args, world, rows, source_sha256)
                print(f"T={tokens}: custom {times[1]:.2f} us, DeepEP+DeepGEMM {times[0]:.2f} us, "
                      f"reference {100 * times[0] / times[1]:.1f}%; saved {args.output}", flush=True)
            # Finish the checkpoint before any rank allocates the next large case.
            dist.barrier()
            if args.max_memory_gib is not None and peak_allocated_bytes > args.max_memory_gib * 2**30:
                if rank == 0:
                    print("Peak allocated-memory limit exceeded; stopping after this case", flush=True)
                break
        if rank == 0 and rows:
            print("| GPUs | tokens/rank | H | I | E | topk | custom us | DeepEP+DeepGEMM us | reference % |")
            print("| ---: | ----------: | ---: | ---: | ---: | ---: | --------: | ----------------: | ----------: |")
            for row in rows:
                print(f"| {world} | {row['tokens']} | {args.hidden} | {args.intermediate} | {args.experts} | {args.topk} | "
                      f"{row['custom_us']:.2f} | {row['baseline_us']:.2f} | {row['reference_percent']:.1f} |")
    finally:
        if sys.exc_info()[0] is None:
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(f"[rank {os.environ.get('RANK', '?')}] benchmark failed", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        # Avoid Python/CUDA/NCCL finalizers on a broken rank; torchrun stops peers.
        os._exit(1)
