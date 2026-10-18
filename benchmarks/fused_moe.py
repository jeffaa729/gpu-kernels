"""BF16 Hopper MoE with PyTorch symmetric peer memory and a ctypes bridge."""

import ctypes
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


class MoePeers(ctypes.Structure):
    _fields_ = ([(name, ctypes.c_void_p * 8) for name in ("x", "ids", "weights", "output")]
                + [(name, ctypes.c_void_p) for name in ("pool", "route_weights", "ready", "offsets", "carrier_experts", "carrier_rows")])


class MoeShape(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in (
        "tokens", "hidden", "intermediate", "experts", "topk", "rank", "world", "capacity"
    )] + [("clamp", ctypes.c_float), ("pool_capacity", ctypes.c_int64)]


class CustomMoE:
    """Allocate once, then invoke collectively on one ordered stream per rank.

    x[T,H], ids/weights[T,K], w1[E/world,2I,H], w2[E/world,H,I]. H/I are
    multiples of 256, with at most 256 experts/rank. IDs are valid global experts,
    distinct within each token.
    Inputs can change in place; W1 is prepacked once and must remain fixed.
    Keep this object alive until GPU work finishes;
    the returned output is overwritten by the next call. Only two ranks tested.
    """

    def __init__(self, x, ids, weights, w1, w2, clamp, group):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        if self.world not in (2, 4, 8):
            raise ValueError("CustomMoE requires 2, 4 or 8 ranks")
        if x.ndim != 2 or ids.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
            raise ValueError("expected x/ids matrices and w1/w2 three-dimensional tensors")
        tokens, hidden = x.shape
        local_experts, _, intermediate = w2.shape
        experts, topk = local_experts * self.world, ids.shape[1]
        if (tokens <= 0 or hidden <= 0 or hidden % 256 or intermediate <= 0
                or intermediate % 256 or not 1 <= local_experts <= 256 or not 1 <= topk <= experts):
            raise ValueError("require T,I,H > 0, H/I divisible by 256, 1 <= experts/rank <= 256, and 1 <= K <= E")
        if x.device.type != "cuda":
            raise ValueError("inputs must be CUDA tensors")
        self.device = x.device
        self._input_specs = (
            ("x", (tokens, hidden), torch.bfloat16),
            ("ids", (tokens, topk), torch.int64),
            ("weights", (tokens, topk), torch.float32),
        )
        for tensor, spec in zip((x, ids, weights), self._input_specs):
            self._check_tensor(tensor, *spec)
        self._check_tensor(w1, "w1", (local_experts, 2 * intermediate, hidden), torch.bfloat16)
        self._check_tensor(w2, "w2", (local_experts, hidden, intermediate), torch.bfloat16)
        self.x, self.ids, self.weights = x, ids, weights
        self.w1, self.w2 = w1, w2
        # Match sm90.cu::moe_block_m, including the pipeline's row alignment.
        block_m = 64 if tokens <= 512 else 128
        self.capacity = ((self.world * tokens + block_m - 1) // block_m) * block_m
        # DeepGEMM layout/mega_moe.cuh::get_num_max_pool_tokens: total routes
        # to this rank plus per-expert padding using the selected BM;
        # this is a full compact pool, not DeepGEMM's reusable ring protocol.
        max_rows = self.world * tokens * min(topk, local_experts) + local_experts * (block_m - 1)
        self.pool_capacity = ((max_rows + block_m - 1) // block_m) * block_m
        # Metadata and scheduling still use int32; payload addresses use int64.
        if max(tokens, hidden, intermediate, experts, self.capacity,
               self.world * tokens * topk + 255, local_experts * self.capacity,
               (self.pool_capacity // block_m) * max(intermediate // 64, hidden // 128)) > 2**31 - 1:
            raise ValueError("routing metadata or tile schedule exceeds int32 indexing")
        self.shape = MoeShape(tokens, hidden, intermediate, experts, topk,
                              self.rank, self.world, self.capacity, clamp, self.pool_capacity)

        library_path = Path(__file__).resolve().parents[1] / "build/libgpu_kernels_fused_moe_bench.so"
        self.library = ctypes.CDLL(str(library_path))
        self._launch = self.library.gpu_kernels_moe_phase
        self._launch.argtypes = [
            ctypes.c_int, ctypes.POINTER(MoePeers), ctypes.POINTER(MoeShape),
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_void_p,
        ]
        self._launch.restype = ctypes.c_int
        self._last_error = self.library.gpu_kernels_fused_moe_last_error
        self._last_error.argtypes = []
        self._last_error.restype = ctypes.c_char_p

        with torch.cuda.device(self.device):
            # DeepGEMM 057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7:
            # deep_gemm/mega/__init__.py::_interleave_weights(gran=8).
            # [gate0:8, up0:8, gate8:16, up8:16, ...], outside timed calls.
            # Copy only the custom weights; the caller's baseline W1 stays intact.
            self.w1 = (w1.detach().reshape(local_experts, 2, intermediate // 8, 8, hidden)
                       .transpose(1, 2).contiguous().reshape_as(w1))
            # Use PyTorch's default CUDA peer-mapping allocator.
            self.x_buffer = symm_mem.empty(x.shape, dtype=x.dtype, device=self.device)
            self.ids_buffer = symm_mem.empty(ids.shape, dtype=ids.dtype, device=self.device)
            self.weights_buffer = symm_mem.empty(weights.shape, dtype=weights.dtype, device=self.device)
            self.slots = symm_mem.empty((tokens, topk, hidden), dtype=torch.bfloat16, device=self.device)
            # Retain both tensors and handles for the full lifetime of all peer mappings.
            self.sym_x, self.sym_ids, self.sym_weights, self.sym_output = (
                symm_mem.rendezvous(tensor, group=group)
                for tensor in (self.x_buffer, self.ids_buffer, self.weights_buffer, self.slots)
            )
            pointer_array = ctypes.c_void_p * 8
            self.peers = MoePeers(*(pointer_array(*handle.buffer_ptrs) for handle in (
                self.sym_x, self.sym_ids, self.sym_weights, self.sym_output
            )))
            self.routes = torch.empty((local_experts, self.capacity), dtype=torch.int32, device=self.device)
            self.counts = torch.empty((local_experts,), dtype=torch.int32, device=self.device)
            self.offsets = torch.empty((local_experts + 1,), dtype=torch.int64, device=self.device)
            self.activation = torch.empty((self.pool_capacity, intermediate),
                                          dtype=torch.bfloat16, device=self.device)
            self.pool = torch.empty((self.pool_capacity, hidden), dtype=torch.bfloat16, device=self.device)
            self.route_weights = torch.empty((local_experts, self.capacity), dtype=torch.float32, device=self.device)
            self.ready = torch.empty((local_experts, self.capacity // block_m), dtype=torch.int32, device=self.device)
            self.peers.pool = self.pool.data_ptr()
            self.peers.route_weights = self.route_weights.data_ptr()
            self.peers.ready = self.ready.data_ptr()
            self.peers.offsets = self.offsets.data_ptr()
            self.carrier_experts = torch.empty((self.world * tokens,), dtype=torch.int32, device=self.device)
            self.carrier_rows = torch.empty((self.world * tokens,), dtype=torch.int64, device=self.device)
            self.peers.carrier_experts = self.carrier_experts.data_ptr()
            self.peers.carrier_rows = self.carrier_rows.data_ptr()
            self.output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=self.device)
            self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._routes_ptr = ctypes.cast(self.routes.data_ptr(), ctypes.POINTER(ctypes.c_int))
        self._counts_ptr = ctypes.cast(self.counts.data_ptr(), ctypes.POINTER(ctypes.c_int))

    def _check_tensor(self, tensor, name, shape, dtype):
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                or tensor.device != self.device or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous {dtype} {shape} on {self.device}")

    @torch.no_grad()
    def copy_inputs(self):
        """Copy current input payload and routing data into the symmetric buffers."""
        sources = (self.x, self.ids, self.weights)
        with torch.cuda.device(self.device):
            for destination, source in zip(
                    (self.x_buffer, self.ids_buffer, self.weights_buffer), sources):
                destination.copy_(source)

    def barrier(self):
        """Collective device barrier on the current stream, using output's handle."""
        with torch.cuda.device(self.device):
            self.sym_output.barrier(channel=0)

    def _phase(self, phase, weights, activation):
        with torch.cuda.device(self.device):
            status = self._launch(
                phase, ctypes.byref(self.peers), ctypes.byref(self.shape),
                weights.data_ptr() if weights is not None else None, activation.data_ptr(),
                self._routes_ptr, self._counts_ptr, self.sms,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        if status:
            message = self._last_error()
            detail = message.decode("utf-8", errors="replace") if message else f"status {status}"
            raise RuntimeError(f"custom MoE phase {phase}: {detail}")

    def metadata(self):
        """Phase 0: rebuild routes/counts and compact aligned offsets on device."""
        self._phase(0, None, self.activation)

    def phase1(self):
        """Phase 1: peer reads, L1 GEMM, clamped SwiGLU and routing weights."""
        self._phase(1, self.w1, self.activation)

    def phase2(self):
        """Phase 2: L2 GEMM and writes to peer output slots."""
        self._phase(2, self.w2, self.activation)

    def reduce(self):
        """Phase 3: reduce local slots; caller must bracket this with barriers."""
        self._phase(3, self.slots, self.output)
        return self.output

    def __call__(self):
        self.copy_inputs()
        self.barrier()  # All three input copies must be visible before peer reads.
        self.metadata()
        self.phase1()
        self.phase2()
        self.barrier()  # All peer writes must finish before local reduction.
        self.reduce()
        self.barrier()  # All readers must finish before any rank reuses slots.
        return self.output
