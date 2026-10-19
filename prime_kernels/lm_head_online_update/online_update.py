"""CuTeDSL prototype for the LM-head online logsumexp state update."""

import operator
import time
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

COMPILE_SECONDS = {}


@cute.kernel
def _online_update_kernel(logits: cute.Tensor, m: cute.Tensor, s: cute.Tensor, t: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    width = logits.shape[1]

    local_max = -cutlass.Float32.inf
    for col in cutlass.range(lane, width, 32):
        local_max = cute.arch.fmax(local_max, logits[row, col])
    chunk_max = cute.arch.warp_reduction(local_max, cute.arch.fmax, threads_in_group=32)

    old_m = m[row]
    new_m = cute.arch.fmax(old_m, chunk_max)
    old_scale = cute.math.exp(old_m - new_m, fastmath=False)

    local_sum = cutlass.Float32.zero
    local_weighted = cutlass.Float32.zero
    for col in cutlass.range(lane, width, 32):
        value = logits[row, col]
        exp_value = cute.math.exp(value - new_m, fastmath=False)
        local_sum += exp_value
        local_weighted += exp_value * value

    chunk_sum = cute.arch.warp_reduction(local_sum, operator.add, threads_in_group=32)
    chunk_weighted = cute.arch.warp_reduction(local_weighted, operator.add, threads_in_group=32)
    if lane == 0:
        s[row] = s[row] * old_scale + chunk_sum
        t[row] = t[row] * old_scale + chunk_weighted
        m[row] = new_m


@cute.jit
def _launch(logits: cute.Tensor, m: cute.Tensor, s: cute.Tensor, t: cute.Tensor, stream: cuda.CUstream):
    _online_update_kernel(logits, m, s, t).launch(grid=[logits.shape[0], 1, 1], block=[32, 1, 1], stream=stream)


def _dynamic_args(tensors):
    return tuple(from_dlpack(tensor).mark_compact_shape_dynamic(mode=0) for tensor in tensors)


@lru_cache(maxsize=16)
def _compiled(width: int, device_index: int):
    import torch

    device = torch.device("cuda", device_index)
    compile_rows = 128
    with torch.cuda.device(device):
        logits = torch.empty((compile_rows, width), dtype=torch.float32, device=device)
        states = tuple(torch.empty((compile_rows,), dtype=torch.float32, device=device) for _ in range(3))
        args = _dynamic_args((logits, *states))
        default_stream = cuda.CUstream(0)
        started = time.perf_counter()
        compiled = cute.compile(_launch, *args, default_stream)
    COMPILE_SECONDS[(device_index, width)] = time.perf_counter() - started
    return compiled


def online_update(m, s, t, logits):
    """Mutate and return the three FP32 state tensors for one logits tile."""
    import torch

    if logits.dtype != torch.float32:
        raise TypeError(f"expected float32 scaled logits, got {logits.dtype}")
    if logits.ndim != 2 or not 0 < logits.shape[0] or not 0 < logits.shape[1] <= 8192:
        raise ValueError(f"expected logits [rows>=1, 1..8192], got {tuple(logits.shape)}")
    if any(state.dtype != torch.float32 or state.shape != logits.shape[:1] for state in (m, s, t)):
        raise ValueError("m, s, and t must be float32 vectors matching logits rows")
    if not all(tensor.is_cuda and tensor.device == logits.device for tensor in (logits, m, s, t)):
        raise ValueError("all tensors must be on the same CUDA device")
    if not all(tensor.is_contiguous() for tensor in (logits, m, s, t)):
        raise ValueError("all tensors must be contiguous")

    device_index = logits.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compiled = _compiled(logits.shape[1], device_index)
    args = _dynamic_args((logits, m, s, t))
    stream = cuda.CUstream(torch.cuda.current_stream(device=logits.device).cuda_stream)
    compiled(*args, stream)
    return m, s, t
