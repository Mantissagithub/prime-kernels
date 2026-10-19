"""CuTe candidate that folds BF16 cast, temperature scaling, reduction, and target capture."""

import operator
import time
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

COMPILE_SECONDS = {}


@cute.kernel
def _fused_update_kernel(
    logits: cute.Tensor,
    m: cute.Tensor,
    s: cute.Tensor,
    t: cute.Tensor,
    inv_temperature: cute.Tensor,
    labels: cute.Tensor,
    target_logits: cute.Tensor,
    vocab_start: cutlass.Int32,
):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    width = logits.shape[1]

    inv_t = inv_temperature[row].to(cutlass.Float32)
    label = labels[row].to(cutlass.Int32)
    target_local = label - vocab_start
    target_in_tile = (label >= vocab_start) & (target_local < width)
    target_value = cutlass.Float32.zero

    local_max = -cutlass.Float32.inf
    for col in cutlass.range(lane, width, 32):
        value = logits[row, col].to(cutlass.Float32) * inv_t
        local_max = cute.arch.fmax(local_max, value)
        if target_in_tile & (col == target_local):
            target_value = value
    chunk_max = cute.arch.warp_reduction(local_max, cute.arch.fmax, threads_in_group=32)

    old_m = m[row]
    new_m = cute.arch.fmax(old_m, chunk_max)
    old_scale = cute.math.exp(old_m - new_m, fastmath=False)

    local_sum = cutlass.Float32.zero
    local_weighted = cutlass.Float32.zero
    for col in cutlass.range(lane, width, 32):
        value = logits[row, col].to(cutlass.Float32) * inv_t
        exp_value = cute.math.exp(value - new_m, fastmath=False)
        local_sum += exp_value
        local_weighted += exp_value * value

    chunk_sum = cute.arch.warp_reduction(local_sum, operator.add, threads_in_group=32)
    chunk_weighted = cute.arch.warp_reduction(local_weighted, operator.add, threads_in_group=32)
    if lane == 0:
        s[row] = s[row] * old_scale + chunk_sum
        t[row] = t[row] * old_scale + chunk_weighted
        m[row] = new_m
    if target_in_tile & (lane == target_local % 32):
        target_logits[row] = target_value


@cute.jit
def _launch(
    logits: cute.Tensor,
    m: cute.Tensor,
    s: cute.Tensor,
    t: cute.Tensor,
    inv_temperature: cute.Tensor,
    labels: cute.Tensor,
    target_logits: cute.Tensor,
    vocab_start: cutlass.Int32,
    stream: cuda.CUstream,
):
    _fused_update_kernel(logits, m, s, t, inv_temperature, labels, target_logits, vocab_start).launch(
        grid=[logits.shape[0], 1, 1], block=[32, 1, 1], stream=stream
    )


def _dynamic_args(tensors):
    return tuple(from_dlpack(tensor).mark_compact_shape_dynamic(mode=0) for tensor in tensors)


@lru_cache(maxsize=32)
def _compiled(width: int, device_index: int, inv_dtype_name: str):
    import torch

    device = torch.device("cuda", device_index)
    inv_dtype = getattr(torch, inv_dtype_name)
    compile_rows = 128
    with torch.cuda.device(device):
        logits = torch.empty((compile_rows, width), dtype=torch.bfloat16, device=device)
        m, s, t, target = (torch.empty((compile_rows,), dtype=torch.float32, device=device) for _ in range(4))
        inv_temperature = torch.empty((compile_rows,), dtype=inv_dtype, device=device)
        labels = torch.empty((compile_rows,), dtype=torch.long, device=device)
        args = _dynamic_args((logits, m, s, t, inv_temperature, labels, target))
        default_stream = cuda.CUstream(0)
        started = time.perf_counter()
        compiled = cute.compile(_launch, *args, cutlass.Int32(0), default_stream)
    COMPILE_SECONDS[(device_index, width, inv_dtype_name)] = time.perf_counter() - started
    return compiled


def fused_online_update(m, s, t, logits, inv_temperature, labels, vocab_start, target_logits):
    """Update FP32 online state from BF16 logits and capture labels in this vocab tile."""
    import torch

    if logits.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 logits, got {logits.dtype}")
    if logits.ndim != 2 or not 0 < logits.shape[0] or not 0 < logits.shape[1] <= 8192:
        raise ValueError(f"expected logits [rows>=1, 1..8192], got {tuple(logits.shape)}")
    rows = logits.shape[0]
    if any(state.dtype != torch.float32 or state.shape != (rows,) for state in (m, s, t, target_logits)):
        raise ValueError("m, s, t, and target_logits must be float32 vectors matching logits rows")
    if inv_temperature.shape != (rows,) or inv_temperature.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("inv_temperature must be a float32 or bfloat16 vector matching logits rows")
    if labels.shape != (rows,) or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must be an int32 or int64 vector matching logits rows")
    tensors = (logits, m, s, t, inv_temperature, labels, target_logits)
    if not all(tensor.is_cuda and tensor.device == logits.device for tensor in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous")

    device_index = logits.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compiled = _compiled(logits.shape[1], device_index, str(inv_temperature.dtype).removeprefix("torch."))
    args = _dynamic_args(tensors)
    stream = cuda.CUstream(torch.cuda.current_stream(device=logits.device).cuda_stream)
    compiled(*args, cutlass.Int32(vocab_start), stream)
    return m, s, t, target_logits
