"""Triton candidate folding BF16 cast, temperature scaling, reduction, and target capture."""

import time

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

COMPILE_SECONDS = {}
FUSED_ONLINE_UPDATE_CALLS = 0


@triton.jit
def _fused_update_kernel(
    M,
    S,
    T,
    LOGITS,
    INV_T,
    LABELS,
    TARGET,
    vocab_start,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < width
    inv_t = tl.load(INV_T + row).to(tl.float32)
    label = tl.load(LABELS + row).to(tl.int32)
    values = tl.load(LOGITS + row * width + cols, valid, other=0).to(tl.float32) * inv_t
    values = tl.where(valid, values, -float("inf"))

    chunk_max = tl.max(values, 0)
    old_m = tl.load(M + row)
    new_m = tl.maximum(old_m, chunk_max)
    old_scale = libdevice.exp(old_m - new_m)
    exp_values = libdevice.exp(values - new_m)
    chunk_sum = tl.sum(tl.where(valid, exp_values, 0.0), 0)
    chunk_weighted = tl.sum(tl.where(valid, exp_values * values, 0.0), 0)

    s = tl.load(S + row)
    t = tl.load(T + row)
    tl.store(S + row, s * old_scale + chunk_sum)
    tl.store(T + row, t * old_scale + chunk_weighted)
    tl.store(M + row, new_m)

    target_local = label - vocab_start
    target_in_tile = (label >= vocab_start) & (target_local < width)
    target_value = tl.sum(tl.where(valid & (cols == target_local), values, 0.0), 0)
    if target_in_tile:
        tl.store(TARGET + row, target_value)


def fused_online_update(m, s, t, logits, inv_temperature, labels, vocab_start, target_logits):
    """Mutate FP32 online state from one BF16 tile and capture a label in that tile."""
    import torch

    global FUSED_ONLINE_UPDATE_CALLS

    if logits.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 logits, got {logits.dtype}")
    if logits.ndim != 2 or not 0 < logits.shape[0] or not 0 < logits.shape[1] <= 8192:
        raise ValueError(f"expected logits [rows>=1, 1..8192], got {tuple(logits.shape)}")
    rows, width = logits.shape
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

    started = time.perf_counter()
    _fused_update_kernel[(rows,)](
        m,
        s,
        t,
        logits,
        inv_temperature,
        labels,
        target_logits,
        vocab_start,
        width,
        triton.next_power_of_2(width),
        num_warps=8,
    )
    signature = (logits.device.index, width, str(inv_temperature.dtype))
    COMPILE_SECONDS.setdefault(signature, time.perf_counter() - started)
    FUSED_ONLINE_UPDATE_CALLS += 1
    return m, s, t, target_logits
