"""Triton BF16 pointwise backward epilogue for the unmasked LM-head path."""

import time

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

COMPILE_SECONDS = {}
BACKWARD_EPILOGUE_CALLS = 0


@triton.jit
def _backward_epilogue_kernel(
    LOGITS,
    INV_T,
    LOGZ,
    GRAD,
    LABELS,
    OUT,
    vocab_start,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    valid = cols < width

    logits = tl.load(LOGITS + row * width + cols, valid, other=0).to(tl.float32)
    inv_t = tl.load(INV_T + row).to(tl.float32)
    logz = tl.load(LOGZ + row).to(tl.float32)
    grad = tl.load(GRAD + row).to(tl.float32)
    label = tl.load(LABELS + row).to(tl.int32)

    scaled = tl.where(valid, logits * inv_t, -float("inf"))
    probabilities = libdevice.exp(scaled - logz)
    grad_logits = (-grad) * probabilities

    target_local = label - vocab_start
    target_in_tile = (label >= vocab_start) & (target_local < width)
    target_grad = tl.where(target_in_tile & (cols == target_local), grad, 0.0)
    grad_logits = grad_logits + target_grad
    grad_logits = grad_logits * inv_t
    tl.store(OUT + row * width + cols, grad_logits, valid)


def fused_backward_epilogue(logits, inv_temperature, logz, grad_logprobs, labels, vocab_start):
    """Return BF16 dlogits for one BF16 logits tile, matching the stock op order."""
    import torch

    global BACKWARD_EPILOGUE_CALLS
    if logits.dtype != torch.bfloat16 or logits.ndim != 2:
        raise TypeError(f"expected BF16 logits [rows,width], got {logits.dtype} {tuple(logits.shape)}")
    rows, width = logits.shape
    if inv_temperature.shape != (rows,) or inv_temperature.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("inv_temperature must be a BF16 or FP32 row vector")
    if logz.shape != (rows,) or logz.dtype != torch.float32:
        raise ValueError("logz must be an FP32 row vector")
    if grad_logprobs.shape != (rows,) or grad_logprobs.dtype != torch.float32:
        raise ValueError("grad_logprobs must be an FP32 row vector")
    if labels.shape != (rows,) or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must be an int32 or int64 row vector")
    tensors = (logits, inv_temperature, logz, grad_logprobs, labels)
    if not all(tensor.is_cuda and tensor.device == logits.device and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("inputs must be contiguous tensors on one CUDA device")
    if width < 1:
        raise ValueError("logits tile width must be positive")

    output = torch.empty_like(logits)
    started = time.perf_counter()
    _backward_epilogue_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        inv_temperature,
        logz,
        grad_logprobs,
        labels,
        output,
        vocab_start,
        width,
        1024,
        enable_fp_fusion=False,
    )
    signature = (width, str(inv_temperature.dtype))
    COMPILE_SECONDS.setdefault(signature, time.perf_counter() - started)
    BACKWARD_EPILOGUE_CALLS += 1
    return output
