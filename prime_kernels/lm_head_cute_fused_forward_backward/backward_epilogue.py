"""CuTeDSL BF16 pointwise backward epilogue for unmasked LM-head rows."""

import time
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

COMPILE_SECONDS = {}
BACKWARD_EPILOGUE_CALLS = 0
TILE_COLUMNS = 1024
THREADS = 128


@cute.kernel
def _backward_epilogue_kernel(
    logits: cute.Tensor,
    inv_temperature: cute.Tensor,
    logz: cute.Tensor,
    grad_logprobs: cute.Tensor,
    labels: cute.Tensor,
    output: cute.Tensor,
    vocab_start: cutlass.Int32,
):
    lane, _, _ = cute.arch.thread_idx()
    row, tile, _ = cute.arch.block_idx()
    width = logits.shape[1]

    inv_t = inv_temperature[row].to(cutlass.Float32)
    logz_value = logz[row]
    grad = grad_logprobs[row]
    label = labels[row].to(cutlass.Int32)
    target_local = label - vocab_start
    target_in_tile = (label >= vocab_start) & (target_local < width)

    for offset in cutlass.range(lane, TILE_COLUMNS, THREADS):
        col = tile * TILE_COLUMNS + offset
        if col < width:
            logits_f32 = logits[row, col].to(cutlass.Float32)
            scaled = logits_f32 * inv_t
            probability = cute.math.exp(scaled - logz_value, fastmath=False)
            grad_logits = (-grad) * probability
            if target_in_tile & (col == target_local):
                grad_logits = grad_logits + grad
            grad_logits = grad_logits * inv_t
            output[row, col] = grad_logits.to(cutlass.BFloat16)


@cute.jit
def _launch(
    logits: cute.Tensor,
    inv_temperature: cute.Tensor,
    logz: cute.Tensor,
    grad_logprobs: cute.Tensor,
    labels: cute.Tensor,
    output: cute.Tensor,
    vocab_start: cutlass.Int32,
    tile_count: cutlass.Int32,
    stream: cuda.CUstream,
):
    _backward_epilogue_kernel(
        logits, inv_temperature, logz, grad_logprobs, labels, output, vocab_start
    ).launch(grid=[logits.shape[0], tile_count, 1], block=[THREADS, 1, 1], stream=stream)


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
        inv_temperature = torch.empty((compile_rows,), dtype=inv_dtype, device=device)
        logz = torch.empty((compile_rows,), dtype=torch.float32, device=device)
        grad_logprobs = torch.empty((compile_rows,), dtype=torch.float32, device=device)
        labels = torch.empty((compile_rows,), dtype=torch.long, device=device)
        output = torch.empty_like(logits)
        args = _dynamic_args((logits, inv_temperature, logz, grad_logprobs, labels, output))
        started = time.perf_counter()
        compiled = cute.compile(
            _launch,
            *args,
            cutlass.Int32(0),
            cutlass.Int32((width + TILE_COLUMNS - 1) // TILE_COLUMNS),
            cuda.CUstream(0),
        )
    COMPILE_SECONDS[(width, inv_dtype_name)] = time.perf_counter() - started
    return compiled


def fused_backward_epilogue(logits, inv_temperature, logz, grad_logprobs, labels, vocab_start):
    """Return BF16 dlogits using the stock pointwise operation order."""
    import torch

    global BACKWARD_EPILOGUE_CALLS
    if logits.dtype != torch.bfloat16 or logits.ndim != 2:
        raise TypeError(f"expected BF16 logits [rows,width], got {logits.dtype} {tuple(logits.shape)}")
    rows, width = logits.shape
    if rows < 1 or width < 1:
        raise ValueError("logits must have positive row and tile dimensions")
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

    device_index = logits.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    inv_dtype_name = str(inv_temperature.dtype).removeprefix("torch.")
    compiled = _compiled(width, device_index, inv_dtype_name)
    output = torch.empty_like(logits)
    args = _dynamic_args((*tensors, output))
    stream = cuda.CUstream(torch.cuda.current_stream(device=logits.device).cuda_stream)
    tile_count = cutlass.Int32((width + TILE_COLUMNS - 1) // TILE_COLUMNS)
    compiled(*args, cutlass.Int32(vocab_start), tile_count, stream)
    BACKWARD_EPILOGUE_CALLS += 1
    return output
