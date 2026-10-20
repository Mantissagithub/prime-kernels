"""TVM-FFI adapter for the existing CuTe LM-head online update kernel."""

import time
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from prime_kernels.lm_head_cute_fused_forward_backward import online_update as _kernel

COMPILE_SECONDS = {}
FUSED_ONLINE_UPDATE_CALLS = 0


@lru_cache(maxsize=32)
def _compiled(width: int, device_index: int, inv_dtype_name: str, labels_dtype_name: str):
    import torch

    device = torch.device("cuda", device_index)
    inv_dtype = getattr(torch, inv_dtype_name)
    labels_dtype = getattr(torch, labels_dtype_name)
    rows = 128
    with torch.cuda.device(device):
        logits = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
        m, s, t, target = (torch.empty((rows,), dtype=torch.float32, device=device) for _ in range(4))
        inv_temperature = torch.empty((rows,), dtype=inv_dtype, device=device)
        labels = torch.empty((rows,), dtype=labels_dtype, device=device)
        args = tuple(
            from_dlpack(tensor, enable_tvm_ffi=True).mark_compact_shape_dynamic(mode=0)
            for tensor in (logits, m, s, t, inv_temperature, labels, target)
        )
        started = time.perf_counter()
        compiled = cute.compile(
            _kernel._launch,
            *args,
            cutlass.Int32(0),
            cuda.CUstream(0),
            options="--enable-tvm-ffi",
        )
    COMPILE_SECONDS[(device_index, width, inv_dtype_name)] = time.perf_counter() - started
    return compiled


def fused_online_update(m, s, t, logits, inv_temperature, labels, vocab_start, target_logits):
    """Update the FP32 online state using TVM-FFI calls into the existing CuTe kernel."""
    import torch

    global FUSED_ONLINE_UPDATE_CALLS
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
    compiled = _compiled(
        logits.shape[1],
        device_index,
        str(inv_temperature.dtype).removeprefix("torch."),
        str(labels.dtype).removeprefix("torch."),
    )
    stream = torch.cuda.current_stream(device=logits.device)
    compiled(logits, m, s, t, inv_temperature, labels, target_logits, cutlass.Int32(vocab_start), stream)
    FUSED_ONLINE_UPDATE_CALLS += 1
    return m, s, t, target_logits
