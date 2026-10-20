"""Fused SwiGLU, `F.silu(gate) * up`, with Triton forward and backward kernels.

Inputs are CUDA tensors of identical shape `[..., I]` in BF16 or FP32. The last dimension must be
contiguous; the leading dimensions may be strided, so `gate` and `up` can be views of one fused
`[..., 2 * I]` projection output (`fused[..., :I]` and `fused[..., I:]`). Views are read in place;
an input that cannot be flattened to rows without a copy is copied by `reshape`.

Rounding matches the PyTorch BF16 reference: silu is computed in FP32 as `x / (1 + exp(-x))` with
libdevice (precise) `exp` and `div_rn`, rounded to the tensor dtype, then multiplied by `up` and
rounded again. The backward recomputes silu and sigmoid from `(gate, up)`, so nothing else is saved.
d_gate and d_up come from one kernel, with the same intermediate roundings as autograd.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)

_TILE_CONFIGS = [
    triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n}, num_warps=num_warps)
    for block_m in (1, 4)
    for block_n in (256, 1024)
    for num_warps in (2, 4)
]


@triton.jit
def _silu_rounded(x, OUT_DTYPE: tl.constexpr):
    """`x / (1 + exp(-x))` in FP32, rounded to OUT_DTYPE and widened back to FP32."""
    silu = libdevice.div_rn(x, 1.0 + libdevice.exp(-x))
    return silu.to(OUT_DTYPE).to(tl.float32)


@triton.autotune(configs=_TILE_CONFIGS, key=["rows", "cols", "OUT_DTYPE"])
@triton.jit
def _swiglu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    rows,
    cols,
    gate_row_stride,
    up_row_stride,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)
    gate = tl.load(gate_ptr + offs_m[:, None].to(tl.int64) * gate_row_stride + offs_n[None, :], mask=mask)
    up = tl.load(up_ptr + offs_m[:, None].to(tl.int64) * up_row_stride + offs_n[None, :], mask=mask)
    silu = _silu_rounded(gate.to(tl.float32), OUT_DTYPE)
    out = (silu * up.to(tl.float32)).to(OUT_DTYPE)
    tl.store(out_ptr + offs_m[:, None].to(tl.int64) * cols + offs_n[None, :], out, mask=mask)


@triton.autotune(configs=_TILE_CONFIGS, key=["rows", "cols", "OUT_DTYPE"])
@triton.jit
def _swiglu_bwd_kernel(
    grad_ptr,
    gate_ptr,
    up_ptr,
    d_gate_ptr,
    d_up_ptr,
    rows,
    cols,
    gate_row_stride,
    up_row_stride,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)
    out_offs = offs_m[:, None].to(tl.int64) * cols + offs_n[None, :]
    gate_offs = offs_m[:, None].to(tl.int64) * gate_row_stride + offs_n[None, :]
    up_offs = offs_m[:, None].to(tl.int64) * up_row_stride + offs_n[None, :]
    grad = tl.load(grad_ptr + out_offs, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offs, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + up_offs, mask=mask).to(tl.float32)

    # d_up = grad * silu(gate), silu rounded to the tensor dtype as in the forward.
    d_up = grad * _silu_rounded(gate, OUT_DTYPE)

    # d_silu = grad * up is rounded to the tensor dtype before silu backward.
    d_silu = (grad * up).to(OUT_DTYPE).to(tl.float32)
    sigmoid = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    d_gate = d_silu * sigmoid * (1.0 + gate * (1.0 - sigmoid))

    tl.store(d_gate_ptr + out_offs, d_gate.to(OUT_DTYPE), mask=mask)
    tl.store(d_up_ptr + out_offs, d_up.to(OUT_DTYPE), mask=mask)


def _triton_dtype(dtype: torch.dtype) -> tl.constexpr:
    return tl.bfloat16 if dtype == torch.bfloat16 else tl.float32


def _grid(rows: int, cols: int, config: dict) -> tuple[int, int]:
    return (triton.cdiv(rows, config["BLOCK_M"]), triton.cdiv(cols, config["BLOCK_N"]), 1)


def _check_inputs(gate: torch.Tensor, up: torch.Tensor) -> None:
    if not (gate.is_cuda and up.is_cuda):
        raise ValueError("swiglu requires CUDA tensors")
    if gate.shape != up.shape:
        raise ValueError(f"swiglu requires gate and up of the same shape, got {tuple(gate.shape)} and {tuple(up.shape)}")
    if gate.dtype != up.dtype or gate.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"swiglu supports bfloat16 or float32 inputs of one dtype, got {gate.dtype} and {up.dtype}")
    if gate.dim() == 0:
        raise ValueError("swiglu requires tensors with at least one dimension")
    if gate.numel() and gate.stride(-1) != 1 and gate.shape[-1] > 1:
        raise ValueError("swiglu requires a contiguous last dimension for gate")
    if up.numel() and up.stride(-1) != 1 and up.shape[-1] > 1:
        raise ValueError("swiglu requires a contiguous last dimension for up")


# Autotuned config per (kernel, rows, cols, dtype), and a compiled launcher per exact launch signature.
# Triton specializes on pointer alignment and on integer args being 1 or divisible by 16, so the launcher
# key carries the strides and pointer alignments. A key miss takes the slow path once.
_CONFIGS: dict[tuple, dict] = {}
_RUNNERS: dict[tuple, object] = {}


def _launch(kernel, tag: str, rows: int, cols: int, dtype: torch.dtype, args: tuple, tensors: tuple, strides: tuple):
    key = (tag, rows, cols, dtype, strides, tuple(t.data_ptr() % 16 for t in tensors))
    runner = _RUNNERS.get(key)
    stream = torch._C._cuda_getCurrentRawStream(tensors[0].device.index)
    if runner is not None:
        runner(*args, stream=stream)
        return
    out_dtype = _triton_dtype(dtype)
    config = _CONFIGS.get((tag, rows, cols, dtype))
    if config is None:
        kernel[lambda meta: _grid(rows, cols, meta)](*args, OUT_DTYPE=out_dtype)
        best = kernel.best_config
        config = _CONFIGS[(tag, rows, cols, dtype)] = {**best.kwargs, "num_warps": best.num_warps}
    grid = _grid(rows, cols, config)
    compiled = kernel.fn.warmup(*args, grid=grid, OUT_DTYPE=out_dtype, **config)
    runner = _RUNNERS[key] = compiled[grid]
    runner(*args, stream=stream)


class _Swiglu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)
        cols = gate.shape[-1]
        gate_rows, up_rows = gate.reshape(-1, cols), up.reshape(-1, cols)
        rows = gate_rows.shape[0]
        out = torch.empty(rows, cols, device=gate.device, dtype=gate.dtype)
        if rows and cols:
            strides = (gate_rows.stride(0), up_rows.stride(0))
            _launch(
                _swiglu_fwd_kernel,
                "fwd",
                rows,
                cols,
                gate.dtype,
                (gate_rows, up_rows, out, rows, cols, *strides),
                (gate_rows, up_rows, out),
                strides,
            )
        return out.view(gate.shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up = ctx.saved_tensors
        cols = gate.shape[-1]
        gate_rows, up_rows = gate.reshape(-1, cols), up.reshape(-1, cols)
        grad_rows = grad_out.contiguous().reshape(-1, cols)
        rows = gate_rows.shape[0]
        d_gate = torch.empty(rows, cols, device=gate.device, dtype=gate.dtype)
        d_up = torch.empty(rows, cols, device=gate.device, dtype=gate.dtype)
        if rows and cols:
            strides = (gate_rows.stride(0), up_rows.stride(0))
            _launch(
                _swiglu_bwd_kernel,
                "bwd",
                rows,
                cols,
                gate.dtype,
                (grad_rows, gate_rows, up_rows, d_gate, d_up, rows, cols, *strides),
                (grad_rows, gate_rows, up_rows, d_gate, d_up),
                strides,
            )
        return d_gate.view(gate.shape), d_up.view(gate.shape)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """`F.silu(gate) * up` with autograd through both inputs.

    `gate` and `up` must have the same shape `[..., I]` and dtype (BF16 or FP32) and a contiguous last
    dimension. Their leading dimensions may be strided views of a larger tensor.
    """
    _check_inputs(gate, up)
    return _Swiglu.apply(gate, up)
