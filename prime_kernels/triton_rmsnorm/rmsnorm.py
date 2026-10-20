"""Triton RMSNorm forward and backward, matching the fp32-upcast PyTorch reference.

Reference: h = x.float(); h = h * rsqrt(mean(h^2) + eps); out = weight * h.to(x.dtype).
Rounding points mirror the reference: the normalized value is rounded to x.dtype before the
weight multiply, and the product is rounded to the promoted output dtype.

Weight gradient uses a two-stage reduction: each program writes an fp32 partial over
PARTIAL_ROWS rows, then a second kernel sums the partials in a fixed order. Each per-row product
is rounded to the output dtype first, as the reference's mul backward does.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

logger = logging.getLogger(__name__)

MAX_HIDDEN = 8192
PARTIAL_ROWS = 8
REDUCE_BLOCK_H = 128
REDUCE_BLOCK_P = 32
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)

_FWD_CONFIGS = [
    triton.Config({"BLOCK_M": block_m}, num_warps=num_warps, num_stages=num_stages)
    for block_m in (1, 2, 4, 8)
    for num_warps in (2, 4, 8)
    for num_stages in (1, 2)
]
_BWD_CONFIGS = [
    triton.Config({}, num_warps=num_warps, num_stages=num_stages)
    for num_warps in (2, 4, 8)
    for num_stages in (1, 2)
]


@triton.jit(do_not_specialize=["n_rows"])
def _fwd_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    rstd_ptr,
    n_rows,
    n_cols,
    inv_cols,
    eps,
    dtype_code: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    # The sum of squares follows the order of PyTorch's CUDA reduction for a contiguous row:
    # 32 lanes, each owning 4 consecutive columns per 128-column chunk with one accumulator per
    # column offset, combined as ((a0 + a1) + a2) + a3, then the lanes halved (lane l with l+16
    # first). This keeps the fp32 variance bitwise equal to torch's mean for hidden sizes 576,
    # 960 and 1024 (see probe_variance_order.py).
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, 32)
    row_mask = rows < n_rows
    row_off = rows[:, None].to(tl.int64) * n_cols

    acc0 = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    for j in tl.static_range(NUM_CHUNKS):
        cols = j * 128 + lanes * 4
        m0 = row_mask[:, None] & (cols[None, :] < n_cols)
        m1 = row_mask[:, None] & (cols[None, :] + 1 < n_cols)
        m2 = row_mask[:, None] & (cols[None, :] + 2 < n_cols)
        m3 = row_mask[:, None] & (cols[None, :] + 3 < n_cols)
        x0 = tl.load(x_ptr + row_off + cols[None, :], m0, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + row_off + cols[None, :] + 1, m1, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + row_off + cols[None, :] + 2, m2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + row_off + cols[None, :] + 3, m3, other=0.0).to(tl.float32)
        acc0 += x0 * x0
        acc1 += x1 * x1
        acc2 += x2 * x2
        acc3 += x3 * x3

    v = ((acc0 + acc1) + acc2) + acc3
    v = tl.sum(tl.reshape(v, (BLOCK_M, 2, 16)), axis=1)
    v = tl.sum(tl.reshape(v, (BLOCK_M, 2, 8)), axis=1)
    v = tl.sum(tl.reshape(v, (BLOCK_M, 2, 4)), axis=1)
    v = tl.sum(tl.reshape(v, (BLOCK_M, 2, 2)), axis=1)
    variance = tl.sum(v, axis=1)
    # mul_rn/add_rn keep torch's two roundings; a contracted FMA here changes rstd by an ulp.
    rstd = tl.math.rsqrt(libdevice.add_rn(libdevice.mul_rn(variance, inv_cols), eps))
    tl.store(rstd_ptr + rows, rstd, row_mask)

    # Second pass writes each chunk; reads are L1/L2 hits since the row was just loaded.
    rstd2 = rstd[:, None]
    for j in tl.static_range(NUM_CHUNKS):
        for k in tl.static_range(4):
            cols = j * 128 + lanes * 4 + k
            m = row_mask[:, None] & (cols[None, :] < n_cols)
            x = tl.load(x_ptr + row_off + cols[None, :], m, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, cols < n_cols, other=0.0).to(tl.float32)
            normed = (x * rstd2).to(x_ptr.dtype.element_ty).to(tl.float32)
            tl.store(out_ptr + row_off + cols[None, :], (w[None, :] * normed).to(out_ptr.dtype.element_ty), m)


@triton.jit(do_not_specialize=["n_rows"])
def _bwd_kernel(
    x_ptr,
    w_ptr,
    g_ptr,
    rstd_ptr,
    dx_ptr,
    dw_part_ptr,
    n_rows,
    n_cols,
    inv_cols,
    dtype_code: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows[:, None].to(tl.int64) * n_cols + cols[None, :]

    x = tl.load(x_ptr + offs, mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, col_mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, row_mask, other=0.0)[:, None]

    normed = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
    dw_partial = tl.sum((g * normed).to(g_ptr.dtype.element_ty).to(tl.float32), axis=0)
    tl.store(dw_part_ptr + pid.to(tl.int64) * n_cols + cols, dw_partial, col_mask)

    # The reference's normalized value is in x.dtype, so autograd rounds its gradient to x.dtype.
    dn = (g * w[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
    dot = tl.sum(dn * x, axis=1)[:, None]
    dx = rstd * dn - rstd * rstd * rstd * inv_cols * x * dot
    tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask)


# Configs chosen by autotune at 1024 rows (tune_sweep.py). Keyed by (n_cols, dtype_code), where
# dtype_code = 2 * index(x.dtype) + index(weight.dtype) with dtypes (bfloat16, float32).
# Other (n_cols, dtype) pairs fall back to autotuning with a one-time warning.
_PINNED = {
    (576, 0): {  # x=bfloat16, weight=bfloat16
        "fwd": {"BLOCK_M": 1, "num_warps": 2, "num_stages": 1},
        "bwd": {"num_warps": 4, "num_stages": 1},
    },
    (576, 1): {  # x=bfloat16, weight=float32
        "fwd": {"BLOCK_M": 1, "num_warps": 4, "num_stages": 1},
        "bwd": {"num_warps": 4, "num_stages": 1},
    },
    (576, 2): {  # x=float32, weight=bfloat16
        "fwd": {"BLOCK_M": 4, "num_warps": 8, "num_stages": 1},
        "bwd": {"num_warps": 4, "num_stages": 2},
    },
    (576, 3): {  # x=float32, weight=float32
        "fwd": {"BLOCK_M": 1, "num_warps": 2, "num_stages": 1},
        "bwd": {"num_warps": 8, "num_stages": 2},
    },
    (960, 0): {  # x=bfloat16, weight=bfloat16
        "fwd": {"BLOCK_M": 4, "num_warps": 2, "num_stages": 1},
        "bwd": {"num_warps": 4, "num_stages": 2},
    },
    (960, 1): {  # x=bfloat16, weight=float32
        "fwd": {"BLOCK_M": 1, "num_warps": 2, "num_stages": 2},
        "bwd": {"num_warps": 8, "num_stages": 1},
    },
    (960, 2): {  # x=float32, weight=bfloat16
        "fwd": {"BLOCK_M": 2, "num_warps": 4, "num_stages": 1},
        "bwd": {"num_warps": 4, "num_stages": 2},
    },
    (960, 3): {  # x=float32, weight=float32
        "fwd": {"BLOCK_M": 1, "num_warps": 2, "num_stages": 1},
        "bwd": {"num_warps": 4, "num_stages": 1},
    },
    (1024, 0): {  # x=bfloat16, weight=bfloat16
        "fwd": {"BLOCK_M": 4, "num_warps": 2, "num_stages": 2},
        "bwd": {"num_warps": 4, "num_stages": 2},
    },
    (1024, 1): {  # x=bfloat16, weight=float32
        "fwd": {"BLOCK_M": 2, "num_warps": 4, "num_stages": 2},
        "bwd": {"num_warps": 8, "num_stages": 1},
    },
    (1024, 2): {  # x=float32, weight=bfloat16
        "fwd": {"BLOCK_M": 1, "num_warps": 2, "num_stages": 1},
        "bwd": {"num_warps": 4, "num_stages": 1},
    },
    (1024, 3): {  # x=float32, weight=float32
        "fwd": {"BLOCK_M": 1, "num_warps": 2, "num_stages": 2},
        "bwd": {"num_warps": 4, "num_stages": 1},
    },
}

_fwd_kernel_tuned = triton.autotune(configs=_FWD_CONFIGS, key=["n_cols", "dtype_code"])(_fwd_kernel)
_bwd_kernel_tuned = triton.autotune(configs=_BWD_CONFIGS, key=["n_cols", "dtype_code"])(_bwd_kernel)
_warned_unpinned = set()


def _warn_unpinned(n_cols: int, dtype_code: int) -> None:
    if (n_cols, dtype_code) not in _warned_unpinned:
        _warned_unpinned.add((n_cols, dtype_code))
        logger.warning("triton_rmsnorm: no pinned config for n_cols=%d dtype_code=%d, autotuning (slow first call)", n_cols, dtype_code)


@triton.jit(do_not_specialize=["n_parts"])
def _reduce_kernel(
    part_ptr,
    out_ptr,
    n_parts,
    n_cols,
    BLOCK_P: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < n_cols
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for p0 in range(0, n_parts, BLOCK_P):
        parts = p0 + tl.arange(0, BLOCK_P)
        mask = (parts[:, None] < n_parts) & col_mask[None, :]
        tile = tl.load(part_ptr + parts[:, None].to(tl.int64) * n_cols + cols[None, :], mask, other=0.0)
        acc += tl.sum(tile, axis=0)
    tl.store(out_ptr + cols, acc.to(out_ptr.dtype.element_ty), col_mask)


def _check_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if not (x.is_cuda and weight.is_cuda):
        raise ValueError("rmsnorm requires CUDA tensors")
    if x.device != weight.device:
        raise ValueError(f"x on {x.device} but weight on {weight.device}")
    if x.dtype not in _SUPPORTED_DTYPES or weight.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"rmsnorm supports bfloat16 and float32, got x={x.dtype}, weight={weight.dtype}")
    if x.dim() < 1 or x.shape[-1] == 0:
        raise ValueError(f"rmsnorm needs a non-empty last dim, got shape {tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError("rmsnorm needs at least one element")
    if x.stride(-1) != 1:
        raise ValueError("rmsnorm requires a contiguous last dim")
    if weight.dim() != 1 or weight.shape[0] != x.shape[-1] or weight.stride(0) != 1:
        raise ValueError(f"weight must be a contiguous 1-D tensor of size {x.shape[-1]}, got {tuple(weight.shape)}")
    if x.shape[-1] > MAX_HIDDEN:
        raise ValueError(f"rmsnorm supports hidden size <= {MAX_HIDDEN}, got {x.shape[-1]}")


def _dtype_code(x: torch.Tensor, weight: torch.Tensor) -> int:
    return _SUPPORTED_DTYPES.index(x.dtype) * len(_SUPPORTED_DTYPES) + _SUPPORTED_DTYPES.index(weight.dtype)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        n_cols = x.shape[-1]
        x2d = x.reshape(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty(x.shape, device=x.device, dtype=torch.result_type(weight, x))
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        dtype_code = _dtype_code(x, weight)
        args = (x2d, weight, out.view(n_rows, n_cols), rstd, n_rows, n_cols, 1.0 / n_cols, eps, dtype_code)
        num_chunks = triton.cdiv(n_cols, 128)
        pinned = _PINNED.get((n_cols, dtype_code))
        if pinned is None:
            _warn_unpinned(n_cols, dtype_code)
            _fwd_kernel_tuned[lambda meta: (triton.cdiv(n_rows, meta["BLOCK_M"]),)](*args, NUM_CHUNKS=num_chunks)
        else:
            cfg = pinned["fwd"]
            _fwd_kernel[(triton.cdiv(n_rows, cfg["BLOCK_M"]),)](
                *args, NUM_CHUNKS=num_chunks, BLOCK_M=cfg["BLOCK_M"], num_warps=cfg["num_warps"], num_stages=cfg["num_stages"]
            )
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.x_shape = x.shape
        ctx.dtype_code = dtype_code
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x2d.shape
        grad2d = grad_out.reshape(n_rows, n_cols).contiguous()
        dx = torch.empty_like(x2d)
        num_parts = triton.cdiv(n_rows, PARTIAL_ROWS)
        dw_part = torch.empty(num_parts, n_cols, device=x2d.device, dtype=torch.float32)
        block_h = triton.next_power_of_2(n_cols)
        bwd_args = (x2d, weight, grad2d, rstd, dx, dw_part, n_rows, n_cols, 1.0 / n_cols, ctx.dtype_code)
        pinned = _PINNED.get((n_cols, ctx.dtype_code))
        if pinned is None:
            _warn_unpinned(n_cols, ctx.dtype_code)
            _bwd_kernel_tuned[(num_parts,)](*bwd_args, BLOCK_M=PARTIAL_ROWS, BLOCK_H=block_h)
        else:
            cfg = pinned["bwd"]
            _bwd_kernel[(num_parts,)](
                *bwd_args, BLOCK_M=PARTIAL_ROWS, BLOCK_H=block_h, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"]
            )
        dw = torch.empty(n_cols, device=x2d.device, dtype=weight.dtype)
        _reduce_kernel[(triton.cdiv(n_cols, REDUCE_BLOCK_H),)](
            dw_part, dw, num_parts, n_cols, BLOCK_P=REDUCE_BLOCK_P, BLOCK_H=REDUCE_BLOCK_H
        )
        return dx.view(ctx.x_shape), dw, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim of `x` (any leading dims), scaled by `weight`.

    x: CUDA bfloat16 or float32, contiguous last dim. weight: CUDA 1-D bfloat16 or float32 of size x.shape[-1].
    Returns torch.result_type(weight, x). Differentiable w.r.t. x and weight.
    """
    _check_inputs(x, weight)
    return _RMSNorm.apply(x, weight, float(eps))
