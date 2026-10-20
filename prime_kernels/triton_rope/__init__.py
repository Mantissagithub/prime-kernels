"""Fused rotary position embedding (q and k in one launch) with Triton forward and backward.

Matches `prime_rl.trainer.models.layers.rotary_emb.apply_rotary_pos_emb` op for op. Each
elementwise op rounds to the input dtype before the next one, as PyTorch does, so the forward
output is bitwise equal to the reference for BF16 inputs.
"""

from __future__ import annotations

import warnings

import torch
import triton
import triton.language as tl

__all__ = ["apply_rotary"]

@triton.jit
def _mul_rn(a, b):
    # PTX mul.rn/add.rn are never contracted into an FMA, so each product and sum keeps its own rounding.
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


_CONFIGS = [triton.Config({"ROWS": rows}, num_warps=warps) for rows in (1, 2, 4, 8, 16, 32) for warps in (1, 2, 4, 8)]


@triton.jit
def _offsets(r, s0, s1, s2, st0, st1, st2):
    i2 = r % s2
    t = r // s2
    i1 = t % s1
    i0 = t // s1
    return i0 * st0 + i1 * st1 + i2 * st2


@triton.jit
def _fwd_tile(
    x_ptr,
    y_ptr,
    cos_ptr,
    sin_ptr,
    tile,
    nrows,
    s0,
    s1,
    s2,
    xst0,
    xst1,
    xst2,
    cst0,
    cst1,
    cst2,
    D,
    rot,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    R: tl.constexpr,
):
    r = tile * ROWS + tl.arange(0, ROWS)
    rmask = r < nrows
    x_row = _offsets(r, s0, s1, s2, xst0, xst1, xst2)
    c_row = _offsets(r, s0, s1, s2, cst0, cst1, cst2)
    d = tl.arange(0, BLOCK_D)
    half = rot // 2
    partner = tl.where(d < half, d + half, d - half)
    sign = tl.broadcast_to(tl.where(d < half, -1.0, 1.0)[None, :], (ROWS, BLOCK_D))
    lane = rmask[:, None] & (d < D)[None, :]
    rl = rmask[:, None] & (d < rot)[None, :]
    x = tl.load(x_ptr + x_row[:, None] + d[None, :], mask=lane, other=0.0).to(tl.float32)
    xp = tl.load(x_ptr + x_row[:, None] + partner[None, :], mask=rl, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + c_row[:, None] + d[None, :], mask=rl, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + c_row[:, None] + d[None, :], mask=rl, other=0.0).to(tl.float32)
    # Same rounding points as the reference: q*cos, rotate_half(q)*sin, then their sum.
    a = _mul_rn(x, c).to(R).to(tl.float32)
    b = _mul_rn(_mul_rn(sign, xp), s).to(R).to(tl.float32)
    y = tl.where(rl, _add_rn(a, b).to(R), x.to(R))
    tl.store(y_ptr + r[:, None] * D + d[None, :], y, mask=lane)


@triton.jit
def _bwd_tile(
    g_ptr,
    gx_ptr,
    cos_ptr,
    sin_ptr,
    tile,
    nrows,
    s0,
    s1,
    s2,
    cst0,
    cst1,
    cst2,
    D,
    rot,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    R: tl.constexpr,
):
    # Gradient of the forward: dx[m] = q*cos path + (+/-) grad_rh[partner] * sin[partner].
    r = tile * ROWS + tl.arange(0, ROWS)
    rmask = r < nrows
    c_row = _offsets(r, s0, s1, s2, cst0, cst1, cst2)
    d = tl.arange(0, BLOCK_D)
    half = rot // 2
    partner = tl.where(d < half, d + half, d - half)
    sign = tl.where(d < half, 1.0, -1.0)
    lane = rmask[:, None] & (d < D)[None, :]
    rl = rmask[:, None] & (d < rot)[None, :]
    g_row = r[:, None] * D
    g = tl.load(g_ptr + g_row + d[None, :], mask=lane, other=0.0).to(tl.float32)
    gp = tl.load(g_ptr + g_row + partner[None, :], mask=rl, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + c_row[:, None] + d[None, :], mask=rl, other=0.0).to(tl.float32)
    sp = tl.load(sin_ptr + c_row[:, None] + partner[None, :], mask=rl, other=0.0).to(tl.float32)
    a = _mul_rn(g, c).to(R).to(tl.float32)
    b = _mul_rn(gp, sp).to(R).to(tl.float32) * sign[None, :]
    gx = tl.where(rl, _add_rn(a, b).to(R), g.to(R))
    tl.store(gx_ptr + g_row + d[None, :], gx, mask=lane)


@triton.jit(do_not_specialize=["nq", "nk", "qs0", "qs1", "qs2", "qst0", "qst1", "qst2", "ks0", "ks1", "ks2", "kst0", "kst1", "kst2", "cqst0", "cqst1", "cqst2", "ckst0", "ckst1", "ckst2"])
def _rope_fwd_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    oq_ptr,
    ok_ptr,
    nq,
    nk,
    D,
    rot,
    esize,
    qs0,
    qs1,
    qs2,
    qst0,
    qst1,
    qst2,
    ks0,
    ks1,
    ks2,
    kst0,
    kst1,
    kst2,
    cqst0,
    cqst1,
    cqst2,
    ckst0,
    ckst1,
    ckst2,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    nqt = (nq + ROWS - 1) // ROWS
    if pid < nqt:
        _fwd_tile(q_ptr, oq_ptr, cos_ptr, sin_ptr, pid, nq, qs0, qs1, qs2, qst0, qst1, qst2, cqst0, cqst1, cqst2, D, rot, ROWS, BLOCK_D, R)
    else:
        _fwd_tile(k_ptr, ok_ptr, cos_ptr, sin_ptr, pid - nqt, nk, ks0, ks1, ks2, kst0, kst1, kst2, ckst0, ckst1, ckst2, D, rot, ROWS, BLOCK_D, R)


@triton.jit(do_not_specialize=["nq", "nk", "qs0", "qs1", "qs2", "ks0", "ks1", "ks2", "cqst0", "cqst1", "cqst2", "ckst0", "ckst1", "ckst2"])
def _rope_bwd_kernel(
    gq_ptr,
    gk_ptr,
    cos_ptr,
    sin_ptr,
    dq_ptr,
    dk_ptr,
    nq,
    nk,
    D,
    rot,
    esize,
    qs0,
    qs1,
    qs2,
    ks0,
    ks1,
    ks2,
    cqst0,
    cqst1,
    cqst2,
    ckst0,
    ckst1,
    ckst2,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    nqt = (nq + ROWS - 1) // ROWS
    if pid < nqt:
        _bwd_tile(gq_ptr, dq_ptr, cos_ptr, sin_ptr, pid, nq, qs0, qs1, qs2, cqst0, cqst1, cqst2, D, rot, ROWS, BLOCK_D, R)
    else:
        _bwd_tile(gk_ptr, dk_ptr, cos_ptr, sin_ptr, pid - nqt, nk, ks0, ks1, ks2, ckst0, ckst1, ckst2, D, rot, ROWS, BLOCK_D, R)


_rope_fwd_autotuned = triton.autotune(configs=_CONFIGS, key=["D", "rot", "esize"])(_rope_fwd_kernel)
_rope_bwd_autotuned = triton.autotune(configs=_CONFIGS, key=["D", "rot", "esize"])(_rope_bwd_kernel)

# (D, rot, element size) -> (ROWS, num_warps): the autotune winners from tune_table.py on the RTX 4060.
# One config per kernel means one compile per key. Other keys fall back to autotuning, with a warning.
_PINNED_FWD = {(64, 64, 2): (16, 1), (128, 128, 2): (32, 2), (64, 64, 4): (8, 1), (128, 128, 4): (4, 4)}
_PINNED_BWD = {(64, 64, 2): (32, 4), (128, 128, 2): (32, 2), (64, 64, 4): (16, 1), (128, 128, 4): (2, 4)}
_warned_unpinned: set[tuple[str, tuple[int, int, int]]] = set()


def _warn_unpinned(direction: str, key: tuple[int, int, int]) -> None:
    if (direction, key) in _warned_unpinned:
        return
    _warned_unpinned.add((direction, key))
    warnings.warn(
        f"triton_rope: no pinned {direction} config for (D, rot, esize)={key}; autotuning (slow first call)",
        stacklevel=4,
    )


def _pad3(values: tuple[int, ...], fill: int) -> tuple[int, ...]:
    if len(values) > 3:
        raise ValueError(f"apply_rotary supports at most 3 leading dims, got {len(values)}")
    return (fill,) * (3 - len(values)) + values


def _broadcast_cos(cos_u: torch.Tensor, lead: tuple[int, ...], name: str) -> torch.Tensor:
    for have, want in zip(cos_u.shape[:-1], lead):
        if have not in (1, want):
            raise ValueError(f"{name} with shape {tuple(cos_u.shape)} does not broadcast to q/k leading shape {lead}")
    return cos_u.expand(*lead, cos_u.shape[-1])


def _geometry(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int):
    """Validates inputs and returns the flattened row geometry shared by forward and backward."""
    if not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda):
        raise ValueError("apply_rotary needs CUDA tensors")
    if q.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"apply_rotary supports bfloat16 and float32, got {q.dtype}")
    if k.dtype != q.dtype or cos.dtype != q.dtype or sin.dtype != q.dtype:
        raise ValueError("q, k, cos and sin must share one dtype")
    if not (2 <= q.ndim <= 4) or k.ndim != q.ndim:
        raise ValueError(f"q and k must have the same rank of 2 to 4, got {q.ndim} and {k.ndim}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dim")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("q and k must have a contiguous last dim")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin shapes differ: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if cos.ndim != q.ndim - 1:
        raise ValueError(f"cos must have q.ndim - 1 dims, got {cos.ndim} for q.ndim {q.ndim}")
    head_dim = q.shape[-1]
    rot = cos.shape[-1]
    if rot > head_dim or rot % 2:
        raise ValueError(f"rotary dim {rot} must be even and at most head dim {head_dim}")
    if q.numel() == 0 or k.numel() == 0:
        raise ValueError("apply_rotary does not support empty q or k")

    cos = cos.contiguous()
    sin = sin.contiguous()
    cos_u = cos.unsqueeze(unsqueeze_dim)
    if cos_u.ndim != q.ndim:
        raise ValueError(f"unsqueeze_dim {unsqueeze_dim} gives rank {cos_u.ndim}, expected {q.ndim}")
    q_lead = tuple(q.shape[:-1])
    k_lead = tuple(k.shape[:-1])
    cos_q = _broadcast_cos(cos_u, q_lead, "cos")
    cos_k = _broadcast_cos(cos_u, k_lead, "cos")

    nq = 1
    for n in q_lead:
        nq *= n
    nk = 1
    for n in k_lead:
        nk *= n
    return {
        "D": head_dim,
        "rot": rot,
        "nq": nq,
        "nk": nk,
        "q_shape": _pad3(q_lead, 1),
        "q_strides": _pad3(q.stride()[:-1], 0),
        "k_shape": _pad3(k_lead, 1),
        "k_strides": _pad3(k.stride()[:-1], 0),
        "cos_q_strides": _pad3(cos_q.stride()[:-1], 0),
        "cos_k_strides": _pad3(cos_k.stride()[:-1], 0),
        "cos": cos,
        "sin": sin,
        "dtype": q.dtype,
    }


def _triton_dtype(dtype: torch.dtype):
    return tl.bfloat16 if dtype == torch.bfloat16 else tl.float32


def _forward(q, k, cos, sin, unsqueeze_dim):
    g = _geometry(q, k, cos, sin, unsqueeze_dim)
    oq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ok = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    esize = 2 if q.dtype == torch.bfloat16 else 4
    args = (
        q, k, g["cos"], g["sin"], oq, ok,
        g["nq"], g["nk"], g["D"], g["rot"], esize,
        *g["q_shape"], *g["q_strides"],
        *g["k_shape"], *g["k_strides"],
        *g["cos_q_strides"], *g["cos_k_strides"],
    )
    kwargs = {"BLOCK_D": triton.next_power_of_2(g["D"]), "R": _triton_dtype(g["dtype"])}
    key = (g["D"], g["rot"], esize)
    if key in _PINNED_FWD:
        rows, warps = _PINNED_FWD[key]
        grid = (triton.cdiv(g["nq"], rows) + triton.cdiv(g["nk"], rows),)
        _rope_fwd_kernel[grid](*args, ROWS=rows, num_warps=warps, **kwargs)
    else:
        _warn_unpinned("forward", key)
        grid = lambda meta: (triton.cdiv(g["nq"], meta["ROWS"]) + triton.cdiv(g["nk"], meta["ROWS"]),)
        _rope_fwd_autotuned[grid](*args, **kwargs)
    return oq, ok


def _backward(gq, gk, q_shape, k_shape, cos, sin, unsqueeze_dim, dtype):
    # Shape-only stand-ins are enough for the geometry; the grads are contiguous in q/k layout.
    q_stub = torch.empty(q_shape, dtype=dtype, device=cos.device)
    k_stub = torch.empty(k_shape, dtype=dtype, device=cos.device)
    g = _geometry(q_stub, k_stub, cos, sin, unsqueeze_dim)
    dq = torch.empty(q_shape, dtype=dtype, device=cos.device)
    dk = torch.empty(k_shape, dtype=dtype, device=cos.device)
    esize = 2 if dtype == torch.bfloat16 else 4
    args = (
        gq, gk, g["cos"], g["sin"], dq, dk,
        g["nq"], g["nk"], g["D"], g["rot"], esize,
        *g["q_shape"], *g["k_shape"],
        *g["cos_q_strides"], *g["cos_k_strides"],
    )
    kwargs = {"BLOCK_D": triton.next_power_of_2(g["D"]), "R": _triton_dtype(dtype)}
    key = (g["D"], g["rot"], esize)
    if key in _PINNED_BWD:
        rows, warps = _PINNED_BWD[key]
        grid = (triton.cdiv(g["nq"], rows) + triton.cdiv(g["nk"], rows),)
        _rope_bwd_kernel[grid](*args, ROWS=rows, num_warps=warps, **kwargs)
    else:
        _warn_unpinned("backward", key)
        grid = lambda meta: (triton.cdiv(g["nq"], meta["ROWS"]) + triton.cdiv(g["nk"], meta["ROWS"]),)
        _rope_bwd_autotuned[grid](*args, **kwargs)
    return dq, dk


class _ApplyRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, unsqueeze_dim):
        oq, ok = _forward(q, k, cos, sin, unsqueeze_dim)
        ctx.save_for_backward(cos, sin)
        ctx.unsqueeze_dim = unsqueeze_dim
        ctx.q_shape = tuple(q.shape)
        ctx.k_shape = tuple(k.shape)
        ctx.dtype = q.dtype
        return oq, ok

    @staticmethod
    def backward(ctx, gq, gk):
        cos, sin = ctx.saved_tensors
        dq, dk = _backward(
            gq.contiguous(), gk.contiguous(), ctx.q_shape, ctx.k_shape, cos, sin, ctx.unsqueeze_dim, ctx.dtype
        )
        return dq, dk, None, None, None


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the first `cos.shape[-1]` dims of q and k. cos and sin get no gradient."""
    return _ApplyRotary.apply(q, k, cos, sin, unsqueeze_dim)
