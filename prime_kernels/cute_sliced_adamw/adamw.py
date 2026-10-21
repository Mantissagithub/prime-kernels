"""Single-slice CuTe AdamW update with the native unfused FP32 operation order."""

import math
import struct
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

THREADS = 128
VECTOR_WIDTH = 4
REPEAT_WIDTH = 2
BLOCK_ELEMENTS = THREADS * VECTOR_WIDTH * REPEAT_WIDTH


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dsl_user_op
def _rn_add(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    result = llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _rn_mul(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    result = llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _rn_div(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    result = llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _rn_sqrt(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    result = llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [a.ir_value(loc=loc, ip=ip)],
        "sqrt.rn.f32 $0, $1;",
        "=f,f",
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _rn_fma(a: cutlass.Float32, b: cutlass.Float32, c: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    result = llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip), c.ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _neg(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    result = llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [a.ir_value(loc=loc, ip=ip)],
        "neg.f32 $0, $1;",
        "=f,f",
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@cute.jit
def _update_one(
    param,
    grad,
    avg,
    avg_sq,
    decay_factor,
    beta1,
    lerp_weight,
    beta2,
    one_minus_beta2,
    eps,
    negative_step_size,
    reciprocal_bc2_sqrt,
):
    grad = cutlass.Float32(grad)
    param = cutlass.Float32(param)
    avg = cutlass.Float32(avg)
    avg_sq = cutlass.Float32(avg_sq)

    if decay_factor != 1.0:
        param = _rn_mul(param, decay_factor)

    if lerp_weight < 0.5 and lerp_weight > -0.5:
        difference = _rn_add(grad, _neg(avg))
        avg = _rn_fma(lerp_weight, difference, avg)
    else:
        difference = _rn_add(grad, _neg(avg))
        complement = _rn_add(cutlass.Float32(1.0), _neg(lerp_weight))
        avg = _rn_fma(_neg(complement), difference, grad)

    avg_sq = _rn_mul(avg_sq, beta2)
    grad_sq = _rn_mul(grad, grad)
    avg_sq = _rn_fma(one_minus_beta2, grad_sq, avg_sq)

    denom = _rn_mul(_rn_sqrt(avg_sq), reciprocal_bc2_sqrt)
    denom = _rn_add(denom, eps)
    quotient = _rn_div(avg, denom)
    param = _rn_fma(negative_step_size, quotient, param)
    return param, avg, avg_sq


@cute.kernel
def _single_kernel(
    params: cute.Tensor,
    grads: cute.Tensor,
    exp_avg: cute.Tensor,
    exp_avg_sq: cute.Tensor,
    decay_factor: cutlass.Float32,
    beta1: cutlass.Float32,
    lerp_weight: cutlass.Float32,
    beta2: cutlass.Float32,
    one_minus_beta2: cutlass.Float32,
    eps: cutlass.Float32,
    negative_step_size: cutlass.Float32,
    reciprocal_bc2_sqrt: cutlass.Float32,
):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    base = block * BLOCK_ELEMENTS
    length = params.shape[0]

    if length - base >= BLOCK_ELEMENTS:
        tv_layout = cute.make_layout(
            (THREADS, (VECTOR_WIDTH, REPEAT_WIDTH)),
            stride=(VECTOR_WIDTH, (1, THREADS * VECTOR_WIDTH)),
        )
        rmem_layout = cute.make_layout(((VECTOR_WIDTH, REPEAT_WIDTH),), stride=((1, VECTOR_WIDTH),))
        vector_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)
        param_tile = cute.make_tensor(
            (params.iterator + base).align(16), cute.make_layout((BLOCK_ELEMENTS,), stride=(1,))
        )
        grad_tile = cute.make_tensor(
            (grads.iterator + base).align(16), cute.make_layout((BLOCK_ELEMENTS,), stride=(1,))
        )
        avg_tile = cute.make_tensor(
            (exp_avg.iterator + base).align(16), cute.make_layout((BLOCK_ELEMENTS,), stride=(1,))
        )
        avg_sq_tile = cute.make_tensor(
            (exp_avg_sq.iterator + base).align(16), cute.make_layout((BLOCK_ELEMENTS,), stride=(1,))
        )
        param_fragment = cute.composition(param_tile, tv_layout)[(lane, None)]
        grad_fragment = cute.composition(grad_tile, tv_layout)[(lane, None)]
        avg_fragment = cute.composition(avg_tile, tv_layout)[(lane, None)]
        avg_sq_fragment = cute.composition(avg_sq_tile, tv_layout)[(lane, None)]
        param_regs = cute.make_rmem_tensor(rmem_layout, cutlass.Float32)
        grad_regs = cute.make_rmem_tensor(rmem_layout, cutlass.Float32)
        avg_regs = cute.make_rmem_tensor(rmem_layout, cutlass.Float32)
        avg_sq_regs = cute.make_rmem_tensor(rmem_layout, cutlass.Float32)
        cute.copy(vector_atom, param_fragment, param_regs)
        cute.copy(vector_atom, grad_fragment, grad_regs)
        cute.copy(vector_atom, avg_fragment, avg_regs)
        cute.copy(vector_atom, avg_sq_fragment, avg_sq_regs)

        for repeat in cutlass.range_constexpr(REPEAT_WIDTH):
            for vector_lane in cutlass.range_constexpr(VECTOR_WIDTH):
                param, avg, avg_sq = _update_one(
                    param_regs[((vector_lane, repeat),)],
                    grad_regs[((vector_lane, repeat),)],
                    avg_regs[((vector_lane, repeat),)],
                    avg_sq_regs[((vector_lane, repeat),)],
                    decay_factor,
                    beta1,
                    lerp_weight,
                    beta2,
                    one_minus_beta2,
                    eps,
                    negative_step_size,
                    reciprocal_bc2_sqrt,
                )
                param_regs[((vector_lane, repeat),)] = param
                avg_regs[((vector_lane, repeat),)] = avg
                avg_sq_regs[((vector_lane, repeat),)] = avg_sq
        cute.copy(vector_atom, param_regs, param_fragment)
        cute.copy(vector_atom, avg_regs, avg_fragment)
        cute.copy(vector_atom, avg_sq_regs, avg_sq_fragment)
    else:
        for repeat in cutlass.range_constexpr(REPEAT_WIDTH):
            for vector_lane in cutlass.range_constexpr(VECTOR_WIDTH):
                index = base + lane * VECTOR_WIDTH + vector_lane + repeat * THREADS * VECTOR_WIDTH
                if index < length:
                    param, avg, avg_sq = _update_one(
                        params[index],
                        grads[index],
                        exp_avg[index],
                        exp_avg_sq[index],
                        decay_factor,
                        beta1,
                        lerp_weight,
                        beta2,
                        one_minus_beta2,
                        eps,
                        negative_step_size,
                        reciprocal_bc2_sqrt,
                    )
                    params[index] = param
                    exp_avg[index] = avg
                    exp_avg_sq[index] = avg_sq


@cute.jit
def _launch(
    params: cute.Tensor,
    grads: cute.Tensor,
    exp_avg: cute.Tensor,
    exp_avg_sq: cute.Tensor,
    decay_factor: cutlass.Float32,
    beta1: cutlass.Float32,
    lerp_weight: cutlass.Float32,
    beta2: cutlass.Float32,
    one_minus_beta2: cutlass.Float32,
    eps: cutlass.Float32,
    negative_step_size: cutlass.Float32,
    reciprocal_bc2_sqrt: cutlass.Float32,
    stream: cuda.CUstream,
):
    blocks = (params.shape[0] + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS
    _single_kernel(
        params,
        grads,
        exp_avg,
        exp_avg_sq,
        decay_factor,
        beta1,
        lerp_weight,
        beta2,
        one_minus_beta2,
        eps,
        negative_step_size,
        reciprocal_bc2_sqrt,
    ).launch(grid=[blocks, 1, 1], block=[THREADS, 1, 1], stream=stream)


@lru_cache(maxsize=4)
def _compiled(device_index: int, numel: int):
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        tensors = tuple(torch.empty(numel, dtype=torch.float32, device=device) for _ in range(4))
        tensor_args = tuple(
            from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_compact_shape_dynamic(mode=0)
            for tensor in tensors
        )
        scalar_args = (cutlass.Float32(1.0),) * 8
        compiled = cute.compile(
            _launch,
            *tensor_args,
            *scalar_args,
            cuda.CUstream(0),
            options="--enable-tvm-ffi",
        )
    return compiled


class CuteUnfusedAdamWSlice:
    """One fixed-shape CUDA slice; caller owns the CPU step increment."""

    def __init__(self, param: torch.Tensor):
        if param.device.type != "cuda" or param.dtype != torch.float32 or not param.is_contiguous():
            raise TypeError("param must be contiguous CUDA FP32")
        if param.numel() == 0 or param.data_ptr() % 16:
            raise ValueError("param must be nonempty and 16-byte aligned")
        self.device = param.device
        self.numel = param.numel()
        self.compiled = _compiled(self.device.index, self.numel)

    def step_(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        step: torch.Tensor,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        tensors = (param, grad, exp_avg, exp_avg_sq)
        if any(
            tensor.device != self.device
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or tensor.numel() != self.numel
            or tensor.data_ptr() % 16
            for tensor in tensors
        ):
            raise ValueError(
                "param, grad, and moments must be contiguous, aligned CUDA FP32 tensors matching the compiled slice"
            )
        if step.device.type != "cpu" or step.dtype != torch.float32 or step.numel() != 1:
            raise TypeError("step must be a CPU FP32 scalar")

        beta1, beta2 = map(float, betas)
        step_value = _f32(float(step.item()) + 1.0)
        bias1 = 1.0 - beta1**step_value
        bias2_sqrt = math.sqrt(1.0 - beta2**step_value)
        decay_factor = _f32(1.0 - float(lr) * float(weight_decay))
        beta1_f = _f32(beta1)
        lerp_weight = _f32(1.0 - beta1)
        beta2_f = _f32(beta2)
        one_minus_beta2 = _f32(1.0 - beta2)
        eps_f = _f32(float(eps))
        negative_step_size = _f32(-(float(lr) / bias1))
        reciprocal_bc2_sqrt = _f32(1.0 / bias2_sqrt)
        stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        self.compiled(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            cutlass.Float32(decay_factor),
            cutlass.Float32(beta1_f),
            cutlass.Float32(lerp_weight),
            cutlass.Float32(beta2_f),
            cutlass.Float32(one_minus_beta2),
            cutlass.Float32(eps_f),
            cutlass.Float32(negative_step_size),
            cutlass.Float32(reciprocal_bc2_sqrt),
            stream,
        )
        torch.autograd.graph.increment_version((param, exp_avg, exp_avg_sq))
