"""Batched FP32 gradient clipping and AdamW updates using Triton."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_BLOCK = 1024


@triton.jit
def _prepare_steps_kernel(pointers, scalars, bias_corrections, num_tensors: tl.constexpr,
                          BLOCK_TENSORS: tl.constexpr):
    tensor_ids = tl.arange(0, BLOCK_TENSORS)
    mask = tensor_ids < num_tensors
    step_addresses = tl.load(pointers + tensor_ids * 5 + 4, mask, other=0).to(tl.uint64)
    step_ptrs = step_addresses.to(tl.pointer_type(tl.float32))
    step = tl.load(step_ptrs, mask, other=0.0)
    step = libdevice.add_rn(step, 1.0)
    tl.store(step_ptrs, step, mask)

    beta1 = tl.load(scalars + 1).to(tl.float64)
    beta2 = tl.load(scalars + 2).to(tl.float64)
    step_f64 = step.to(tl.float64)
    pow_beta1 = libdevice.pow(beta1, step_f64).to(tl.float32)
    pow_beta2 = libdevice.pow(beta2, step_f64).to(tl.float32)
    bias_correction1 = libdevice.add_rn(1.0, -pow_beta1)
    bias_correction2 = libdevice.add_rn(1.0, -pow_beta2)
    bias_correction2_sqrt = libdevice.sqrt_rn(bias_correction2)
    tl.store(bias_corrections + tensor_ids * 2, bias_correction1, mask)
    tl.store(bias_corrections + tensor_ids * 2 + 1, bias_correction2_sqrt, mask)


@triton.jit
def _clip_adamw_multi_kernel(pointers, lengths, block_tensor_ids, block_offsets, scalars,
                             clip_coef, bias_corrections, BLOCK: tl.constexpr):
    block_id = tl.program_id(0)
    tensor_id = tl.load(block_tensor_ids + block_id)
    base = tl.load(block_offsets + block_id)
    length = tl.load(lengths + tensor_id)
    offsets = base + tl.arange(0, BLOCK)
    mask = offsets < length

    param_ptr = tl.cast(tl.load(pointers + tensor_id * 5 + 0), tl.pointer_type(tl.float32))
    grad_ptr = tl.cast(tl.load(pointers + tensor_id * 5 + 1), tl.pointer_type(tl.float32))
    avg_ptr = tl.cast(tl.load(pointers + tensor_id * 5 + 2), tl.pointer_type(tl.float32))
    avg_sq_ptr = tl.cast(tl.load(pointers + tensor_id * 5 + 3), tl.pointer_type(tl.float32))

    lr = tl.load(scalars + 0)
    beta1 = tl.load(scalars + 1)
    beta2 = tl.load(scalars + 2)
    weight_decay = tl.load(scalars + 3)
    eps = tl.load(scalars + 4)
    clip = tl.load(clip_coef)

    param = tl.load(param_ptr + offsets, mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask, other=0.0)
    avg = tl.load(avg_ptr + offsets, mask, other=0.0)
    avg_sq = tl.load(avg_sq_ptr + offsets, mask, other=0.0)
    grad = libdevice.mul_rn(grad, clip)

    if weight_decay != 0.0:
        decay = libdevice.mul_rn(lr, weight_decay)
        param = libdevice.fma(-decay, param, param)

    inner_avg = libdevice.fma(-beta1, grad, grad)
    avg = libdevice.fma(beta1, avg, inner_avg)
    grad_sq = libdevice.mul_rn(grad, grad)
    inner_avg_sq = libdevice.fma(-beta2, grad_sq, grad_sq)
    avg_sq = libdevice.fma(beta2, avg_sq, inner_avg_sq)

    bias_correction1 = tl.load(bias_corrections + tensor_id * 2)
    bias_correction2_sqrt = tl.load(bias_corrections + tensor_id * 2 + 1)
    step_size = libdevice.div_rn(lr, bias_correction1)
    denom = libdevice.div_rn(libdevice.sqrt_rn(avg_sq), bias_correction2_sqrt)
    denom = libdevice.add_rn(denom, eps)
    update = libdevice.div_rn(libdevice.mul_rn(step_size, avg), denom)
    param = libdevice.add_rn(param, -update)

    tl.store(param_ptr + offsets, param, mask)
    tl.store(grad_ptr + offsets, grad, mask)
    tl.store(avg_ptr + offsets, avg, mask)
    tl.store(avg_sq_ptr + offsets, avg_sq, mask)


class ClipAdamWPlan:
    """Precomputed launch layout for one parameter group of CUDA FP32 tensors.

    Each call accepts the current parameter, gradient, moment, and step tensors. Gradients and
    state pointers are rebuilt on every call, so replacing a gradient cannot leave a stale pointer.
    The caller supplies the already-computed global-norm clip coefficient; norm calculation and
    clip policy remain outside this kernel.
    """

    def __init__(self, params):
        if not params:
            raise ValueError("at least one parameter is required")
        first = params[0]
        if not first.is_cuda or first.dtype != torch.float32 or not first.is_contiguous():
            raise TypeError("parameters must be contiguous CUDA FP32 tensors")
        self.device = first.device
        self.shapes = tuple(tuple(param.shape) for param in params)
        lengths = [param.numel() for param in params]
        if any(param.device != self.device or param.dtype != torch.float32 or not param.is_contiguous()
               for param in params):
            raise TypeError("parameters must be contiguous CUDA FP32 tensors on one device")

        tensor_ids = []
        offsets = []
        for tensor_id, length in enumerate(lengths):
            for offset in range(0, length, _BLOCK):
                tensor_ids.append(tensor_id)
                offsets.append(offset)
        self.lengths = torch.tensor(lengths, dtype=torch.int32, device=self.device)
        self.block_tensor_ids = torch.tensor(tensor_ids, dtype=torch.int32, device=self.device)
        self.block_offsets = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        self.num_tensors = len(lengths)
        self._block_tensors = triton.next_power_of_2(self.num_tensors)
        self.bias_corrections = torch.empty((self.num_tensors, 2), dtype=torch.float32, device=self.device)
        self._scalars = None
        self._hparams = None

    def _validate(self, params, grads, exp_avg, exp_avg_sq, steps, clip_coef):
        tensor_lists = (params, grads, exp_avg, exp_avg_sq, steps)
        if any(len(items) != self.num_tensors for items in tensor_lists):
            raise ValueError("parameter, gradient, moment, and step lists must match the plan")
        for index, (param, grad, avg, avg_sq, step, shape) in enumerate(
            zip(params, grads, exp_avg, exp_avg_sq, steps, self.shapes, strict=True)
        ):
            for name, tensor in (("parameter", param), ("gradient", grad), ("exp_avg", avg), ("exp_avg_sq", avg_sq)):
                if tensor.device != self.device or tensor.dtype != torch.float32 or not tensor.is_contiguous():
                    raise TypeError(f"{name} {index} must be contiguous CUDA FP32 on the plan device")
                if tuple(tensor.shape) != shape:
                    raise ValueError(f"{name} {index} shape does not match the plan")
            if step.device != self.device or step.dtype != torch.float32 or step.numel() != 1:
                raise TypeError(f"step {index} must be a CUDA FP32 scalar on the plan device")
        if clip_coef.device != self.device or clip_coef.dtype != torch.float32 or clip_coef.numel() != 1:
            raise TypeError("clip_coef must be a CUDA FP32 scalar on the plan device")

    def step_(self, params, grads, exp_avg, exp_avg_sq, steps, *, lr, betas, eps, weight_decay,
              clip_coef):
        """Clip gradients and apply one in-place AdamW update across the parameter group."""
        self._validate(params, grads, exp_avg, exp_avg_sq, steps, clip_coef)
        beta1, beta2 = betas
        hparams = (float(lr), float(beta1), float(beta2), float(weight_decay), float(eps))
        if self._hparams != hparams:
            self._scalars = torch.tensor(hparams, dtype=torch.float32, device=self.device)
            self._hparams = hparams

        addresses = []
        for tensors in zip(params, grads, exp_avg, exp_avg_sq, steps, strict=True):
            addresses.extend(tensor.data_ptr() for tensor in tensors)
        pointers = torch.tensor(addresses, dtype=torch.int64, device=self.device)

        _prepare_steps_kernel[(1,)](
            pointers,
            self._scalars,
            self.bias_corrections,
            self.num_tensors,
            self._block_tensors,
            num_warps=4,
        )
        _clip_adamw_multi_kernel[(self.block_tensor_ids.numel(),)](
            pointers,
            self.lengths,
            self.block_tensor_ids,
            self.block_offsets,
            self._scalars,
            clip_coef,
            self.bias_corrections,
            BLOCK=_BLOCK,
            num_warps=4,
        )
        for grad in grads:
            torch.autograd.graph.increment_version(grad)
        for step in steps:
            torch.autograd.graph.increment_version(step)
