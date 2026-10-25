"""Batched decode step for SmolLM2-135M on SM 8.9 (csrc/megakernel.cu, R5 stage A).

csrc/megakernel.cu is a byte-identical copy of the R5 kernel,
sha256 170d6b5c651d69664325d8d1fdc1d718ef0b8e01a9254b4e1f1730d2e095dc0d.
The kernel is compiled with nvcc on first use into ~/.cache/prime_kernels/decode_megakernel/<hash>
and loaded through ctypes. It is one cooperative launch per call (24 CTAs x 256 threads).

decode_step(**kwargs) takes the contract of integration-spec.md. Requirements:
- input_ids int32 or int64 [B]; positions int64 [B]; seq_lens int32 [B]; slot_mapping int64 [B] (-1 skips
  the K/V write); block_table int32 [B, W] with row stride W, so the context limit is W * block_size <= 8192.
- layers: list of 30 tuples (input_layernorm, qkv, o, post_attention_layernorm, gate_up, down), BF16 CUDA.
- kv_caches: one BF16 CUDA tensor per layer shaped [blocks, kv_heads=3, block_size, 128] with K in
  elements 0..63 and V in 64..127 (vLLM FLASH_ATTN layout; strides are read from the tensor).
- cos_sin_cache BF16 or FP32 [max_pos, 64]; eps must be 1e-5.

Steady state allocates nothing: the scratch workspace and output buffer are created on the first call, and
the device pointer tables are rebuilt only when a layer or cache data pointer changes. Padded rows with
slot_mapping -1 skip the cache write. The returned tensor is a view of a shared output buffer that the next
call overwrites, so consume it before the next step.
"""

import ctypes
import hashlib
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension

SOURCE = Path(__file__).resolve().parent / "csrc" / "megakernel.cu"
SOURCE_SHA256 = "170d6b5c651d69664325d8d1fdc1d718ef0b8e01a9254b4e1f1730d2e095dc0d"  # final R5 source, built with -DMK_PREFETCH=1
CACHE_ROOT = Path.home() / ".cache" / "prime_kernels" / "decode_megakernel"
NVCC_FLAGS = ["-O3", "-arch=sm_89", "-std=c++17", "-lineinfo", "-DMK_PREFETCH=1"]  # MK_COUNTER_SYNC stays off

MAX_BATCH = 4
MAX_SEQ = 8192
HIDDEN = 576
KV_HEADS = 3
HEAD_DIM = 64
LAYERS = 30
EPS = 1e-5

_lib = None
_runtimes = {}


class _Runtime:
    """Per-device scratch, output buffer and cached pointer tables."""

    def __init__(self, lib, device):
        self.lib = lib
        self.scratch = torch.empty(lib.mk_workspace_bytes(MAX_BATCH, MAX_SEQ), dtype=torch.uint8, device=device)
        self.out = torch.empty((MAX_BATCH, HIDDEN), dtype=torch.bfloat16, device=device)
        self.device = device
        self.table_key = None
        self.layers_t = None
        self.kv_t = None


def _library():
    global _lib
    if _lib is None:
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError(f"{SOURCE} does not match the R5 kernel sha256 {SOURCE_SHA256}")
        if shutil.which("ninja") is None:
            # torch's extension builder shells out to ninja; the venv that provides it may not be on PATH.
            os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
        build_dir = CACHE_ROOT / f"{SOURCE_SHA256[:16]}-prefetch"
        build_dir.mkdir(parents=True, exist_ok=True)
        library = cpp_extension.load(
            name="prime_decode_megakernel",
            sources=[str(SOURCE)],
            extra_cuda_cflags=NVCC_FLAGS,
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=False,
        )
        lib = ctypes.CDLL(str(library))
        vp, ci, ll, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_size_t
        lib.mk_query.argtypes = [ctypes.POINTER(ci)] * 7
        lib.mk_query.restype = ci
        lib.mk_workspace_bytes.argtypes = [ci, ci]
        lib.mk_workspace_bytes.restype = sz
        lib.mk_decode.argtypes = [vp, ci, vp, ci, vp, vp, vp, ci, vp, vp, vp, vp, vp,
                                  ll, ll, ll, ll, ll, ci, vp, ci, vp, vp, sz, ci]
        lib.mk_decode.restype = ci
        lib.mk_status_string.argtypes = [ci]
        lib.mk_status_string.restype = ctypes.c_char_p
        _lib = lib
    return _lib


def _runtime(device):
    if device not in _runtimes:
        _runtimes[device] = _Runtime(_library(), device)
    return _runtimes[device]


def _check_kv(kv_caches):
    first = kv_caches[0]
    if first.dtype != torch.bfloat16 or first.dim() != 4:
        raise ValueError("kv_caches must be 4-D BF16 tensors")
    if first.shape[1] != KV_HEADS or first.shape[3] != 2 * HEAD_DIM or first.stride(3) != 1:
        raise ValueError(f"kv_caches must be [blocks, {KV_HEADS}, block_size, {2 * HEAD_DIM}] with unit last stride")
    for kv in kv_caches:
        if kv.shape != first.shape or kv.stride() != first.stride() or kv.dtype != first.dtype:
            raise ValueError("all layers must share one KV cache shape, dtype and strides")
        if not kv.is_cuda:
            raise ValueError("kv_caches must be CUDA tensors")


def _tables(rt, layers, kv_caches):
    """Device pointer tables for the layer weights and the per-layer KV base pointers."""
    key = (
        tuple(t.data_ptr() for layer in layers for t in layer),
        tuple(kv.data_ptr() for kv in kv_caches),
    )
    if key != rt.table_key:
        layer_table = torch.tensor([[t.data_ptr() for t in layer] for layer in layers], dtype=torch.int64)
        kv_table = torch.tensor([kv.data_ptr() for kv in kv_caches], dtype=torch.int64)
        if rt.layers_t is None:
            rt.layers_t = layer_table.to(rt.device)
            rt.kv_t = kv_table.to(rt.device)
        else:
            # Update in place so CUDA graphs captured with these tables see the new pointers.
            rt.layers_t.copy_(layer_table)
            rt.kv_t.copy_(kv_table)
        rt.table_key = key
    return rt.layers_t, rt.kv_t


def decode_step(
    *,
    input_ids,
    positions,
    seq_lens,
    block_table,
    slot_mapping,
    embed,
    layers,
    final_norm,
    kv_caches,
    cos_sin_cache,
    eps,
):
    batch = input_ids.shape[0]
    if not 1 <= batch <= MAX_BATCH:
        raise ValueError(f"batch must be 1..{MAX_BATCH}, got {batch}")
    if eps != EPS:
        raise ValueError(f"the kernel is built for rms eps {EPS}, got {eps}")
    if len(layers) != LAYERS or len(kv_caches) != LAYERS:
        raise ValueError(f"expected {LAYERS} layers and KV caches")
    if input_ids.dtype not in (torch.int32, torch.int64) or positions.dtype != torch.int64:
        raise ValueError("input_ids must be int32 or int64 and positions int64")
    if seq_lens.dtype != torch.int32 or block_table.dtype != torch.int32 or slot_mapping.dtype != torch.int64:
        raise ValueError("seq_lens and block_table must be int32, slot_mapping int64")
    if block_table.stride(1) != 1 or block_table.shape[0] < batch:
        raise ValueError("block_table rows must be contiguous and cover the batch")
    if cos_sin_cache.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("cos_sin_cache must be BF16 or FP32")
    _check_kv(kv_caches)

    device = input_ids.device
    kv0 = kv_caches[0]
    block_size = kv0.shape[2]
    max_seq = block_table.shape[1] * block_size
    if max_seq > MAX_SEQ:
        raise ValueError(f"context limit {max_seq} exceeds the kernel's {MAX_SEQ}")

    rt = _runtime(device)
    layers_t, kv_t = _tables(rt, layers, kv_caches)
    lib = rt.lib
    stream = torch.cuda.current_stream(device).cuda_stream
    rc = lib.mk_decode(
        stream,
        batch,
        input_ids.data_ptr(),
        int(input_ids.dtype == torch.int64),
        positions.data_ptr(),
        seq_lens.data_ptr(),
        block_table.data_ptr(),
        block_table.stride(0),
        slot_mapping.data_ptr(),
        layers_t.data_ptr(),
        embed.data_ptr(),
        final_norm.data_ptr(),
        kv_t.data_ptr(),
        0,
        kv0.shape[3] // 2,
        kv0.stride(0),
        kv0.stride(2),
        kv0.stride(1),
        block_size,
        cos_sin_cache.data_ptr(),
        int(cos_sin_cache.dtype == torch.float32),
        rt.out.data_ptr(),
        rt.scratch.data_ptr(),
        rt.scratch.numel(),
        max_seq,
    )
    if rc != 0:
        raise RuntimeError(f"decode_megakernel failed: {lib.mk_status_string(rc).decode()}")
    return rt.out[:batch]
