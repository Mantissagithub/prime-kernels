"""Experimental SmolLM2-360M fused decode step for SM 8.9.

This package is intentionally not registered for serving: its archived R7 kernel
fails six strict gates in the 30-case held-out numerical suite.

The returned tensor is a view into a shared per-device output buffer and is
overwritten by the next call on that device. Consume or copy it before then.
"""

import ctypes
import hashlib
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension

SOURCE = Path(__file__).resolve().parent / "csrc" / "block360.cu"
SOURCE_SHA256 = "0f24bb3a9502310e1e25fcd7c40f0ec299dd2a0a018a5b2cd4850f231e709f66"
INCLUDE_SHA256 = "5c7b651550e07f57affd81f90c814bd561e12ea7d838e50b0e7878777fb2ad7d"
CACHE_ROOT = Path.home() / ".cache" / "prime_kernels" / "decode_megakernel_360m"
NVCC_FLAGS = ["-O3", "-arch=sm_89", "-std=c++17", "-lineinfo"]

MAX_BATCH = 2
MAX_SEQ = 8192
HIDDEN = 960
INTERMEDIATE = 2560
LAYERS = 32
KV_HEADS = 5
HEAD_DIM = 64
BLOCK_SIZE = 16
EPS = 1e-5

_lib = None
_runtimes = {}


class _Runtime:
    def __init__(self, lib, device):
        self.lib = lib
        self.device = device
        self.scratch = torch.empty(lib.decode360_workspace_bytes(), dtype=torch.uint8, device=device)
        self.out = torch.empty((MAX_BATCH, HIDDEN), dtype=torch.bfloat16, device=device)
        self.input_ids_i32 = torch.empty((MAX_BATCH,), dtype=torch.int32, device=device)
        self.layer_outputs = None
        self.layers_t = None
        self.kv_t = None
        self.table_key = None


def _library():
    global _lib
    if _lib is None:
        include = SOURCE.with_name("block360_kernel.inc")
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError(f"{SOURCE} does not match archived R9 source")
        if hashlib.sha256(include.read_bytes()).hexdigest() != INCLUDE_SHA256:
            raise RuntimeError(f"{include} does not match archived R9 include")
        if shutil.which("ninja") is None:
            os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
        build_dir = CACHE_ROOT / f"r9-{SOURCE_SHA256[:16]}-{INCLUDE_SHA256[:8]}"
        build_dir.mkdir(parents=True, exist_ok=True)
        library = cpp_extension.load(
            name="prime_decode_megakernel_360m_r9",
            sources=[str(SOURCE)],
            extra_cuda_cflags=NVCC_FLAGS,
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=False,
        )
        lib = ctypes.CDLL(str(library))
        vp, ci, ll, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_size_t
        lib.decode360_workspace_bytes.argtypes = []
        lib.decode360_workspace_bytes.restype = sz
        lib.decode360_debug_bytes.argtypes = []
        lib.decode360_debug_bytes.restype = sz
        lib.decode360.argtypes = [vp, ci, vp, vp, vp, vp, vp, vp, ci, vp, vp, vp,
                                  ll, ll, ll, ll, ll, ci, vp, ci, vp, vp, vp, sz,
                                  vp, sz, ci, ci]
        lib.decode360.restype = ci
        lib.decode360_status_string.argtypes = [ci]
        lib.decode360_status_string.restype = ctypes.c_char_p
        _lib = lib
    return _lib


def _runtime(device):
    if device not in _runtimes:
        _runtimes[device] = _Runtime(_library(), device)
    return _runtimes[device]


def _check_tensor(tensor, name, *, dtype=None, device=None, shape=None):
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")


def _tables(rt, layers, kv_caches):
    key = (tuple(t.data_ptr() for layer in layers for t in layer),
           tuple(t.data_ptr() for t in kv_caches))
    if key != rt.table_key:
        layer_table = torch.tensor([[t.data_ptr() for t in layer] for layer in layers], dtype=torch.int64)
        kv_table = torch.tensor([t.data_ptr() for t in kv_caches], dtype=torch.int64)
        if rt.layers_t is None:
            rt.layers_t = layer_table.to(rt.device)
            rt.kv_t = kv_table.to(rt.device)
        else:
            rt.layers_t.copy_(layer_table)
            rt.kv_t.copy_(kv_table)
        rt.table_key = key
    return rt.layers_t, rt.kv_t


def decode_step(*, input_ids, positions, seq_lens, block_table, slot_mapping, embed,
                layers, final_norm, kv_caches, cos_sin_cache, eps,
                layer_outputs=None, debug=None, debug_layer_index=-1):
    """Run one experimental 360M decode step; inputs follow vLLM FLASH_ATTN metadata."""
    batch = input_ids.shape[0]
    if not 1 <= batch <= MAX_BATCH:
        raise ValueError(f"batch must be 1..{MAX_BATCH}, got {batch}")
    if eps != EPS:
        raise ValueError(f"the kernel is built for rms eps {EPS}, got {eps}")
    if len(layers) != LAYERS or len(kv_caches) != LAYERS:
        raise ValueError(f"expected {LAYERS} layers and KV caches")
    if debug_layer_index < -1 or debug_layer_index >= LAYERS:
        raise ValueError("debug_layer_index must be -1 or a layer index 0..31")

    device = input_ids.device
    _check_tensor(input_ids, "input_ids", device=device)
    if input_ids.dtype not in (torch.int32, torch.int64) or input_ids.ndim != 1:
        raise ValueError("input_ids must be a 1-D int32 or int64 tensor")
    _check_tensor(positions, "positions", dtype=torch.int64, device=device, shape=(batch,))
    _check_tensor(seq_lens, "seq_lens", dtype=torch.int32, device=device, shape=(batch,))
    _check_tensor(slot_mapping, "slot_mapping", dtype=torch.int64, device=device, shape=(batch,))
    if block_table.dtype != torch.int32 or block_table.device != device or block_table.ndim != 2 or block_table.shape[0] < batch:
        raise ValueError("block_table must be int32 [B, width] on the input device")
    if block_table.stride(1) != 1:
        raise ValueError("block_table must have contiguous rows")
    if not 1 <= block_table.shape[1] * BLOCK_SIZE <= MAX_SEQ:
        raise ValueError(f"block_table context capacity must be 1..{MAX_SEQ}")
    _check_tensor(embed, "embed", dtype=torch.bfloat16, device=device)
    _check_tensor(final_norm, "final_norm", dtype=torch.bfloat16, device=device, shape=(HIDDEN,))
    if embed.ndim != 2 or embed.shape[1] != HIDDEN:
        raise ValueError("embed must be BF16 [vocab, 960]")
    for i, layer in enumerate(layers):
        if len(layer) != 6:
            raise ValueError(f"layer {i} must contain six weight tensors")
        expected = ((HIDDEN,), (HIDDEN + 2 * KV_HEADS * HEAD_DIM, HIDDEN),
                    (HIDDEN, HIDDEN), (HIDDEN,), (2 * INTERMEDIATE, HIDDEN),
                    (HIDDEN, INTERMEDIATE))
        for name, tensor, shape in zip(("norm1", "qkv", "o", "norm2", "gate_up", "down"), layer, expected):
            _check_tensor(tensor, f"layer {i} {name}", dtype=torch.bfloat16, device=device, shape=shape)
            if not tensor.is_contiguous():
                raise ValueError(f"layer {i} {name} must be contiguous")
    if len(kv_caches) != LAYERS:
        raise ValueError(f"expected {LAYERS} KV caches")
    kv0 = kv_caches[0]
    _check_tensor(kv0, "kv_caches[0]", dtype=torch.bfloat16, device=device)
    if kv0.ndim != 4 or kv0.shape[1] != KV_HEADS or kv0.shape[3] != 2 * HEAD_DIM or kv0.stride(3) != 1:
        raise ValueError("KV caches must use vLLM FLASH_ATTN [blocks, 5, 16, 128] K/V-packed layout")
    if kv0.shape[2] != BLOCK_SIZE:
        raise ValueError("the kernel requires KV block_size=16")
    for i, kv in enumerate(kv_caches):
        _check_tensor(kv, f"kv_caches[{i}]", dtype=torch.bfloat16, device=device, shape=tuple(kv0.shape))
        if kv.stride() != kv0.stride():
            raise ValueError("all KV caches must share strides")
    _check_tensor(cos_sin_cache, "cos_sin_cache", device=device)
    if cos_sin_cache.dtype not in (torch.bfloat16, torch.float32) or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != HEAD_DIM:
        raise ValueError("cos_sin_cache must be BF16 or FP32 [max_position, 64]")
    if layer_outputs is not None:
        _check_tensor(layer_outputs, "layer_outputs", dtype=torch.bfloat16, device=device,
                      shape=(LAYERS, batch, HIDDEN))
        if not layer_outputs.is_contiguous():
            raise ValueError("layer_outputs must be contiguous")
    if debug is not None:
        _check_tensor(debug, "debug", dtype=torch.uint8, device=device)
        if not debug.is_contiguous():
            raise ValueError("debug must be contiguous raw byte storage")

    rt = _runtime(device)
    if debug is not None:
        if debug.numel() < rt.lib.decode360_debug_bytes():
            raise ValueError("debug buffer is too small")
        if debug.data_ptr() % 16:
            raise ValueError("debug buffer must be 16-byte aligned")
    layers_t, kv_t = _tables(rt, layers, kv_caches)
    stream = torch.cuda.current_stream(device).cuda_stream
    if input_ids.dtype == torch.int64:
        rt.input_ids_i32[:batch].copy_(input_ids)
        input_ids_ptr = rt.input_ids_i32.data_ptr()
    else:
        input_ids_ptr = input_ids.data_ptr()
    rc = rt.lib.decode360(
        stream, batch, input_ids_ptr, embed.data_ptr(), final_norm.data_ptr(),
        positions.data_ptr(), seq_lens.data_ptr(), block_table.data_ptr(), block_table.stride(0),
        slot_mapping.data_ptr(), layers_t.data_ptr(), kv_t.data_ptr(),
        0, HEAD_DIM, kv0.stride(0), kv0.stride(2), kv0.stride(1), BLOCK_SIZE,
        cos_sin_cache.data_ptr(), int(cos_sin_cache.dtype == torch.float32), rt.out.data_ptr(),
        layer_outputs.data_ptr() if layer_outputs is not None else None,
        rt.scratch.data_ptr(), rt.scratch.numel(), debug.data_ptr() if debug is not None else None,
        debug.numel() * debug.element_size() if debug is not None else 0,
        block_table.shape[1] * BLOCK_SIZE, debug_layer_index,
    )
    if rc != 0:
        raise RuntimeError(f"decode_megakernel_360m failed: {rt.lib.decode360_status_string(rc).decode()}")
    return rt.out[:batch]
