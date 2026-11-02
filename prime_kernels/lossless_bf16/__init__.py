"""Fixed-capacity, lossless CUDA cache for BF16 gate/up matrices."""

import ctypes
import math
import os
from pathlib import Path

import torch
from torch.utils import cpp_extension

LAYERS = 32
ROWS = 5120
COLS = 960
CHUNKS = ROWS * (COLS // 8)
SOURCE = Path(__file__).resolve().parent / "csrc" / "lossless_bf16.cu"
INCLUDE = SOURCE.with_name("encoded_gu.h")
_library = None


def _load_library():
    global _library
    if _library is None:
        if os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") is None:
            os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
        repo_root = Path(__file__).resolve().parents[3]
        build_dir = repo_root / "outputs" / "megakernel-20261025" / "lossless-codec-integration-r1" / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        library = cpp_extension.load(
            name="prime_lossless_bf16",
            sources=[str(SOURCE)],
            extra_include_paths=[str(INCLUDE.parent)],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=False,
        )
        lib = ctypes.CDLL(str(library))
        vp, sz, ui, ci = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int
        lib.lossless_scan_workspace_bytes.argtypes = []
        lib.lossless_scan_workspace_bytes.restype = sz
        lib.lossless_encode_layer.argtypes = [vp] * 8 + [sz, vp, ui, vp]
        lib.lossless_encode_layer.restype = ci
        lib.lossless_decode_layer.argtypes = [vp, vp, vp, vp, vp]
        lib.lossless_decode_layer.restype = ci
        lib.lossless_write_descriptor.argtypes = [vp, ci, vp, vp, vp, vp, vp, vp]
        lib.lossless_write_descriptor.restype = ci
        _library = lib
    return _library


def _check(rc, operation):
    if rc:
        raise RuntimeError(f"{operation} failed with CUDA error {rc}")


class EncodedGUCache:
    """Encode 32 contiguous CUDA BF16 [5120, 960] gate/up matrices."""

    def __init__(self, weights):
        if len(weights) != LAYERS:
            raise ValueError(f"weights must contain {LAYERS} layers")
        self.device = weights[0].device
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EncodedGUCache must be prepared before CUDA graph capture")
        self._validate(weights)
        self._lib = _load_library()
        self._stream = torch.cuda.current_stream(self.device)
        self._stream_handle = self._stream.cuda_stream
        self._tables = []
        self._enabled = []
        self._raw_counts = []
        self._raw_capacities = [0] * LAYERS
        self._generation = 0
        self._scan_workspace_bytes = self._lib.lossless_scan_workspace_bytes()
        self._scan_workspace = torch.empty(self._scan_workspace_bytes, dtype=torch.uint8, device=self.device)
        self._flags = torch.empty((CHUNKS,), dtype=torch.uint32, device=self.device)
        self._prefix = torch.empty((CHUNKS,), dtype=torch.uint32, device=self.device)
        self._escape_counts = torch.empty((LAYERS,), dtype=torch.uint32, device=self.device)
        for _ in range(LAYERS):
            self._tables.append({
                "plane": torch.empty((ROWS, COLS), dtype=torch.uint8, device=self.device),
                "headers": torch.empty((ROWS, COLS // 8), dtype=torch.uint32, device=self.device),
                "raw": torch.empty((8,), dtype=torch.uint16, device=self.device),
                "row_offsets": torch.empty((ROWS + 1,), dtype=torch.uint32, device=self.device),
            })
            self._enabled.append(torch.empty((), dtype=torch.uint32, device=self.device))
        self.table = torch.empty((LAYERS * 5 * ctypes.sizeof(ctypes.c_void_p),),
                                 dtype=torch.uint8, device=self.device)
        self._weights = tuple(weights)
        self._prime_capacities(self._weights)
        self._pack_all(self._weights)
        self._write_descriptors()
        self._stream.synchronize()
        self._generation = 1

    def _validate(self, weights):
        for i, tensor in enumerate(weights):
            if (not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or
                    tensor.device != self.device or tensor.dtype != torch.bfloat16 or
                    tuple(tensor.shape) != (ROWS, COLS) or not tensor.is_contiguous()):
                raise ValueError(f"weights[{i}] must be contiguous CUDA BF16 [{ROWS}, {COLS}] on {self.device}")

    def _capturing(self):
        return torch.cuda.is_current_stream_capturing()

    def _encode(self, index, tensor):
        table = self._tables[index]
        return self._lib.lossless_encode_layer(
            tensor.data_ptr(), table["plane"].data_ptr(), table["headers"].data_ptr(),
            table["raw"].data_ptr(), table["row_offsets"].data_ptr(),
            self._flags.data_ptr(), self._prefix.data_ptr(),
            self._scan_workspace.data_ptr(), self._scan_workspace_bytes,
            self._enabled[index].data_ptr(), self._raw_capacities[index], self._stream_handle)

    def _prime_capacities(self, weights):
        for i, tensor in enumerate(weights):
            _check(self._encode(i, tensor), f"count escapes for layer {i}")
            torch.cuda.synchronize(self.device)
            count = int(self._prefix[-1].item())
            capacity = max(1, math.ceil(count * 1.25))
            self._tables[i]["raw"] = torch.empty((capacity * 8,), dtype=torch.uint16, device=self.device)
            self._raw_counts.append(count)
            self._raw_capacities[i] = capacity

    def _pack_all(self, weights):
        if self._capturing():
            raise RuntimeError("EncodedGUCache.refresh cannot run during CUDA graph capture")
        self._validate(weights)
        for i, tensor in enumerate(weights):
            _check(self._encode(i, tensor), f"encode layer {i}")
        self._weights = tuple(weights)

    def _write_descriptors(self):
        for i, table in enumerate(self._tables):
            _check(self._lib.lossless_write_descriptor(
                self.table.data_ptr(), i, table["plane"].data_ptr(),
                table["headers"].data_ptr(), table["raw"].data_ptr(),
                table["row_offsets"].data_ptr(), self._enabled[i].data_ptr(),
                self._stream_handle), f"write descriptor {i}")

    @torch.inference_mode()
    def refresh(self, weights):
        """Synchronously repack existing buffers after callers quiesce decode."""
        if self._capturing():
            raise RuntimeError("EncodedGUCache.refresh cannot run during CUDA graph capture")
        self._validate(weights)
        torch.cuda.synchronize(self.device)
        self._stream = torch.cuda.current_stream(self.device)
        self._stream_handle = self._stream.cuda_stream
        for i, tensor in enumerate(weights):
            _check(self._encode(i, tensor), f"refresh layer {i}")
            self._escape_counts[i:i + 1].copy_(self._prefix[-1:])
        self._stream.synchronize()
        self._raw_counts = [int(value) for value in self._escape_counts.cpu().tolist()]
        self._weights = tuple(weights)
        self._generation += 1

    def statistics(self):
        active = [int(flag.item()) for flag in self._enabled]
        fixed_bytes = sum(t["plane"].numel() * t["plane"].element_size() +
                          t["headers"].numel() * t["headers"].element_size() +
                          t["row_offsets"].numel() * t["row_offsets"].element_size() +
                          t["raw"].numel() * t["raw"].element_size() for t in self._tables)
        return {
            "generation": self._generation,
            "active_layers": sum(active),
            "layers": LAYERS,
            "escape_chunks": list(self._raw_counts),
            "raw_capacities": list(self._raw_capacities),
            "allocated_bytes": fixed_bytes + self.table.numel() +
                              self._scan_workspace.numel() + self._flags.numel() * 4 +
                              self._prefix.numel() * 4 + self._escape_counts.numel() * 4 + LAYERS * 4,
        }

    def decode_layer(self, index):
        if not 0 <= index < LAYERS:
            raise IndexError(index)
        if not int(self._enabled[index].item()):
            return self._weights[index].clone()
        output = torch.empty((ROWS, COLS), dtype=torch.bfloat16, device=self.device)
        table = self._tables[index]
        _check(self._lib.lossless_decode_layer(table["plane"].data_ptr(),
            table["headers"].data_ptr(), table["raw"].data_ptr(),
            output.data_ptr(), self._stream_handle), f"decode layer {index}")
        return output
