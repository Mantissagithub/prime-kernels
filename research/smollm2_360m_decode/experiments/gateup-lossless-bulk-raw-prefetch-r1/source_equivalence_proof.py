#!/usr/bin/env python3
"""Prove the bulk raw-prefetch kernel is a narrow edit to frozen SIMD source."""
import hashlib
import json
import re
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gateup-lossless-simd-decode-r1" / "projection_probe.cu"
CANDIDATE = HERE / "projection_probe.cu"
BASE_SHA256 = "e76c27c96d1f43948bf79eae95adc0eccd697b533e989593ca135b65a8a85975"


def extract(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise ValueError(f"unterminated function: {signature}")


base = BASE.read_text()
candidate = CANDIDATE.read_text()
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA256
control_kernel = extract(base, "__global__ void gateup_lossless_simd_kernel(")
assert extract(candidate, "__global__ void gateup_lossless_simd_kernel(") == control_kernel

expected = control_kernel.replace(
    "__global__ void gateup_lossless_simd_kernel(",
    "__global__ void gateup_lossless_bulk_raw_kernel(", 1,
).replace(
    "const bf16* x, float* y) {",
    "const bf16* x, float* y, size_t raw_bytes) {",
    1,
)
csr_prefetch = """        const uint32_t first = __ldg(raw_offsets + row);
        const uint32_t last = __ldg(raw_offsets + row + 1);
        if (last > first) prefetch_range_l2(raw + size_t(first) * 8, size_t(last - first) * 16);
"""
flat_prefetch = """  const uintptr_t raw_begin = reinterpret_cast<uintptr_t>(raw) & ~uintptr_t(127);
  const uintptr_t raw_end = (reinterpret_cast<uintptr_t>(raw) + raw_bytes + 127) & ~uintptr_t(127);
  const uintptr_t raw_first = raw_begin +
      (static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x) * 128;
  const uintptr_t raw_stride = static_cast<size_t>(kCTAs) * kThreads * 128;
  for (uintptr_t p = raw_first; p < raw_end; p += raw_stride)
    asm volatile("prefetch.global.L2 [%0];" : : "l"(p));"""
assert expected.count(csr_prefetch) == 1
expected = expected.replace(csr_prefetch, "", 1)
prefetch_site = "  __syncthreads();\n"
assert expected.count(prefetch_site) == 1
expected = expected.replace(prefetch_site, prefetch_site + flat_prefetch + "\n", 1)
candidate_kernel = extract(candidate, "__global__ void gateup_lossless_bulk_raw_kernel(")
if candidate_kernel != expected:
    raise AssertionError("\n".join(difflib.unified_diff(
        expected.splitlines(), candidate_kernel.splitlines(),
        fromfile="expected", tofile="candidate", lineterm="")))
assert "extern \"C\" int run_gateup_lossless_bulk_raw(" in candidate
assert "size_t raw_bytes,\n                                             cudaStream_t stream)" in candidate

print(json.dumps({
    "frozen_source_sha256": BASE_SHA256,
    "original_simd_kernel_unchanged": True,
    "candidate_difference": "replace three per-row CSR raw-prefetch lines with one flat raw-table prefetch block and add the raw_bytes scalar",
    "candidate_projection_math_and_decode": "identical to frozen SIMD kernel",
    "candidate_abi": "run_gateup_lossless_bulk_raw has plane, headers, raw, offsets, input, output, raw_bytes, stream",
}, indent=2))
