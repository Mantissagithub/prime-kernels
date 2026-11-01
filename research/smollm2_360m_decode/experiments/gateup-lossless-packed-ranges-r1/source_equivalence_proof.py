#!/usr/bin/env python3
"""Check packed-range source against the frozen predicated-decode control."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = (HERE.parents[2] / "outputs/megakernel-20261025/gateup-lossless-predicated-decode-r1/projection_probe.cu")
EXPECTED_BASELINE_SHA256 = "0b56cdc994d445ddcb52f9ffb91565e80cf4de1d9b89723c58a5daa12f8c80f9"
baseline = BASELINE.read_text()
candidate = (HERE / "projection_probe.cu").read_text()
baseline_hash = hashlib.sha256(baseline.encode()).hexdigest()
assert baseline_hash == EXPECTED_BASELINE_SHA256, baseline_hash
assert candidate.startswith(baseline), "original control source was modified"

start = baseline.index("__global__ void gateup_lossless_predicated_kernel(")
end = baseline.index("\n__global__ void gateup_lossless_kernel(", start)
expected_kernel = baseline[start:end]
old = """        const uint32_t first = __ldg(raw_offsets + row);
        const uint32_t last = __ldg(raw_offsets + row + 1);
        if (last > first) prefetch_range_l2(raw + size_t(first) * 8, size_t(last - first) * 16);"""
new = """        const uint32_t range = __ldg(packed_ranges + row);
        const uint32_t first = range & 0xFFFFFFu;
        const uint32_t count = range >> 24;
        if (count) prefetch_range_l2(raw + size_t(first) * 8, size_t(count) * 16);"""
assert expected_kernel.count(old) == 1
expected_kernel = expected_kernel.replace("gateup_lossless_predicated_kernel",
                                          "gateup_lossless_packed_ranges_kernel", 1)
expected_kernel = expected_kernel.replace("const uint32_t* raw_offsets",
                                          "const uint32_t* packed_ranges", 1)
expected_kernel = expected_kernel.replace(old, new, 1)

candidate_start = candidate.index("__global__ void gateup_lossless_packed_ranges_kernel(", len(baseline))
candidate_end = candidate.index("\nextern \"C\" int run_gateup_lossless_packed_ranges(", candidate_start)
actual_kernel = candidate[candidate_start:candidate_end]
assert actual_kernel.rstrip("\n") == expected_kernel.rstrip("\n"), "candidate kernel differs beyond packed range prefetch lookup"
assert candidate.count("__ldg(raw_offsets + row)") == 3, "all original kernels must remain intact"
assert candidate.count("__ldg(packed_ranges + row)") == 1
assert "decode_chunk_predicated(" in actual_kernel
assert "decode_chunk_simd(" not in actual_kernel

export = candidate[candidate_end:].strip()
expected_export = """extern "C" int run_gateup_lossless_packed_ranges(const void* plane, const void* headers,
                                                  const void* raw, const void* packed_ranges,
                                                  const void* input, void* output, cudaStream_t stream) {
  gateup_lossless_packed_ranges_kernel<<<kCTAs, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<const uint32_t*>(packed_ranges),
      static_cast<const bf16*>(input), static_cast<float*>(output));
  return static_cast<int>(cudaGetLastError());
}"""
assert export == expected_export
assert "run_gateup_lossless_packed_ranges" in candidate

result = {
    "baseline_source_sha256": baseline_hash,
    "candidate_source_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
    "baseline_source_is_exact_candidate_prefix": True,
    "candidate_kernel_is_control_clone_with_one_prefetch_range_change": True,
    "candidate_export_uses_same_seven_argument_pointer_abi": True,
    "decode_helper_and_projection_arithmetic_unchanged": True,
    "mismatches": 0,
}
(HERE / "source-equivalence-proof.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
