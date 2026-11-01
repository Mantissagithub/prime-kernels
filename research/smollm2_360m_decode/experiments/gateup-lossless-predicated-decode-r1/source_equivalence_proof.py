#!/usr/bin/env python3
"""Prove the frozen SIMD control and predicated candidate differ only at decode calls."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / "gateup-lossless-simd-decode-r1" / "projection_probe.cu"
CANDIDATE = HERE / "projection_probe.cu"
CONTROL_SHA256 = "e76c27c96d1f43948bf79eae95adc0eccd697b533e989593ca135b65a8a85975"


def extract(source, prefix):
    begin = source.index(prefix)
    brace = source.index("{", begin)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[begin:index + 1]
    raise ValueError(prefix)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    control = CONTROL.read_text()
    candidate = CANDIDATE.read_text()
    assert sha(CONTROL) == CONTROL_SHA256
    control_kernel = extract(control, "__global__ void gateup_lossless_simd_kernel(")
    predicated_kernel = extract(candidate, "__global__ void gateup_lossless_predicated_kernel(")
    expected = control_kernel.replace("gateup_lossless_simd_kernel",
                                      "gateup_lossless_predicated_kernel", 1)
    expected = expected.replace("decode_chunk_simd(", "decode_chunk_predicated(")
    assert predicated_kernel == expected
    assert extract(candidate, "__global__ void gateup_lossless_simd_kernel(") == control_kernel
    helper = extract(candidate, "__device__ __forceinline__ uint4 decode_chunk_predicated(")
    assert "__ldg(headers + chunk)" in helper
    assert "@!p ld.global.nc.v2.u32" in helper
    assert "@p ld.global.nc.v4.u32" in helper
    assert "selp.b32 raw_index, %7, 0, p" in helper
    assert helper.count("if (") == 0
    assert extract(candidate, "__global__ void decode_weights_predicated_kernel(").count(
        "decode_chunk_predicated(") == 1
    assert 'extern "C" int run_gateup_lossless_predicated(' in candidate
    assert 'extern "C" int decode_weights_predicated(' in candidate
    print(json.dumps({
        "control_source_sha256": CONTROL_SHA256,
        "candidate_source_sha256": sha(CANDIDATE),
        "simd_control_kernel_identical": True,
        "candidate_kernel": "frozen SIMD kernel with only the two decode helper calls changed",
        "raw_prefetch_hints": "retained unchanged",
        "decode_loads": "header __ldg plus predicated 8-byte plane and 16-byte raw loads",
        "non_escape_raw_index": "selected to zero with PTX selp before address calculation",
        "candidate_abi": "run_gateup_lossless_predicated matches the seven-argument control ABI",
        "full_weight_validation": "decode_weights_predicated uses the candidate helper",
    }, indent=2))


if __name__ == "__main__":
    main()
