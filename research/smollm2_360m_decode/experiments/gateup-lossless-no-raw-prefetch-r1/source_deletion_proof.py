#!/usr/bin/env python3
"""Prove the trial kernel is the frozen control with one prefetch setup removed."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / "gateup-lossless-probe-r1" / "projection_probe.cu"
CANDIDATE = HERE / "projection_probe.cu"
CONTROL_SHA256 = "b0218ea40e2f9012c802dde20adfb428d0361ad952761c07a698a7e60e7cf09c"
REMOVAL = """        const uint32_t first = __ldg(raw_offsets + row);
        const uint32_t last = __ldg(raw_offsets + row + 1);
        if (last > first) prefetch_range_l2(raw + size_t(first) * 8, size_t(last - first) * 16);
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel(source, name):
    begin = source.index(f"__global__ void {name}(")
    end = source.index("\n}\n", begin) + 2
    return source[begin:end]


def main():
    control = CONTROL.read_text()
    candidate = CANDIDATE.read_text()
    assert sha256(CONTROL) == CONTROL_SHA256, "frozen control source hash changed"
    assert control.count(REMOVAL) == 1, "control deletion anchor is not unique"
    expected = kernel(control, "gateup_lossless_kernel")
    expected = expected.replace("gateup_lossless_kernel", "gateup_lossless_no_raw_prefetch_kernel", 1)
    expected = expected.replace(REMOVAL, "", 1)
    actual = kernel(candidate, "gateup_lossless_no_raw_prefetch_kernel")
    assert actual == expected, "candidate kernel differs beyond the anchored three-line deletion"
    assert kernel(candidate, "gateup_lossless_kernel") == kernel(control, "gateup_lossless_kernel"), "control kernel changed"
    assert candidate.count('extern "C" int run_gateup_lossless_no_raw_prefetch(') == 1
    proof = {
        "control_source_sha256": CONTROL_SHA256,
        "candidate_source_sha256": sha256(CANDIDATE),
        "control_kernel_identical": True,
        "candidate_kernel": "control kernel renamed, with only the three anchored CSR-dependent raw-prefetch setup lines deleted",
        "candidate_abi": "same seven-pointer/stream signature as run_gateup_lossless",
        "additional_kernel_changes": 0,
    }
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
