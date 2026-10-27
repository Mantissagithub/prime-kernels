#!/usr/bin/env python3
"""Source-only ABI and scope checks for private R25 PV64 candidate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import ctypes
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R23 = REPO / "outputs/megakernel-20261025/decode-360-cuda-r23-a-zero"
VALIDATOR = REPO / "outputs/megakernel-20261025/block-360-heldout-r13/validate_full32.py"
SOURCE = HERE / "block360.cu"
INCLUDE = HERE / "block360_kernel.inc"
BASE_SOURCE = R23 / "block360.cu"
BASE_INCLUDE = R23 / "block360_kernel.inc"
LIBRARY = HERE / "build" / "libdecode360_r25_pv64.so"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_full32_r25", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load canonical decoder ABI validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    validator = load_validator()
    validator.validate_abi_source(INCLUDE)
    if len(validator.ABI_PARAMETERS) != 28:
        raise RuntimeError("canonical decoder ABI is not 28 parameters")
    if SOURCE.read_bytes() != BASE_SOURCE.read_bytes():
        raise RuntimeError("R25 translation unit differs from frozen R23")

    before = BASE_INCLUDE.read_text().splitlines()
    after = INCLUDE.read_text().splitlines()
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    changed = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed.append({"op": tag, "r23_lines": [i1 + 1, i2], "r25_lines": [j1 + 1, j2]})
            if i1 < 524 or i2 > 644:
                raise RuntimeError("R25 edit escaped the single B2 PV dispatch region")
    if not changed or len(changed) > 3:
        raise RuntimeError(f"expected only localized PV dispatch edits, found {len(changed)} hunks")

    text = INCLUDE.read_text()
    required = (
        "const bool use_long_pv64_schedule = max_seq > 2048 && !use_long_b2_schedule;",
        "const bool use_r21_schedule = use_long_b2_schedule;",
        "constexpr int y_chunk = 64;",
        "base += ygroups * y_chunk",
        "acc = __fmaf_rn(prob, value, acc);",
        "const float left = add_rn(add_rn(p0, p4), add_rn(p2, p6));",
        "const float right = add_rn(add_rn(p1, p5), add_rn(p3, p7));",
        "const float result = add_rn(left, right);",
        "float& acc = (j & 3) == 0 ? acc0 : (j & 3) == 1 ? acc1 : (j & 3) == 2 ? acc2 : acc3;",
    )
    for item in required:
        if item not in text:
            raise RuntimeError(f"expected native schedule/source guard is absent: {item}")
    if "seq_s[0] == 512 && seq_s[1] == 512" not in text:
        raise RuntimeError("homogeneous B2/K512 branch selector changed")

    lib = ctypes.CDLL(str(LIBRARY))
    decode = lib.decode360
    decode.argtypes = validator.ctypes_abi_types(ctypes)
    decode.restype = ctypes.c_int
    lib.decode360_workspace_bytes.argtypes = []
    lib.decode360_workspace_bytes.restype = ctypes.c_size_t
    lib.decode360_debug_bytes.argtypes = []
    lib.decode360_debug_bytes.restype = ctypes.c_size_t
    workspace = int(lib.decode360_workspace_bytes())
    debug_bytes = int(lib.decode360_debug_bytes())
    if len(decode.argtypes) != 28 or workspace != 536320:
        raise RuntimeError("R25 DSO ABI/workspace differs from frozen R23 contract")

    result = {
        "status": "source_preflight_passed",
        "mode": "CPU source and DSO preflight; no CUDA calls",
        "abi_parameter_count": len(validator.ABI_PARAMETERS),
        "abi_status": "canonical 28-argument declarations validated",
        "r23_source_sha256": sha256(BASE_SOURCE),
        "r25_source_sha256": sha256(SOURCE),
        "r23_include_sha256": sha256(BASE_INCLUDE),
        "r25_include_sha256": sha256(INCLUDE),
        "r25_library_sha256": sha256(LIBRARY),
        "r25_workspace_bytes": workspace,
        "r25_debug_bytes": debug_bytes,
        "dso_status": "decode360 loaded with canonical 28 ctypes argument declarations; workspace/debug query exports valid",
        "include_edit_hunks": changed,
        "dispatch": "B2 PV when max(seq_s)>2048, except homogeneous K512; B1 and other branches retain R23 paths",
        "candidate_schedule": {
            "ygroups": 8, "chunk": 64, "accumulators": 1,
            "accumulation": "FP32 __fmaf_rn in increasing j order within each y segment",
            "y_fold": "halves tree: (p0+p4)+(p2+p6), (p1+p5)+(p3+p7), then left+right",
        },
        "task_mapping": "unchanged kPvTasks/task iteration and CTA assignment; one task per q-head and 32-wide dim tile",
        "case_validation_contract": "full 46-case validator required; includes B2 context 2048 (K=2049) and long-context controls",
    }
    output = HERE / "results-r1" / "source-preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
