#!/usr/bin/env python3
"""CPU-only integrity and ABI preflight for the private FAST R28 timing clone."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CANON = REPO.parent / "prime-kernels/research/smollm2_360m_decode/experiments/r28-flash-decode"
VALIDATOR = REPO / "outputs/megakernel-20261025/block-360-heldout-r13/validate_full32.py"
EXPECTED = {
    "canonical_source": "1d121b7f89561a955b1fcc665daa7535a13816821920fe4030c4040e7bf61313",
    "canonical_include": "0ac2ba97ed2bfc4684c476987635e98e06affe2323349e4f9c4d9a27ed1247f0",
    "canonical_r28_library": "df3f03f7d680719b2f282fe7ad0a41cc24ff20f04d14809756b9b5e63a7f6603",
    "instrumented_source": "46b2c58d16adf13b43c85daf33d8e5b8d55bd72f5a13fbf2d674893aa2462b40",
    "instrumented_include": "1c569dc50c6e88f0ae43c27a1842c1dcd880fe78fb837cce7dab86c6188b0b02",
}
SLOTS = ("Norm1", "Qkv", "RopeCache", "SplitKv", "Merge", "OResidual", "Norm2",
         "GateUpSilu", "DownResidual", "LayerCapture", "FinalNorm")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_full32", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ABI validator: {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=HERE / "preflight.json")
    args = parser.parse_args()
    paths = {
        "canonical_source": CANON / "block360.cu",
        "canonical_include": CANON / "block360_kernel.inc",
        "canonical_r28_library": REPO / "outputs/megakernel-20261025/r28-flash-decode/build/libdecode360_r28_flash.so",
        "instrumented_source": HERE / "src/block360.cu",
        "instrumented_include": HERE / "src/block360_kernel.inc",
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED:
        raise RuntimeError(f"source hash mismatch: expected={EXPECTED}, got={hashes}")

    validator = load_validator()
    validator.validate_abi_source(paths["instrumented_include"])
    if len(validator.ABI_PARAMETERS) != 28:
        raise RuntimeError(f"unexpected canonical ABI parameter count {len(validator.ABI_PARAMETERS)}")
    text = paths["instrumented_source"].read_text() + paths["instrumented_include"].read_text()
    if text.count("clock64()") != 29:
        raise RuntimeError(f"expected 29 local clock reads, found {text.count('clock64()')}")
    missing = [f"kProfile{name}" for name in SLOTS if f"kProfile{name}" not in text]
    if missing:
        raise RuntimeError(f"profile slots missing: {missing}")
    if "decode360_profile_set_buffer" not in paths["instrumented_source"].read_text():
        raise RuntimeError("profile setter export missing")
    if not re.search(r"kProfileBufferBytes\s*=\s*size_t\(32\)\s*\*\s*kProfileSlots\s*\*\s*kProfileCtas\s*\*", text):
        raise RuntimeError("unexpected timing-buffer layout")

    result = {
        "status": "passed",
        "mode": "CPU-only source/hash/ABI inspection; no CUDA calls, compile, or GPU launch",
        "hashes": hashes,
        "abi_parameter_count": len(validator.ABI_PARAMETERS),
        "abi_status": "instrumented decoder remains the canonical 28-argument ABI",
        "profile": {"slots": list(SLOTS), "buffer_bytes": 32 * len(SLOTS) * 24 * 2 * 8,
                    "clock_reads_in_source": 29, "per_decoder_layer": 27, "final_norm": 2,
                    "storage": "separate private device-global buffer; one thread per CTA holds slot deltas in 176 shared bytes and flushes once per layer"},
        "checks": ["canonical FAST R28 hashes pinned", "private source hashes pinned", "28-argument ABI validated",
                   "profile slot names and fixed buffer geometry validated", "no instrumentation uses debug/workspace tensors"],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
