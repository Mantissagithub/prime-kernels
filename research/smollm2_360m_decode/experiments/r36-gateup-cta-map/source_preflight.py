from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "r30-phase-prefetch/candidate"
CANDIDATE = ROOT / "candidate"
OLD_LOOP = '''  for (int r = gw; r < kI; r += kGlobalWarps) {
    const bf16* rows[2] = {L.gate_up + static_cast<size_t>(r) * kH,'''
NEW_LOOP = '''  const int gate_gw = B == 2 ? blockIdx.x * kWarps + warp : gw;
  for (int r = gate_gw; r < kI; r += kGlobalWarps) {
    const bf16* rows[2] = {L.gate_up + static_cast<size_t>(r) * kH,'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base_cu = (BASE / "block360.cu").read_text()
    candidate_cu = (CANDIDATE / "block360.cu").read_text()
    base_inc = (BASE / "block360_kernel.inc").read_text()
    candidate_inc = (CANDIDATE / "block360_kernel.inc").read_text()
    expected_inc = base_inc.replace(OLD_LOOP, NEW_LOOP, 1)
    assert base_inc.count(OLD_LOOP) == 1
    assert candidate_cu == base_cu, "R36 .cu must be identical to the accepted R30 baseline"
    assert candidate_inc == expected_inc, "R36 .inc differs outside the single gate/up start-row expression"
    result = {
        "status": "source-only-preflight-passed",
        "compile_or_gpu_run": False,
        "baseline": "accepted R30 r30-phase-prefetch/candidate",
        "hypothesis": "B2 CTA-contiguous gate/up rows may improve DRAM locality or warp arrival distribution; tail balance may worsen",
        "change": "gate_gw = blockIdx.x * kWarps + warp for B2; gw for B1",
        "cu_identical_to_r30": True,
        "inc_only_gate_up_initial_row_changed": True,
        "all_other_projection_loops_phases_prefetch_barriers_math_fma_order_outputs_and_abi_unchanged": True,
        "sha256": {
            "r30_cu": sha256(BASE / "block360.cu"),
            "r36_cu": sha256(CANDIDATE / "block360.cu"),
            "r30_inc": sha256(BASE / "block360_kernel.inc"),
            "r36_inc": sha256(CANDIDATE / "block360_kernel.inc"),
        },
        "inc_diff": list(difflib.unified_diff(base_inc.splitlines(), candidate_inc.splitlines(), fromfile="R30", tofile="R36", lineterm="")),
    }
    (ROOT / "source-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
