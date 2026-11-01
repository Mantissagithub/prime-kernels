from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "r30-phase-prefetch/candidate"
CANDIDATE = ROOT / "candidate"
ANCHOR = '''  if (p.debug && blockIdx.x == 0) {
    for (int i = t; i < B * kH; i += kThreads)
      debug_layer[kDbgNorm2 + i] = __float2bfloat16_rn(vec_s[i]);
  }

  // R27 FAST gate/up:'''
EARLY = '''  if (p.debug && blockIdx.x == 0) {
    for (int i = t; i < B * kH; i += kThreads)
      debug_layer[kDbgNorm2 + i] = __float2bfloat16_rn(vec_s[i]);
  }

  if constexpr (B == 2) {
    for (int k = 0; k < (kH + kGlobalWarps - 1) / kGlobalWarps; ++k) {
      const int r = gw + k * kGlobalWarps;
      if (r < kH) prefetch_row(L.down + static_cast<size_t>(r) * kI, kI * sizeof(bf16));
    }
  }

  // R27 FAST gate/up:'''
AFTER = '''  for (int k = 0; k < (kH + kGlobalWarps - 1) / kGlobalWarps; ++k) {
    const int r = gw + k * kGlobalWarps;
    if (r < kH) prefetch_row(L.down + static_cast<size_t>(r) * kI, kI * sizeof(bf16));
  }
  sync_grid();'''
CONDITIONAL_AFTER = '''  if constexpr (B != 2) {
    for (int k = 0; k < (kH + kGlobalWarps - 1) / kGlobalWarps; ++k) {
      const int r = gw + k * kGlobalWarps;
      if (r < kH) prefetch_row(L.down + static_cast<size_t>(r) * kI, kI * sizeof(bf16));
    }
  }
  sync_grid();'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base_cu = (BASE / "block360.cu").read_text()
    candidate_cu = (CANDIDATE / "block360.cu").read_text()
    base_inc = (BASE / "block360_kernel.inc").read_text()
    candidate_inc = (CANDIDATE / "block360_kernel.inc").read_text()
    assert base_inc.count(ANCHOR) == 1
    assert base_inc.count(AFTER) == 1
    expected_inc = base_inc.replace(ANCHOR, EARLY, 1).replace(AFTER, CONDITIONAL_AFTER, 1)
    assert candidate_cu == base_cu, "R37 .cu must be identical to accepted R30 baseline"
    assert candidate_inc == expected_inc, "R37 include differs outside the conditional down-prefetch relocation"
    report = {
        "status": "source-only-preflight-passed",
        "compile_or_gpu_run": False,
        "baseline": "accepted R30 r30-phase-prefetch/candidate",
        "hypothesis": "Moving only B2 down-weight L2 hints before gate/up may overlap hint latency with gate/up work; contention may erase or reverse any gain",
        "cu_identical_to_r30": True,
        "inc_is_exact_expected_relocation": True,
        "b2_down_prefetch_before_gate_up": True,
        "b1_down_prefetch_position_unchanged": True,
        "gate_up_math_row_map_all_projection_math_barriers_workspace_abi_and_outputs_unchanged": True,
        "sha256": {
            "r30_cu": sha256(BASE / "block360.cu"),
            "r37_cu": sha256(CANDIDATE / "block360.cu"),
            "r30_inc": sha256(BASE / "block360_kernel.inc"),
            "r37_inc": sha256(CANDIDATE / "block360_kernel.inc"),
        },
        "inc_diff": list(difflib.unified_diff(base_inc.splitlines(), candidate_inc.splitlines(), fromfile="R30", tofile="R37", lineterm="")),
    }
    (ROOT / "source-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
