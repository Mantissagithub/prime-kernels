from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R30 = ROOT.parent / "r30-phase-prefetch" / "candidate"
CANDIDATE = ROOT / "candidate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    base_cu = (R30 / "block360.cu").read_text()
    candidate_cu = (CANDIDATE / "block360.cu").read_text()
    base = (R30 / "block360_kernel.inc").read_text()
    candidate = (CANDIDATE / "block360_kernel.inc").read_text()

    require("using VecT = std::conditional_t<B == 2, bf16, float>;" in candidate,
            "B1/B2 shared vector storage selection is missing")
    require("__shared__ __align__(16) VecT vec_s[kMaxB * kI];" in candidate,
            "shared vector storage is not aligned for packed loads")
    require(candidate.count("warp_gemv_rows<kH, B, 1>(rows, vec_s, lane, res);") == 2 and
            candidate.count("warp_gemv_rows<kH, B, 2>(rows, vec_s, lane, res);") == 1 and
            candidate.count("warp_gemv_rows<kI, B, 1>(rows, vec_s, lane, res);") == 1,
            "not every R30 projection consumes the shared vector")
    require("const uint4 packed = *reinterpret_cast<const uint4*>(vs + b * K + c * 8);" in candidate_cu,
            "B2 projection reads are not packed 16-byte BF16 loads")
    require(candidate_cu.count("acc[i][b][e & 3] = fmaf(w, xv[e], acc[i][b][e & 3]);") == 2,
            "FMA accumulation order differs between packed and original paths")
    require("if constexpr (B == 2) {\n    for (int idx = t; idx < B * kI; idx += kThreads) vec_s[idx] = ws[kOffAct + idx];" in candidate,
            "B2 down projection adds a conversion pass instead of copying packed values")
    require("#include <type_traits>" in candidate_cu, "compile-time representation selection is unavailable")
    build = (CANDIDATE / "build.sh").read_text()
    require("-O3 -arch=sm_89 -lineinfo" in build and "--use_fast_math" not in build and
            "-fmad=false" not in build,
            "build flags differ from the R30 numeric mode")
    require(("template <int B>\nstatic int query_props" in base) and
            base[base.index("template <int B>\nstatic int query_props"):] ==
            candidate[candidate.index("template <int B>\nstatic int query_props"):],
            "host ABI and launch validation changed")
    declaration_start = "namespace cg = cooperative_groups;"
    declarations_end = "__device__ __forceinline__ float bf2f"
    base_declarations = base_cu[base_cu.index(declaration_start):base_cu.index(declarations_end)]
    candidate_declarations = candidate_cu[candidate_cu.index(declaration_start):candidate_cu.index(declarations_end)]
    require(base_declarations == candidate_declarations,
            "kernel constants, Params, or device declarations changed")
    unchanged_tail = "// Prefetch a row into L2, one 128-byte line per lane step."
    base_tail = base_cu[base_cu.index(unchanged_tail):base_cu.index('#include "block360_kernel.inc"')]
    candidate_tail = candidate_cu[candidate_cu.index(unchanged_tail):candidate_cu.index('#include "block360_kernel.inc"')]
    require(base_tail == candidate_tail, "R30 prefetch or RoPE helpers changed")

    result = {
        "status": "source-only-preflight-passed",
        "compile_or_gpu_run": False,
        "baseline": "private R30 phase-prefetch candidate",
        "candidate": "B2 shared vectors use BF16; B1 shared vectors remain FP32",
        "data_shape": {
            "norm_and_attention_per_request": [960],
            "swiglu_activation_per_request": [2560],
            "projection_input_load": "eight BF16 values as one aligned 16-byte vector",
        },
        "exactness_argument": [
            "RMS norm, merged attention, and SwiGLU producers already round to BF16 before projection.",
            "The B2 path stores those same BF16 values directly and expands each loaded BF16 to its exact FP32 value before the original FMA sequence.",
            "The down input is already BF16 in workspace and now copies directly into BF16 shared storage.",
            "The projection FMA lane assignment, row assignment, accumulator fold, and output BF16 stores are unchanged.",
            "The B1 compile-time specialization retains the original FP32 shared loads, conversions, and stores.",
        ],
        "workspace_and_abi": "unchanged",
        "fast_math": False,
        "extra_conversion_pass": False,
        "source_sha256": {
            "r30": {name: sha256(R30 / name) for name in ("block360.cu", "block360_kernel.inc")},
            "r34": {name: sha256(CANDIDATE / name) for name in ("block360.cu", "block360_kernel.inc")},
        },
        "kernel_diff": list(difflib.unified_diff(base.splitlines(), candidate.splitlines(), lineterm="")),
        "cu_diff": list(difflib.unified_diff(base_cu.splitlines(), candidate_cu.splitlines(), lineterm="")),
    }
    (ROOT / "source-preflight-r1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"kernel_diff", "cu_diff"}}, indent=2))


if __name__ == "__main__":
    main()
