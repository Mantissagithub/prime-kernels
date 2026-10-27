#!/usr/bin/env python3
"""CPU-only coverage check for the R19 B2 warp-to-CTA start-index mapping."""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "block360.cu").read_text()
KERNEL = (HERE / "block360_kernel.inc").read_text()


def integer_constant(name: str) -> int:
    match = re.search(rf"constexpr int {name} = (\d+);", SOURCE)
    if not match:
        raise RuntimeError(f"could not find integer constant {name}")
    return int(match.group(1))


ctas = integer_constant("kCTAs")
threads = integer_constant("kThreads")
warps = threads // 32
global_warps = ctas * warps
hidden = integer_constant("kH")
intermediate = integer_constant("kI")

if "const int gw = B == 2 ? warp * kCTAs + blockIdx.x : blockIdx.x * kWarps + warp;" not in KERNEL:
    raise RuntimeError("R19 B2 mapping expression is missing")
if "const int gw = blockIdx.x * kWarps + warp;" in KERNEL:
    raise RuntimeError("original unconditional mapping remains")

# Derive the tile counts from the actual loop bounds used by QKV, O/down, and gate/up.
if "tile < 1600 / 16" not in KERNEL or "tile < kH / 16" not in KERNEL or "tile < kI / 16" not in KERNEL:
    raise RuntimeError("expected projection tile loop bounds changed")
tasks = {"qkv": 1600 // 16, "o_down": hidden // 16, "gate_up": intermediate // 16}

old_ids = [cta * warps + warp for cta in range(ctas) for warp in range(warps)]
new_ids = [warp * ctas + cta for cta in range(ctas) for warp in range(warps)]
if sorted(old_ids) != list(range(global_warps)) or sorted(new_ids) != list(range(global_warps)):
    raise RuntimeError("old or R19 mapping is not a bijection over global warp IDs")

results = {}
for name, task_count in tasks.items():
    for mapping_name, warp_ids in (("R16", old_ids), ("R19", new_ids)):
        counts = []
        observed = []
        for cta in range(ctas):
            ids = warp_ids[cta * warps : (cta + 1) * warps]
            assigned = [task for gw in ids for task in range(gw, task_count, global_warps)]
            counts.append(len(assigned))
            observed.extend(assigned)
        if sorted(observed) != list(range(task_count)):
            raise RuntimeError(f"{mapping_name} {name}: task coverage is not exactly once")
        results.setdefault(name, {})[mapping_name] = {
            "task_count": task_count,
            "active_ctas": sum(count > 0 for count in counts),
            "min_tasks_per_cta": min(counts),
            "max_tasks_per_cta": max(counts),
            "tasks_per_cta": counts,
        }

for name, mapping_results in results.items():
    r16 = mapping_results["R16"]
    r19 = mapping_results["R19"]
    if r19["active_ctas"] != ctas:
        raise RuntimeError(f"R19 {name}: not every CTA receives work")
    if r19["max_tasks_per_cta"] - r19["min_tasks_per_cta"] > 1:
        raise RuntimeError(f"R19 {name}: CTA work distribution differs by more than one")
    r19["task_coverage"] = "every task exactly once"
    r16["task_coverage"] = "every task exactly once"
    r19["mapping_delta"] = f"R16 active CTAs {r16['active_ctas']} -> R19 active CTAs {r19['active_ctas']}"

print(json.dumps({
    "mode": "CPU-only source-derived mapping coverage",
    "ctas": ctas,
    "warps_per_cta": warps,
    "global_warps": global_warps,
    "B1_mapping": "unchanged: blockIdx.x * kWarps + warp",
    "B2_mapping": "R19: warp * kCTAs + blockIdx.x",
    "task_results": results,
}, indent=2, sort_keys=True))
