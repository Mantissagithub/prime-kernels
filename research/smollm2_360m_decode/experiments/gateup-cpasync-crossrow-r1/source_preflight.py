#!/usr/bin/env python3
"""Check baseline preservation and cross-row task/buffer invariants on CPU."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "gateup-cpasync-probe-r1" / "projection_probe.cu"
SOURCE = HERE / "projection_probe.cu"
N, GLOBAL_WARPS, ITERS, CHUNKS = 2560, 192, 4, 120
BASELINE_SOURCE_SHA256 = "05f4c790b240bb8322a559ca3e34a8c84f88d76b40a99a56b80061c4a4aeed27"


def function_slice(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for end in range(opening, len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if depth == 0:
                return source[start:end + 1]
    raise AssertionError(f"unterminated function: {marker}")


baseline_bytes = BASELINE.read_bytes()
source_bytes = SOURCE.read_bytes()
baseline_source = baseline_bytes.decode()
source = source_bytes.decode()
assert hashlib.sha256(baseline_bytes).hexdigest() == BASELINE_SOURCE_SHA256
assert baseline_source[:baseline_source.index("template <bool AsyncWeights>")] == source[:source.index("template <bool AsyncWeights>")]
marker = "template <bool AsyncWeights>\n__global__ void gateup_kernel"
assert function_slice(baseline_source, marker) == function_slice(source, marker)

coverage = [0] * N
task_count = 0
max_tasks_per_warp = 0
for gw in range(GLOBAL_WARPS):
    rows = [gw + index * GLOBAL_WARPS for index in range((N - gw + GLOBAL_WARPS - 1) // GLOBAL_WARPS)]
    tasks = [(gw + (task >> 2) * GLOBAL_WARPS, task & 3) for task in range(len(rows) * ITERS)]
    assert tasks == [(row, iteration) for row in rows for iteration in range(ITERS)]
    pending = []
    read_tasks = set()
    for task, (row, iteration) in enumerate(tasks):
        buffer = iteration & 1
        if task == 0:
            pending.append((task, buffer))
        assert any(pending_task == task and slot == buffer for pending_task, slot in pending)
        if task + 1 < len(tasks):
            next_iteration = (task + 1) & 3
            next_buffer = next_iteration & 1
            assert all(slot != next_buffer for _, slot in pending)
            pending.append((task + 1, next_buffer))
            assert pending[0][0] == task
            pending.pop(0)
        else:
            assert pending == [(task, buffer)]
            pending.clear()
        assert task not in read_tasks
        read_tasks.add(task)
        if iteration == ITERS - 1:
            coverage[row] += 1
        task_count += 1
    assert not pending and len(read_tasks) == len(tasks)
    max_tasks_per_warp = max(max_tasks_per_warp, len(tasks))

for iteration in range(ITERS):
    chunks = [lane + 32 * iteration for lane in range(32)]
    valid = [chunk for chunk in chunks if chunk < CHUNKS]
    assert valid == list(range(iteration * 32, min((iteration + 1) * 32, CHUNKS)))
assert coverage == [1] * N

print(json.dumps({
    "baseline_source_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
    "candidate_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "baseline_helper_prefix_identical": True,
    "baseline_kernel_identical": True,
    "warp_starts_checked": GLOBAL_WARPS,
    "output_rows_checked": N,
    "output_rows_covered_exactly_once": sum(count == 1 for count in coverage),
    "row_tile_tasks_checked": task_count,
    "max_tasks_per_warp": max_tasks_per_warp,
    "partial_final_tile_valid_chunks": 24,
    "partial_final_tile_zero_fill_lanes": 8,
    "duplicate_tasks": sum(count > 1 for count in coverage),
    "missing_tasks": sum(count == 0 for count in coverage),
    "pending_buffer_reuse_errors": 0,
}, indent=2))
