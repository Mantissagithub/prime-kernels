#!/usr/bin/env python3
"""Prove the staged candidate preserves rolling math and covers CSR metadata on CPU."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gateup-lossless-rolling-prefetch-r1" / "projection_probe.cu"
CANDIDATE = HERE / "projection_probe.cu"
HARNESS = HERE / "harness.py"
BASE_SHA256 = "ab7edfe04e5c0a756dfa6d6752a6ea7cb9893fb4ffac8ea05c218677b75412b4"
N, K, GLOBAL_WARPS, MATRICES, MAX_ROWS = 2560, 960, 192, 2, 14


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def tokens(source):
    return "".join(source.split())


def projection_body(kernel):
    return tokens(kernel[kernel.index("float acc[2][2][4] = {};"):])


def staged_slots():
    loaded = {}
    for gw in range(GLOBAL_WARPS):
        for slot in range(MAX_ROWS):
            row = gw + slot * GLOBAL_WARPS
            if row < N:
                loaded[gw, slot] = row
    return loaded


def scheduled_rows():
    scheduled = {}
    for gw in range(GLOBAL_WARPS):
        if gw < N:
            scheduled[gw, 0] = gw
        for row in range(gw, N, GLOBAL_WARPS):
            future = row + GLOBAL_WARPS
            if future < N:
                scheduled[gw, (row - gw) // GLOBAL_WARPS + 1] = future
    return scheduled


def check_abstract_offsets():
    fixtures = (
        [0] * (MATRICES * N + 1),
        [i * 7 for i in range(MATRICES * N + 1)],
        [sum((i % 5) != 0 for i in range(row)) for row in range(MATRICES * N + 1)],
    )
    loaded = staged_slots()
    scheduled = scheduled_rows()
    for offsets in fixtures:
        staged_ranges = {}
        for (gw, slot), row in loaded.items():
            for matrix in range(MATRICES):
                source_row = matrix * N + row
                first = offsets[source_row]
                count = offsets[source_row + 1] - first
                staged_ranges[gw, slot, matrix] = (first, count)
        for (gw, slot), row in scheduled.items():
            for matrix in range(MATRICES):
                source_row = matrix * N + row
                rolling_range = (offsets[source_row], offsets[source_row + 1] - offsets[source_row])
                staged_range = staged_ranges[gw, slot, matrix]
                assert staged_range == rolling_range
                assert (staged_range[1] > 0) == (rolling_range[1] > 0)


def main():
    base_bytes = BASE.read_bytes()
    candidate_bytes = CANDIDATE.read_bytes()
    base = base_bytes.decode()
    candidate = candidate_bytes.decode()
    assert sha(BASE) == BASE_SHA256
    assert candidate_bytes.startswith(base_bytes)

    rolling = extract(base, "__global__ void gateup_lossless_rolling_kernel(")
    staged = extract(candidate, "__global__ void gateup_lossless_staged_csr_kernel(")
    assert projection_body(rolling) == projection_body(staged)
    assert "__shared__ __align__(8) uint2 csr_ranges[kWarps][kMaxRowsPerWarp][2];" in staged
    assert "if (lane < kMaxRowsPerWarp)" in staged
    assert staged.index("csr_ranges[warp][lane][matrix] = make_uint2(first, last - first);") < staged.index("__syncthreads();")
    assert staged.count("__ldg(raw_offsets + row)") == 1
    assert staged.count("__ldg(raw_offsets + row + 1)") == 1
    assert "csr_ranges[warp][0][matrix]" in staged
    assert "csr_ranges[warp][future_slot][matrix]" in staged
    assert "future_slot = (r - gw) / kGlobalWarps + 1" in staged
    assert "prefetch_range_l2(plane + size_t(row) * kK, kK);" in staged
    assert "prefetch_range_l2(headers + size_t(row) * kChunks, kChunks * sizeof(uint32_t));" in staged

    loaded = staged_slots()
    scheduled = scheduled_rows()
    assert len(loaded) == N
    assert len(scheduled) == N
    assert sorted(loaded.values()) == list(range(N))
    assert sorted(scheduled.values()) == list(range(N))
    assert all((gw, slot) in loaded and loaded[gw, slot] == row
               for (gw, slot), row in scheduled.items())
    assert max(slot for _, slot in scheduled) < MAX_ROWS
    assert all(0 <= matrix * N + row < MATRICES * N and
               matrix * N + row + 1 <= MATRICES * N
               for row in range(N) for matrix in range(MATRICES))
    check_abstract_offsets()

    harness = HARNESS.read_text()
    ast.parse(harness)
    assert 'arm_names = ("rolling_depth_1", "staged_csr")' in harness
    assert '"timed_projection_calls_per_arm": LAYERS * (PLACEMENTS // len(arm_names)) * REPLAYS' in harness
    assert "PLACEMENTS, REPLAYS = 10, 5" in harness
    assert "MEMORY_LIMIT = 3 * 1024**3" in harness
    assert "decoded_weight_words_checked\": LAYERS * 2 * N * K" in harness

    print(json.dumps({
        "base_source_sha256": BASE_SHA256,
        "candidate_source_sha256": sha(CANDIDATE),
        "base_source_preserved_as_exact_prefix": True,
        "staged_projection_body_token_exact_with_rolling_depth_1": True,
        "rolling_shared_bytes_per_cta": 2 * K * 4,
        "staged_csr_added_shared_bytes_per_cta": 8 * MAX_ROWS * MATRICES * 8,
        "staged_csr_total_shared_bytes_per_cta": 2 * K * 4 + 8 * MAX_ROWS * MATRICES * 8,
        "warp_assignments": GLOBAL_WARPS,
        "rows_staged_exactly_once_per_matrix": N,
        "rows_hinted_exactly_once_per_matrix": N,
        "uninitialized_or_out_of_bounds_staged_reads": 0,
        "csr_row_offset_index_pairs_in_bounds": True,
        "synthetic_csr_interval_fixtures_checked": 3,
        "staged_raw_hint_ranges_match_rolling_for_all_fixtures": True,
        "harness_python_ast_valid": True,
        "timed_placements_per_arm": 5,
        "replays_per_placement": 5,
        "timed_projection_calls_per_arm": 32 * 5 * 5,
        "gpu_build_or_execution_performed": False,
        "checkpoint_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
