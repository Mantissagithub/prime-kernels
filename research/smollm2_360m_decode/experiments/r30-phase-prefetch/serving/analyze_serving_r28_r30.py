#!/usr/bin/env python3
"""Summarize the private R28/R30 serving ABBA runs for one tag."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ARM_FILES = {
    "A1": "A1-r28-{tag}.json",
    "B1": "B1-r30-{tag}.json",
    "B2": "B2-r30-{tag}.json",
    "A2": "A2-r28-{tag}.json",
}
R28 = ("A1", "A2")
R30 = ("B1", "B2")
LIMIT_MIB = 3072


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def shape_key(shape: dict[str, Any]) -> str:
    return f"{shape['prompt_len']}/{shape['batch']}"


def requests(shape: dict[str, Any]) -> list[dict[str, Any]]:
    return [request for run in shape["runs"] for request in run["requests"]]


def itl_from_timestamps(request: dict[str, Any]) -> float:
    return (float(request["last_token_ts"]) - float(request["first_token_ts"])) / 63 * 1000


def pct(candidate: float, reference: float) -> float:
    return (candidate / reference - 1) * 100


def pair_request(x: dict[str, Any], y: dict[str, Any]) -> dict[str, Any]:
    tx, ty = x["token_ids"], y["token_ids"]
    if len(tx) != len(ty):
        raise ValueError(f"generated lengths differ: {len(tx)} vs {len(ty)}")
    first = next((i for i, (a, b) in enumerate(zip(tx, ty)) if a != b), None)
    gen_limit = len(tx) if first is None else first
    top_limit = len(tx) if first is None else first + 1
    dlp = [abs(x["sampled_logprob"][i] - y["sampled_logprob"][i]) for i in range(gen_limit)]
    top_diffs: list[float] = []
    candidate_diff_steps = 0
    exact_top_steps = 0
    for step in range(top_limit):
        xs = {int(t): (float(lp), int(rank)) for t, lp, rank in x["top5"][step]}
        ys = {int(t): (float(lp), int(rank)) for t, lp, rank in y["top5"][step]}
        if xs.keys() != ys.keys():
            candidate_diff_steps += 1
        if xs == ys:
            exact_top_steps += 1
        for token in xs.keys() & ys.keys():
            top_diffs.append(abs(xs[token][0] - ys[token][0]))
    return {
        "identical_sequence": first is None,
        "first_mismatch_step": first,
        "tokens_equal": sum(a == b for a, b in zip(tx, ty)),
        "tokens_total": len(tx),
        "sampled_logprob_max_abs_before_mismatch": max(dlp, default=0.0),
        "sampled_logprob_exact_before_mismatch": all(
            x["sampled_logprob"][i] == y["sampled_logprob"][i] for i in range(gen_limit)
        ),
        "top5_logprob_max_abs_through_mismatch": max(top_diffs, default=0.0),
        "top5_exact_steps_through_mismatch": exact_top_steps,
        "top5_steps_compared": top_limit,
        "top5_candidate_set_diff_steps": candidate_diff_steps,
    }


def compare_arms(x: dict[str, Any], y: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_y = {shape_key(shape): shape for shape in y["shapes"]}
    for shape_x in x["shapes"]:
        key = shape_key(shape_x)
        shape_y = by_y[key]
        rows = []
        for run_x, run_y in zip(shape_x["runs"], shape_y["runs"], strict=True):
            for req_x, req_y in zip(run_x["requests"], run_y["requests"], strict=True):
                rows.append(pair_request(req_x, req_y))
        total_tokens = sum(row["tokens_total"] for row in rows)
        out[key] = {
            "sequences": len(rows),
            "identical_sequences": sum(row["identical_sequence"] for row in rows),
            "greedy_token_agreement": sum(row["tokens_equal"] for row in rows) / total_tokens,
            "first_mismatch_steps": [row["first_mismatch_step"] for row in rows
                                     if row["first_mismatch_step"] is not None],
            "sampled_logprob_max_abs_before_mismatch": max(
                row["sampled_logprob_max_abs_before_mismatch"] for row in rows
            ),
            "sampled_logprob_exact_sequences": sum(
                row["sampled_logprob_exact_before_mismatch"] for row in rows
            ),
            "top5_logprob_max_abs_through_mismatch": max(
                row["top5_logprob_max_abs_through_mismatch"] for row in rows
            ),
            "top5_exact_steps_through_mismatch": sum(
                row["top5_exact_steps_through_mismatch"] for row in rows
            ),
            "top5_steps_compared": sum(row["top5_steps_compared"] for row in rows),
            "top5_candidate_set_diff_steps": sum(row["top5_candidate_set_diff_steps"] for row in rows),
        }
    return out


def clock_stats(payload: dict[str, Any]) -> dict[str, Any]:
    path = HERE / payload["clock_log"]
    start, end = payload["measure_window"]["start"], payload["measure_window"]["end"]
    clocks: list[int] = []
    temps: list[float] = []
    throttles: set[str] = set()
    with path.open() as handle:
        for row in csv.reader(handle):
            if len(row) < 5 or not (start <= row[0].strip() <= end):
                continue
            sm, throttle, temp = row[1].strip(), row[2].strip(), row[3].strip()
            if sm.isdigit():
                clocks.append(int(sm))
            if throttle.startswith("0x") and throttle != "0x0000000000000000":
                throttles.add(throttle)
            try:
                temps.append(float(temp))
            except ValueError:
                pass
    return {
        "sm_clock_mhz_median": statistics.median(clocks) if clocks else None,
        "sm_clock_mhz_min": min(clocks) if clocks else None,
        "sm_clock_mhz_max": max(clocks) if clocks else None,
        "temp_c_median": statistics.median(temps) if temps else None,
        "throttle_reasons": sorted(throttles),
    }


def analyze(tag: str) -> dict[str, Any]:
    arms = {name: load_json(HERE / filename.format(tag=tag)) for name, filename in ARM_FILES.items()}
    for name, payload in arms.items():
        if payload["arm"] != ("r28" if name in R28 else "r30"):
            raise ValueError(f"arm label mismatch for {name}: {payload['arm']}")
        if (payload["prompt_lengths"], payload["batches"], payload["new_tokens"],
                payload["warmups"], payload["repeats"]) != ([32, 128, 1024], [1, 2], 64, 3, 7):
            raise ValueError(f"serving shape/config mismatch for {name}")
        for shape in payload["shapes"]:
            expected_batch = shape["batch"]
            if len(shape["runs"]) != 7:
                raise ValueError(f"expected 7 measured runs for {name}/{shape_key(shape)}")
            for run in shape["runs"]:
                if len(run["requests"]) != expected_batch:
                    raise ValueError(f"expected batch={expected_batch} requests for {name}/{shape_key(shape)}")
            for request in requests(shape):
                if len(request["token_ids"]) != 64:
                    raise ValueError(f"expected 64 generated tokens for {name}/{shape_key(shape)}")
                if len(request["top5"]) != 64 or len(request["sampled_logprob"]) != 64:
                    raise ValueError(f"expected 64 top5 and sampled-logprob rows for {name}/{shape_key(shape)}")
                derived = itl_from_timestamps(request)
                if not math.isclose(derived, request["itl_ms"], rel_tol=0, abs_tol=1e-8):
                    raise ValueError(f"stored ITL does not match token timestamps for {name}/{shape_key(shape)}")

    by_arm = {name: {shape_key(shape): shape for shape in payload["shapes"]}
              for name, payload in arms.items()}
    shapes = [shape_key(shape) for shape in arms["A1"]["shapes"]]
    latency_rows = []
    for key in shapes:
        all_itl = {
            arm: [itl_from_timestamps(request) for request in requests(by_arm[arm][key])]
            for arm in ARM_FILES
        }
        r28_pool = [value for arm in R28 for value in all_itl[arm]]
        r30_pool = [value for arm in R30 for value in all_itl[arm]]
        r28_a1 = statistics.median(all_itl["A1"])
        r28_a2 = statistics.median(all_itl["A2"])
        r30_b1 = statistics.median(all_itl["B1"])
        r30_b2 = statistics.median(all_itl["B2"])
        latency_rows.append({
            "shape": key,
            "per_arm_request_itl_median_ms": {arm: statistics.median(values) for arm, values in all_itl.items()},
            "r28_pooled_request_itl_median_ms": statistics.median(r28_pool),
            "r30_pooled_request_itl_median_ms": statistics.median(r30_pool),
            "r30_vs_r28_pooled_pct": pct(statistics.median(r30_pool), statistics.median(r28_pool)),
            "r30_both_beat_both_r28": max(r30_b1, r30_b2) < min(r28_a1, r28_a2),
            "r28_control_spread_pct": pct(r28_a2, r28_a1),
            "request_counts": {arm: len(values) for arm, values in all_itl.items()},
        })

    pairs = {
        "R28_noise_A1_vs_A2": compare_arms(arms["A1"], arms["A2"]),
        "R30_self_B1_vs_B2": compare_arms(arms["B1"], arms["B2"]),
        "B1_R30_vs_A1_R28": compare_arms(arms["B1"], arms["A1"]),
        "B1_R30_vs_A2_R28": compare_arms(arms["B1"], arms["A2"]),
        "B2_R30_vs_A1_R28": compare_arms(arms["B2"], arms["A1"]),
        "B2_R30_vs_A2_R28": compare_arms(arms["B2"], arms["A2"]),
    }
    guards = {}
    for name in ARM_FILES:
        arm_label = ARM_FILES[name].split("{tag}")[0].rstrip("-")
        for kind, prefix in (("outer", "guard-"), ("3gb", "guard-3gb-")):
            path = HERE / f"{prefix}{arm_label}-{tag}" / "result.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing {kind} guard result for {name}: {path}")
            guard = load_json(path)
            if guard.get("returncode") != 0 or guard.get("stop_reason") is not None:
                raise ValueError(f"{kind} guard failed for {name}: {guard}")
            if kind == "3gb" and guard.get("peak_sampled_gpu_memory_mib", LIMIT_MIB + 1) > LIMIT_MIB:
                raise ValueError(f"3 GiB GPU guard limit exceeded for {name}: {guard}")
            guards[f"{kind}_{name}"] = guard
    results = {
        "tag": tag,
        "arms_order": list(ARM_FILES),
        "measurement": "pooled per-request ITL reconstructed from RequestOutput first/last token timestamps",
        "shapes": latency_rows,
        "both_r30_arms_beat_both_r28_per_shape": all(row["r30_both_beat_both_r28"] for row in latency_rows),
        "parity": pairs,
        "route_and_hashes": {name: {"worker_before": arms[name]["worker_before"],
                                     "worker_after": arms[name]["worker_after"],
                                     "candidate": arms[name]["candidate"]} for name in arms},
        "clocks": {name: clock_stats(payload) for name, payload in arms.items()},
        "guards": guards,
    }
    output = HERE / f"results-r28-r30-{tag}.json"
    readme = HERE / f"README-r28-r30-{tag}.md"
    if output.exists() or readme.exists():
        raise FileExistsError(f"refusing to overwrite analyzed tag {tag}")
    output.write_text(json.dumps(results, indent=2) + "\n")
    write_readme(readme, results)
    return results


def write_readme(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# R28/R30 private serving comparison",
        "",
        f"Tag: `{results['tag']}`. ITL is reconstructed from each request's first/last token timestamps and pooled across the requests from both arms of each candidate.",
        "",
        "## Inter-token latency",
        "",
        "| Prompt/batch | A1 R28 | B1 R30 | B2 R30 | A2 R28 | R28 pooled | R30 pooled | R30 vs R28 | Both R30 < both R28 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in results["shapes"]:
        v = row["per_arm_request_itl_median_ms"]
        lines.append(
            f"| {row['shape']} | {v['A1']:.3f} | {v['B1']:.3f} | {v['B2']:.3f} | {v['A2']:.3f} | "
            f"{row['r28_pooled_request_itl_median_ms']:.3f} | {row['r30_pooled_request_itl_median_ms']:.3f} | "
            f"{row['r30_vs_r28_pooled_pct']:+.1f}% | {'yes' if row['r30_both_beat_both_r28'] else 'no'} |"
        )
    lines.extend([
        "",
        "The strict per-shape condition compares the slower of the two R30 arm medians with the faster of the two R28 arm medians. Request wall and TTFT are retained in the raw arm JSONs but are not used for a speed conclusion.",
        "",
        "## Sequence and logprob parity",
        "",
        "| Pair | Shape | Identical sequences | Token agreement | Sampled logprob max abs Δ | Exact sampled logprob sequences | Top-5 max abs Δ | Exact top-5 steps | Candidate-set changes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for pair, rows in results["parity"].items():
        for shape, item in rows.items():
            lines.append(
                f"| {pair} | {shape} | {item['identical_sequences']}/{item['sequences']} | "
                f"{item['greedy_token_agreement']:.4f} | {item['sampled_logprob_max_abs_before_mismatch']:.6g} | "
                f"{item['sampled_logprob_exact_sequences']}/{item['sequences']} | "
                f"{item['top5_logprob_max_abs_through_mismatch']:.6g} | "
                f"{item['top5_exact_steps_through_mismatch']}/{item['top5_steps_compared']} | "
                f"{item['top5_candidate_set_diff_steps']} |"
            )
    lines.extend([
        "",
        "## Integrity and resources",
        "",
        "Worker source/include/library hashes and route state are in the JSON artifact. The local serving wrapper stops an arm when sampled GPU use exceeds 3072 MiB; outer host, disk, cgroup, and timeout monitoring comes from `runtime/run_gpu.sh`.",
        "",
        "No timing conclusion should be drawn from wall/TTFT alone. Review pooled timestamp ITL, per-shape strict both-beat-both, R28 control spread, and sequence/logprob exactness together.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    args = parser.parse_args()
    result = analyze(args.tag)
    print(json.dumps({"tag": args.tag, "both_r30_arms_beat_both_r28_per_shape":
                      result["both_r30_arms_beat_both_r28_per_shape"]}, indent=2))


if __name__ == "__main__":
    main()
