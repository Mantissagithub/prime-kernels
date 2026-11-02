"""Compare fixed-work serving runs without pooling repeats as processes."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def compare(paths):
    runs = [(path, json.loads(path.read_text())) for path in paths]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("duplicate input file")
    reference = runs[0][1]
    arms = {arm: [] for arm in ("baseline", "candidate")}
    reference_shapes = {(s["prompt_length"], s["batch"]): s for s in reference["shapes"]}
    if set(reference_shapes) != {(length, batch) for length in (32, 128, 1024) for batch in (1, 2)}:
        raise ValueError("missing expected shape")
    max_logprob_delta = 0.0
    for path, run in runs:
        if run["arm"] not in arms:
            raise ValueError(f"unknown arm: {path}")
        for key in ("model", "harness_sha256", "engine", "sampling", "runtime", "warmups", "repeats"):
            if run[key] != reference[key]:
                raise ValueError(f"incompatible {key}: {path}")
        shapes = {(s["prompt_length"], s["batch"]): s for s in run["shapes"]}
        if set(shapes) != set(reference_shapes) or len(run["shapes"]) != len(shapes):
            raise ValueError(f"missing or duplicate shapes: {path}")
        for key, shape in shapes.items():
            control = reference_shapes[key]
            if len(shape["parity_outputs"]) != key[1]:
                raise ValueError(f"request count mismatch: {path}, {key}")
            expected_hashes = []
            for actual, expected in zip(shape["parity_outputs"], control["parity_outputs"], strict=True):
                for field in ("prompt_token_ids", "token_ids", "prompt_tokens", "output_tokens"):
                    if actual[field] != expected[field]:
                        raise ValueError(f"unequal {field}: {path}, {key}")
                if len(actual["prompt_token_ids"]) != key[0] or len(actual["token_ids"]) != 64:
                    raise ValueError(f"incorrect token work: {path}, {key}")
                if actual["token_sha256"] != token_hash(actual["token_ids"]):
                    raise ValueError(f"token hash mismatch: {path}, {key}")
                expected_hashes.append(actual["token_sha256"])
                if len(actual["sampled_logprob"]) != 64:
                    raise ValueError(f"missing logprobs: {path}, {key}")
                for a, b in zip(actual["sampled_logprob"], expected["sampled_logprob"], strict=True):
                    if not math.isfinite(a) or not math.isfinite(b):
                        raise ValueError("nonfinite logprob")
                    max_logprob_delta = max(max_logprob_delta, abs(a - b))
            if shape["timed_token_hashes"] != [expected_hashes] * run["repeats"]:
                raise ValueError(f"timed outputs changed: {path}, {key}")
            samples = shape["batch_wall_ms_samples"]
            if len(samples) != run["repeats"] or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError(f"invalid timing samples: {path}, {key}")
            if statistics.median(samples) != shape["batch_wall_ms"]["median"]:
                raise ValueError(f"incorrect median: {path}, {key}")
        arms[run["arm"]].append((str(path), shapes))
    if not all(arms.values()):
        raise ValueError("both baseline and candidate processes are required")
    enough = all(len(values) >= 5 for values in arms.values())
    result = {"input_files": [str(path) for path in paths], "exact_tokens_and_equal_work": True,
              "max_abs_sampled_logprob_delta": max_logprob_delta,
              "process_counts": {arm: len(values) for arm, values in arms.items()},
              "scope": "64-token synchronous vLLM generation; excludes startup and RL training",
              "guards": "Guard records must be checked separately; this comparator validates result files.",
              "shapes": []}
    for key in sorted(reference_shapes):
        row = {"prompt_length": key[0], "batch": key[1]}
        values = {}
        for arm, processes in arms.items():
            medians = [shapes[key]["batch_wall_ms"]["median"] for _, shapes in processes]
            values[arm] = medians
            row[arm] = {"process_medians_ms": medians, "median_ms": statistics.median(medians),
                        "min_ms": min(medians), "max_ms": max(medians)}
        row["median_latency_change_percent"] = 100 * (row["candidate"]["median_ms"] / row["baseline"]["median_ms"] - 1)
        row["verdict"] = (
            "pilot_only" if not enough else
            "candidate_lower_all_process_medians" if max(values["candidate"]) < min(values["baseline"]) else
            "candidate_higher_all_process_medians" if min(values["candidate"]) > max(values["baseline"]) else
            "inconclusive_overlapping_process_ranges"
        )
        result["shapes"].append(row)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.files)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for row in result["shapes"]:
        print(f"ctx={row['prompt_length']} B={row['batch']}: {row['baseline']['median_ms']:.3f} -> "
              f"{row['candidate']['median_ms']:.3f} ms ({row['median_latency_change_percent']:+.2f}%); {row['verdict']}")
