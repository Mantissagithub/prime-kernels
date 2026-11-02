import argparse
import json
import statistics
from pathlib import Path


def analyze(path):
    report = json.loads(path.read_text())
    output = []
    for case in report["cases"]:
        if not case["parity_outside_timing"]["A_vs_B"]["exact_bitwise_all_layers"]:
            raise ValueError("profile changed outputs")
        samples = {arm: [row["median_ms_per_step"] for row in case["blocks"] if row["arm"] == arm]
                   for arm in ("A-Package", "B-Lossless")}
        phase_snapshots = []
        for placement in case["blocks"]:
            if placement["arm"] != "B-Lossless":
                continue
            profile = placement["profile_cycles"]
            if len(profile) != 32 or any(len(layer) != 24 for layer in profile):
                raise ValueError("incomplete layer/CTA coverage")
            if any(len(cta) != 4 or any(value <= 0 for value in cta) for layer in profile for cta in layer):
                raise ValueError("invalid profile cell")
            phase_snapshots.append([
                statistics.median(sum(profile[layer][cta][phase] for layer in range(32)) for cta in range(24))
                for phase in range(4)])
        if len(phase_snapshots) != 6:
            raise ValueError("expected six instrumented placements")
        medians = {arm: statistics.median(values) for arm, values in samples.items()}
        phases = [statistics.median(row[index] for row in phase_snapshots) for index in range(4)]
        output.append({"case": case["case_id"], "batch": case["batch"], "context": case["context"],
                       "median_ms": medians,
                       "instrumentation_overhead_percent": 100 * (medians["B-Lossless"] / medians["A-Package"] - 1),
                       "phase_order": ["metadata_first_hints_grid_wait", "norm2", "thread0_GU_loop", "down_hints_grid_wait"],
                       "median_CTA_layer_sum_cycles": phases,
                       "phase_cycle_ranges_across_placements": [[min(row[index] for row in phase_snapshots), max(row[index] for row in phase_snapshots)] for index in range(4)]})
    return {"source": str(path), "scope": "thread0 scheduled intervals, not pure memory latency or additive wall shares", "cases": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.report)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for row in result["cases"]:
        print(row["case"], f"overhead {row['instrumentation_overhead_percent']:+.2f}%",
              row["median_CTA_layer_sum_cycles"])
