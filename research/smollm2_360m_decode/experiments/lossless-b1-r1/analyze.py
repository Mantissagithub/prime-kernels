import argparse
import json
import statistics
from pathlib import Path


def summarize(files):
    cases = {}
    identities = None
    for file in files:
        report = json.loads(file.read_text())
        if identities is None:
            identities = report["arms"]
        if report["arms"] != identities or len(report["cases"]) != 6:
            raise ValueError(f"changed identities or missing cases: {file}")
        for case in report["cases"]:
            proof = case["parity_outside_timing"]["A_vs_B"]
            if not proof["exact_bitwise_all_layers"]:
                raise ValueError(f"parity failure: {file}")
            if case["context"] == 32:
                checks = proof["graph_refresh"]
                if not all(checks[field] for field in (
                    "dense_fallback_graph_ignores_poisoned_encoded_weights",
                    "graph_replay_matches_eager_bf16", "graph_output_changed")):
                    raise ValueError(f"refresh/fallback failure: {file}")
            samples = {arm: [row["median_ms_per_step"] for row in case["blocks"] if row["arm"] == arm]
                       for arm in ("A-Package", "B-Lossless")}
            if any(len(values) != 6 for values in samples.values()):
                raise ValueError("expected six placements per arm")
            medians = {arm: statistics.median(values) for arm, values in samples.items()}
            strict = [max(row["median_ms_per_step"] for row in case["blocks"]
                          if row["arm"] == "B-Lossless" and row["block"] == block)
                      < min(row["median_ms_per_step"] for row in case["blocks"]
                            if row["arm"] == "A-Package" and row["block"] == block)
                      for block in range(3)]
            cases.setdefault(case["case_id"], []).append({
                "file": str(file), "batch": case["batch"], "context": case["context"],
                "medians_ms": medians, "strict_blocks": strict,
                "change_percent": 100 * (medians["B-Lossless"] / medians["A-Package"] - 1)})
    return {"processes": len(files), "arms": identities, "cases": [
        {"case": name, "batch": rows[0]["batch"], "context": rows[0]["context"],
         "process_results": rows,
         "median_paired_change_percent": statistics.median(row["change_percent"] for row in rows),
         "paired_change_range_percent": [min(row["change_percent"] for row in rows),
                                          max(row["change_percent"] for row in rows)],
         "strict_blocks_passed": sum(sum(row["strict_blocks"]) for row in rows),
         "strict_blocks_total": len(rows) * 3}
        for name, rows in cases.items()]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.files)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for row in result["cases"]:
        print(row["case"], f"{row['median_paired_change_percent']:+.3f}%",
              row["paired_change_range_percent"], f"strict {row['strict_blocks_passed']}/{row['strict_blocks_total']}")
