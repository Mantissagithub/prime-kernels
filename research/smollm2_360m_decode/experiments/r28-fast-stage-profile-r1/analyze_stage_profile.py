#!/usr/bin/env python3
"""Summarize captured CTA-local cycles and instrumentation event overhead."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SLOTS = ("norm1", "qkv", "rope_cache", "split_kv", "merge", "o_residual",
         "norm2", "gate_up_silu", "down_residual", "layer_capture", "final_norm")
PROJECTIONS = ("qkv", "o_residual", "gate_up_silu", "down_residual")
ATTENTION = ("split_kv", "merge")
NORMS = ("norm1", "norm2")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "p90": percentile(values, 0.90), "max": max(values) if values else None}


def summarize_cycles(case: dict) -> dict:
    raw = case["profile_cycles"]
    if len(raw) != 32 or any(len(layer) != len(SLOTS) for layer in raw):
        raise RuntimeError(f"{case['case_id']}: unexpected layer/slot geometry")
    by_cta = {name: [0] * 24 for name in SLOTS}
    wait_by_cta = {name: [0] * 24 for name in SLOTS}
    for layer_index, layer in enumerate(raw):
        for slot_index, name in enumerate(SLOTS):
            for cta, fields in enumerate(layer[slot_index]):
                if len(fields) != 2:
                    raise RuntimeError(f"{case['case_id']}: malformed timing fields")
                work, wait = map(int, fields)
                by_cta[name][cta] += work
                wait_by_cta[name][cta] += wait
    phase = {}
    for name in SLOTS:
        ctas = [0] if name == "final_norm" else list(range(24))
        work = [by_cta[name][cta] for cta in ctas]
        wait = [wait_by_cta[name][cta] for cta in ctas]
        phase[name] = {"work_cycles_per_cta_summed_over_layers": distribution(work),
                       "grid_wait_cycles_per_cta_summed_over_layers": distribution(wait),
                       "work_plus_wait_cycles_per_cta_summed_over_layers": distribution([a + b for a, b in zip(work, wait)]),
                       "cta_coverage": ctas}
    totals_by_cta = [sum(by_cta[name][cta] + wait_by_cta[name][cta]
                         for name in SLOTS if name not in ("layer_capture", "final_norm"))
                     for cta in range(24)]
    step_totals_by_cta = totals_by_cta.copy()
    step_totals_by_cta[0] += by_cta["final_norm"][0]
    fractions = {}
    for name in SLOTS:
        if name in ("layer_capture", "final_norm"):
            continue
        vals = [(by_cta[name][cta] + wait_by_cta[name][cta]) / totals_by_cta[cta]
                for cta in range(24) if totals_by_cta[cta] > 0]
        fractions[name] = {"median_fraction_per_cta": statistics.median(vals),
                           "p90_fraction_per_cta": percentile(vals, 0.90),
                           "max_fraction_per_cta": max(vals)}
    families = {}
    for label, names in (("projections", PROJECTIONS), ("attention", ATTENTION), ("normalizations", NORMS),
                         ("rope_and_kv_cache", ("rope_cache",))):
        vals = [sum(by_cta[name][cta] + wait_by_cta[name][cta] for name in names) / totals_by_cta[cta]
                for cta in range(24) if totals_by_cta[cta] > 0]
        families[label] = {"median_fraction_per_cta": statistics.median(vals),
                           "p90_fraction_per_cta": percentile(vals, 0.90),
                           "max_fraction_per_cta": max(vals)}
    waits = [sum(wait_by_cta[name][cta] for name in SLOTS if name not in ("layer_capture", "final_norm"))
             for cta in range(24)]
    denom = [totals_by_cta[cta] for cta in range(24)]
    wait_fraction = [waits[cta] / denom[cta] for cta in range(24) if denom[cta] > 0]
    families["all_grid_waits"] = {"median_fraction_per_cta": statistics.median(wait_fraction),
                                  "p90_fraction_per_cta": percentile(wait_fraction, 0.90),
                                  "max_fraction_per_cta": max(wait_fraction)}
    capture = case["profile_capture_timing"]
    before, after = capture["sm_clock_mhz_before"], capture["sm_clock_mhz_after"]
    event_ms = float(capture["cuda_event_ms"])
    if event_ms <= 0:
        raise RuntimeError(f"{case['case_id']}: nonpositive profile capture event")
    clock_drift_percent = (abs(after - before) / ((before + after) / 2.0) * 100.0
                           if before is not None and after is not None and before > 0 and after > 0 else None)
    cycle_event_check = {
        "event_ms": event_ms, "sm_clock_mhz_sample_before": before,
        "sm_clock_mhz_sample_after": after, "sample_drift_percent": clock_drift_percent,
        "max_same_cta_stage_sum_effective_million_cycles_per_event_ms": max(step_totals_by_cta) / (event_ms * 1_000_000.0),
        "same_cta_step_cycle_distribution": distribution(step_totals_by_cta),
        "interpretation": "effective diagnostic ratio only: maximum named-stage cycle sum for one CTA divided by a separate full-kernel event; named intervals omit instrumentation, setup, and flush gaps, so this is not a calibrated clock64 rate; outside-kernel clocks.sm samples are not used to convert cycles",
    }
    return {"case_id": case["case_id"], "batch": case["batch"], "context": case["context"],
            "phase_cycles": phase, "per_cta_phase_fractions": fractions, "per_cta_family_fractions": families,
            "same_cta_step_cycles_vs_full_event": cycle_event_check,
            "fraction_denominator": "same-CTA sum across layers for named active stages; capture and final norm excluded"}


def summarize_events(cases: list[dict]) -> dict:
    result = {}
    for case in cases:
        by_arm: dict[str, list[float]] = {}
        block_medians: dict[str, list[float]] = {}
        for block in case["blocks"]:
            arm = block["arm"]
            vals = [float(value) for value in block["events_ms"]]
            by_arm.setdefault(arm, []).extend(vals)
            block_medians.setdefault(arm, []).append(float(block["median_ms_per_step"]))
        base = statistics.median(by_arm["A-R28-uninstrumented"])
        profiled = statistics.median(by_arm["B-R28-profiled"])
        result[case["case_id"]] = {
            arm: {"samples": len(vals), "median_ms": statistics.median(vals),
                  "p90_ms": percentile(vals, 0.90), "block_medians_ms": block_medians[arm]}
            for arm, vals in by_arm.items()
        }
        result[case["case_id"]]["instrumented_overhead_percent"] = (profiled / base - 1.0) * 100.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=HERE / "stage-profile-summary.json")
    args = parser.parse_args()
    documents = [json.loads(path.read_text()) for path in args.inputs]
    arm_identity = documents[0]["arms"]
    cases = []
    for path, data in zip(args.inputs, documents, strict=True):
        if data.get("mode", "").find("clock64") < 0:
            raise RuntimeError(f"{path}: input does not identify the clock64 profile method")
        if tuple(data.get("profile_slot_names", ())) != SLOTS:
            raise RuntimeError(f"{path}: profile slot name mismatch")
        if data["arms"] != arm_identity:
            raise RuntimeError(f"{path}: arm source/library hashes differ between case files")
        cases.extend(data["cases"])
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case IDs across input files")
    for case in cases:
        parity = case.get("parity_outside_timing", {}).get("A_vs_B", {})
        if parity.get("all_hidden_and_kv_bitwise_equal") is not True:
            raise RuntimeError(f"{case['case_id']}: exact hidden plus all-layer KV parity was not established")
    summary = {
        "schema": "r28-fast-stage-profile-summary-r1",
        "inputs": [str(path) for path in args.inputs],
        "interpretation": [
            "Cycle deltas are local to one CTA; no cross-CTA timestamps are subtracted.",
            "Per-CTA stage totals sum serial layer intervals on that CTA only; distributions do not sum across CTAs.",
            "Same-CTA named-stage cycles divided by a separate full-kernel event are an effective diagnostic ratio with instrumentation, setup, and flush gaps omitted; it is not a calibrated clock rate.",
            "Work includes memory/local synchronization stalls; grid wait is arrival skew and wait at an existing grid barrier.",
            "Fractions are diagnostic shares of same-CTA cycle totals, not kernel wall-time or hardware-counter attribution.",
            "Event overhead compares instrumented and uninstrumented R28; it is not a speed result.",
        ],
        "instrumentation_overhead": summarize_events(cases),
        "per_case_stage_cycles": [summarize_cycles(case) for case in cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
