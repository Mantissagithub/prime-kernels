#!/usr/bin/env python3
"""Audit rollout accounting and phase timings for the lossless RL run trio."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = {
    "baseline_r30": Path(
        "outputs/megakernel-20261025/r30-full-forkserver-r1/runs/"
        "live-lossless-baseline-20261102-r1-on/r30-full-forkserver-on-2update"
    ),
    "candidate_old_r2": Path(
        "outputs/megakernel-20261025/lossless-full-kernel-integration-r1/runs/"
        "lossless-20261102-r2-candidate/lossless-full-forkserver-on-2update"
    ),
    "candidate_fixed_r1": Path(
        "outputs/megakernel-20261025/lossless-full-kernel-integration-r1/runs/"
        "lossless-fix-20261102-r1-candidate/lossless-full-forkserver-on-2update"
    ),
}

TOTAL_KEYS = {
    "rollouts": "progress/total_rollouts",
    "tasks": "progress/total_tasks",
    "tokens": "progress/total_tokens",
}
WINDOW_KEYS = (
    "rollout/attempts",
    "rollout/discarded",
    "rollout/no_signal",
    "rollout/stale",
    "rollout/errored",
    "rollout/cancelled",
    "rollout/traces/queued",
    "rollout/traces/discarded",
    "rollout/queued/pruned",
    "rollout/queued/cancelled",
)
GENERATION_RATE = re.compile(r"Avg generation throughput:\s*([0-9.]+) tokens/s")
LOG_TIME = re.compile(r"\b(\d\d-\d\d\s+\d\d:\d\d:\d\d)\b")
CACHE_REFRESH = re.compile(r"Lossless 360M GU cache refreshed: \{'generation': (\d+)")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def metric(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def audit_run(name: str, run_dir: Path) -> dict:
    metrics_path = run_dir / "monitors/file/metrics.jsonl"
    inference_path = run_dir / "logs/attempt_1/inference.log"
    rows = read_jsonl(metrics_path)
    orch = [
        row
        for row in rows
        if row.get("producer") == "orch" and row.get("step") is not None and TOTAL_KEYS["rollouts"] in row
    ]
    trainer = [
        row
        for row in rows
        if row.get("producer") == "trainer" and row.get("step") is not None and "time/step" in row
    ]
    if not orch:
        raise ValueError(f"No orchestrator progress rows found in {metrics_path}")

    previous = {name: 0.0 for name in TOTAL_KEYS}
    step_rows = []
    for row in orch:
        counts = {}
        for label, key in TOTAL_KEYS.items():
            cumulative = metric(row.get(key))
            counts[label] = cumulative - previous[label]
            previous[label] = cumulative
        delta_attempts = metric(row.get("rollout/attempts"))
        step_rows.append(
            {
                "step": row["step"],
                "delta": counts,
                "window_attempts": delta_attempts,
                "window_counts": {key: metric(row.get(key)) for key in WINDOW_KEYS},
                "generation_tokens": {
                    key: metric(row.get(key))
                    for key in ("progress/input_tokens", "progress/output_tokens")
                },
                "cumulative": {label: metric(row.get(key)) for label, key in TOTAL_KEYS.items()},
                "orchestrator_seconds": {
                    key: metric(row.get(key))
                    for key in ("time/step", "time/wait_for_policy")
                    if isinstance(row.get(key), (int, float))
                },
                "trainable_fraction": row.get("train/alphabet-sort/all/agent/is_trainable/mean"),
                "admitted_fraction": row.get("train/alphabet-sort/all/agent/is_admitted/mean"),
            }
        )

    delta_sums = {
        label: sum(row["delta"][label] for row in step_rows)
        for label in TOTAL_KEYS
    }
    final_totals = step_rows[-1]["cumulative"]
    reconciles = all(delta_sums[key] == final_totals[key] for key in TOTAL_KEYS)
    attempts_match_rollout_delta = all(
        row["window_attempts"] == row["delta"]["rollouts"] for row in step_rows
    )
    window_totals = {
        key: sum(row["window_counts"][key] for row in step_rows)
        for key in WINDOW_KEYS
    }
    trainer_times = [
        {
            "step": row["step"],
            "seconds": {
                key: metric(row.get(key))
                for key in (
                    "time/step",
                    "time/wait_for_batch",
                    "time/forward_backward",
                    "time/broadcast_weights",
                    "time/load_data",
                )
                if isinstance(row.get(key), (int, float))
            },
        }
        for row in trainer
    ]

    generation_rates = []
    refreshes = []
    if inference_path.exists():
        with inference_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                time_match = LOG_TIME.search(line)
                timestamp = time_match.group(1) if time_match else None
                rate_match = GENERATION_RATE.search(line)
                if rate_match:
                    generation_rates.append(
                        {"timestamp": timestamp, "tokens_per_second": float(rate_match.group(1))}
                    )
                refresh_match = CACHE_REFRESH.search(line)
                if refresh_match:
                    refreshes.append(
                        {"timestamp": timestamp, "generation": int(refresh_match.group(1))}
                    )

    rates = [sample["tokens_per_second"] for sample in generation_rates]
    return {
        "name": name,
        "run_dir": str(run_dir),
        "metrics_source": str(metrics_path),
        "inference_log_source": str(inference_path),
        "steps": step_rows,
        "delta_sums": delta_sums,
        "final_cumulative": final_totals,
        "reconciles_cumulative_totals": reconciles,
        "attempts_match_rollout_delta": attempts_match_rollout_delta,
        "window_totals": window_totals,
        "trainer_times": trainer_times,
        "generation_throughput_log_samples": generation_rates,
        "generation_throughput_snapshot_median_tokens_per_second": (
            statistics.median(rates) if rates else None
        ),
        "lossless_cache_refresh_events": refreshes,
        "lossless_cache_refresh_durations_logged": False,
        "generation_latency_metric_present": any(
            "generation" in key.lower()
            and any(part in key.lower() for part in ("time", "latency", "duration"))
            for row in rows
            for key in row
        ),
    }


def fmt(value: object) -> str:
    if value is None:
        return "not logged"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def report(audits: list[dict]) -> str:
    lines = [
        "# Lossless RL rollout attribution",
        "",
        "Read-only accounting audit for three recorded two-update runs.",
        "Counts and timings below come from the recorded files listed for each run. All time values are seconds unless the field says tokens/s.",
        "",
        "## Per-run accounting",
        "",
        "`progress/total_*` are cumulative snapshots. Per-step deltas are differences between adjacent cumulative snapshots; `rollout/*` counts are that step's window. The script checks both accounting identities.",
        "",
        "Raw count fields: `progress/total_rollouts`, `progress/total_tasks`, and `progress/total_tokens` (cumulative sampled workload); `progress/input_tokens` and `progress/output_tokens` (tokens in the shipped effective batch, per step); `rollout/attempts`, `rollout/no_signal`, `rollout/discarded`, `rollout/stale`, `rollout/errored`, `rollout/cancelled`, `rollout/traces/queued`, and `rollout/traces/discarded` (per-step counts). `train/alphabet-sort/all/agent/is_trainable/mean` and `train/alphabet-sort/all/agent/is_admitted/mean` are per-step fractions.",
        "",
        "| Run | step | rollout delta | task delta | total tokens from all rollouts | shipped input tokens | shipped output tokens | attempts | no signal | stale | errored | cancelled | queued traces | discarded traces | trainable fraction | admitted fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in audits:
        for step in audit["steps"]:
            counts = step["window_counts"]
            lines.append(
                f"| {audit['name']} | {step['step']} | {step['delta']['rollouts']:.0f} | {step['delta']['tasks']:.0f} | {step['delta']['tokens']:.0f} | "
                f"{step['generation_tokens']['progress/input_tokens']:.0f} | {step['generation_tokens']['progress/output_tokens']:.0f} | "
                f"{counts['rollout/attempts']:.0f} | {counts['rollout/no_signal']:.0f} | {counts['rollout/stale']:.0f} | {counts['rollout/errored']:.0f} | "
                f"{counts['rollout/cancelled']:.0f} | {counts['rollout/traces/queued']:.0f} | {counts['rollout/traces/discarded']:.0f} | "
                f"{fmt(step['trainable_fraction'])} | {fmt(step['admitted_fraction'])} |"
            )
    for audit in audits:
        lines.append(
            f"\n{audit['name']}: cumulative final = {audit['final_cumulative']}; summed step deltas = {audit['delta_sums']}; "
            f"reconciles = {audit['reconciles_cumulative_totals']}; attempts match rollout delta = {audit['attempts_match_rollout_delta']}; "
            f"summed per-step no-signal/discarded/stale/errored/cancelled = "
            f"{audit['window_totals']['rollout/no_signal']:.0f}/{audit['window_totals']['rollout/discarded']:.0f}/"
            f"{audit['window_totals']['rollout/stale']:.0f}/{audit['window_totals']['rollout/errored']:.0f}/"
            f"{audit['window_totals']['rollout/cancelled']:.0f}; queued traces = "
            f"{audit['window_totals']['rollout/traces/queued']:.0f}."
        )
    lines.extend(
        [
            "",
            "## Trainer phase timings",
            "",
            "Source: `producer=trainer` rows from each `metrics.jsonl`. These phases can overlap with orchestrator/inference activity; do not add phase durations to estimate end-to-end wall time.",
            "Raw fields are `time/step`, `time/wait_for_batch`, `time/forward_backward`, `time/broadcast_weights`, and `time/load_data` (seconds).",
            "",
            "| Run | step | trainer step | wait for batch | forward/backward | broadcast weights | load data |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for audit in audits:
        for timing in audit["trainer_times"]:
            values = timing["seconds"]
            lines.append(
                f"| {audit['name']} | {timing['step']} | {fmt(values.get('time/step'))} | {fmt(values.get('time/wait_for_batch'))} | "
                f"{fmt(values.get('time/forward_backward'))} | {fmt(values.get('time/broadcast_weights'))} | {fmt(values.get('time/load_data'))} |"
            )
    lines.extend(
        [
            "",
            "## Generation and cache observations",
            "",
            "The RL metrics contain no generation-latency field. Inference logs do contain vLLM `Avg generation throughput` snapshots; their medians below are unweighted summaries of interval snapshots, not per-run token-time totals. Cache refresh log entries have event timestamps and generation IDs but no refresh duration field.",
            "The throughput field is the inference-log text `Avg generation throughput: N tokens/s`; refresh events match `Lossless 360M GU cache refreshed` and expose `generation` plus timestamp, with no duration value.",
            "",
            "| Run | generation throughput snapshots | snapshot median (tokens/s) | lossless cache refresh events (generation @ log time) | duration logged? |",
            "|---|---:|---:|---|---|",
        ]
    )
    for audit in audits:
        refreshes = ", ".join(
            f"{event['generation']} @ {event['timestamp']}" for event in audit["lossless_cache_refresh_events"]
        ) or "none"
        lines.append(
            f"| {audit['name']} | {len(audit['generation_throughput_log_samples'])} | "
            f"{fmt(audit['generation_throughput_snapshot_median_tokens_per_second'])} | {refreshes} | no |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and next measurement",
            "",
            "Across these runs, each update ships four queued traces, while sampled rollouts vary substantially because groups without usable RL signal do not fill the batch. The throughput comparison is therefore not equal-work evidence. Reconcile counts before interpreting time; retain the reward/advantage filtering and report sampled rollouts, tokens, no-signal groups, and queued traces alongside phase timings.",
            "",
            "The logged trainer phase breakdown makes `wait_for_batch` the limiting phase in these runs (with forward/backward and weight broadcast much smaller); the long wait tracks how many groups yield usable traces, not evidence that decoder compute dominates RL wall time. For the decoder comparison, use a separate fixed-work inference run: same policy snapshot, predetermined prompts, generation settings, and a target measured token workload, recording per-request input/output tokens and latency. Keep full-RL results diagnostic and report signal yield rather than changing rewards, dtypes, task sampling, or zero-advantage filtering to force equal prompt counts. Do not sum overlapped orchestrator, inference, and trainer durations into wall time.",
            "",
            "## Source files",
            "",
        ]
    )
    for audit in audits:
        lines.append(f"- `{audit['name']}` metrics: `{audit['metrics_source']}`")
        lines.append(f"  Inference log: `{audit['inference_log_source']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_RUNS["baseline_r30"])
    parser.add_argument("--old-candidate", type=Path, default=DEFAULT_RUNS["candidate_old_r2"])
    parser.add_argument("--fixed-candidate", type=Path, default=DEFAULT_RUNS["candidate_fixed_r1"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("report.md"))
    parser.add_argument("--json-output", type=Path, default=Path(__file__).with_name("audit.json"))
    args = parser.parse_args()
    runs = (
        ("baseline_r30", args.baseline),
        ("candidate_old_r2", args.old_candidate),
        ("candidate_fixed_r1", args.fixed_candidate),
    )
    audits = []
    for name, run_dir in runs:
        resolved = run_dir if run_dir.is_absolute() else REPO / run_dir
        audits.append(audit_run(name, resolved))
    if not all(audit["reconciles_cumulative_totals"] for audit in audits):
        raise SystemExit("Cumulative progress totals do not reconcile with per-step deltas")
    if not all(audit["attempts_match_rollout_delta"] for audit in audits):
        raise SystemExit("Per-step rollout attempts do not match rollout-count deltas")
    output = args.output if args.output.is_absolute() else REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report(audits), encoding="utf-8")
    json_output = args.json_output if args.json_output.is_absolute() else REPO / args.json_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json_output)
    for audit in audits:
        print(
            f"{audit['name']}: {audit['final_cumulative']} reconciles; "
            f"queued={audit['window_totals']['rollout/traces/queued']:.0f}; "
            f"no_signal={audit['window_totals']['rollout/no_signal']:.0f}; "
            f"generation_rate_samples={len(audit['generation_throughput_log_samples'])}; "
            f"cache_refreshes={len(audit['lossless_cache_refresh_events'])}"
        )


if __name__ == "__main__":
    main()
