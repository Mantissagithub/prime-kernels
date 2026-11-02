# Lossless RL rollout attribution

Read-only accounting audit for three recorded two-update runs.
Counts and timings below come from the recorded files listed for each run. All time values are seconds unless the field says tokens/s.

## Per-run accounting

`progress/total_*` are cumulative snapshots. Per-step deltas are differences between adjacent cumulative snapshots; `rollout/*` counts are that step's window. The script checks both accounting identities.

Raw count fields: `progress/total_rollouts`, `progress/total_tasks`, and `progress/total_tokens` (cumulative sampled workload); `progress/input_tokens` and `progress/output_tokens` (tokens in the shipped effective batch, per step); `rollout/attempts`, `rollout/no_signal`, `rollout/discarded`, `rollout/stale`, `rollout/errored`, `rollout/cancelled`, `rollout/traces/queued`, and `rollout/traces/discarded` (per-step counts). `train/alphabet-sort/all/agent/is_trainable/mean` and `train/alphabet-sort/all/agent/is_admitted/mean` are per-step fractions.

| Run | step | rollout delta | task delta | total tokens from all rollouts | shipped input tokens | shipped output tokens | attempts | no signal | stale | errored | cancelled | queued traces | discarded traces | trainable fraction | admitted fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_r30 | 1 | 304 | 152 | 33311 | 326 | 181 | 304 | 300 | 0 | 0 | 0 | 4 | 300 | 0.013 | 1.000 |
| baseline_r30 | 2 | 140 | 70 | 15560 | 318 | 173 | 140 | 136 | 0 | 0 | 0 | 4 | 136 | 0.029 | 1.000 |
| candidate_old_r2 | 1 | 302 | 151 | 33042 | 326 | 181 | 302 | 298 | 0 | 0 | 0 | 4 | 298 | 0.013 | 1.000 |
| candidate_old_r2 | 2 | 68 | 34 | 7742 | 320 | 199 | 68 | 64 | 0 | 0 | 0 | 4 | 64 | 0.059 | 1.000 |
| candidate_fixed_r1 | 1 | 302 | 151 | 33058 | 326 | 181 | 302 | 298 | 0 | 0 | 0 | 4 | 298 | 0.013 | 1.000 |
| candidate_fixed_r1 | 2 | 90 | 45 | 9949 | 322 | 169 | 90 | 86 | 0 | 0 | 0 | 4 | 86 | 0.044 | 1.000 |

baseline_r30: cumulative final = {'rollouts': 444.0, 'tasks': 222.0, 'tokens': 48871.0}; summed step deltas = {'rollouts': 444.0, 'tasks': 222.0, 'tokens': 48871.0}; reconciles = True; attempts match rollout delta = True; summed per-step no-signal/discarded/stale/errored/cancelled = 436/436/0/0/0; queued traces = 8.

candidate_old_r2: cumulative final = {'rollouts': 370.0, 'tasks': 185.0, 'tokens': 40784.0}; summed step deltas = {'rollouts': 370.0, 'tasks': 185.0, 'tokens': 40784.0}; reconciles = True; attempts match rollout delta = True; summed per-step no-signal/discarded/stale/errored/cancelled = 362/362/0/0/0; queued traces = 8.

candidate_fixed_r1: cumulative final = {'rollouts': 392.0, 'tasks': 196.0, 'tokens': 43007.0}; summed step deltas = {'rollouts': 392.0, 'tasks': 196.0, 'tokens': 43007.0}; reconciles = True; attempts match rollout delta = True; summed per-step no-signal/discarded/stale/errored/cancelled = 384/384/0/0/0; queued traces = 8.

## Trainer phase timings

Source: `producer=trainer` rows from each `metrics.jsonl`. These phases can overlap with orchestrator/inference activity; do not add phase durations to estimate end-to-end wall time.
Raw fields are `time/step`, `time/wait_for_batch`, `time/forward_backward`, `time/broadcast_weights`, and `time/load_data` (seconds).

| Run | step | trainer step | wait for batch | forward/backward | broadcast weights | load data |
|---|---:|---:|---:|---:|---:|---:|
| baseline_r30 | 1 | 33.096 | 28.405 | 2.436 | 1.066 | 0.000 |
| baseline_r30 | 2 | 14.286 | 12.173 | 0.835 | 1.275 | 0.000 |
| candidate_old_r2 | 1 | 30.336 | 26.526 | 2.156 | 0.932 | 0.000 |
| candidate_old_r2 | 2 | 6.720 | 4.782 | 1.062 | 0.873 | 0.000 |
| candidate_fixed_r1 | 1 | 34.065 | 29.505 | 2.649 | 0.985 | 0.001 |
| candidate_fixed_r1 | 2 | 9.563 | 7.284 | 0.777 | 1.485 | 0.000 |

## Generation and cache observations

The RL metrics contain no generation-latency field. Inference logs do contain vLLM `Avg generation throughput` snapshots; their medians below are unweighted summaries of interval snapshots, not per-run token-time totals. Cache refresh log entries have event timestamps and generation IDs but no refresh duration field.
The throughput field is the inference-log text `Avg generation throughput: N tokens/s`; refresh events match `Lossless 360M GU cache refreshed` and expose `generation` plus timestamp, with no duration value.

| Run | generation throughput snapshots | snapshot median (tokens/s) | lossless cache refresh events (generation @ log time) | duration logged? |
|---|---:|---:|---|---|
| baseline_r30 | 4 | 301.850 | none | no |
| candidate_old_r2 | 3 | 328.700 | 2 @ 11-02 16:13:19, 3 @ 11-02 16:13:48, 4 @ 11-02 16:13:55 | no |
| candidate_fixed_r1 | 4 | 300.550 | 2 @ 11-02 17:12:26, 3 @ 11-02 17:13:00, 4 @ 11-02 17:13:10 | no |

## Interpretation and next measurement

Across these runs, each update ships four queued traces, while sampled rollouts vary substantially because groups without usable RL signal do not fill the batch. The throughput comparison is therefore not equal-work evidence. Reconcile counts before interpreting time; retain the reward/advantage filtering and report sampled rollouts, tokens, no-signal groups, and queued traces alongside phase timings.

The logged trainer phase breakdown makes `wait_for_batch` the limiting phase in these runs (with forward/backward and weight broadcast much smaller); the long wait tracks how many groups yield usable traces, not evidence that decoder compute dominates RL wall time. For the decoder comparison, use a separate fixed-work inference run: same policy snapshot, predetermined prompts, generation settings, and a target measured token workload, recording per-request input/output tokens and latency. Keep full-RL results diagnostic and report signal yield rather than changing rewards, dtypes, task sampling, or zero-advantage filtering to force equal prompt counts. Do not sum overlapped orchestrator, inference, and trainer durations into wall time.

## Source files

- `baseline_r30` metrics: `/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r30-full-forkserver-r1/runs/live-lossless-baseline-20261102-r1-on/r30-full-forkserver-on-2update/monitors/file/metrics.jsonl`
  Inference log: `/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r30-full-forkserver-r1/runs/live-lossless-baseline-20261102-r1-on/r30-full-forkserver-on-2update/logs/attempt_1/inference.log`
- `candidate_old_r2` metrics: `/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/lossless-full-kernel-integration-r1/runs/lossless-20261102-r2-candidate/lossless-full-forkserver-on-2update/monitors/file/metrics.jsonl`
  Inference log: `/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/lossless-full-kernel-integration-r1/runs/lossless-20261102-r2-candidate/lossless-full-forkserver-on-2update/logs/attempt_1/inference.log`
- `candidate_fixed_r1` metrics: `/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/lossless-full-kernel-integration-r1/runs/lossless-fix-20261102-r1-candidate/lossless-full-forkserver-on-2update/monitors/file/metrics.jsonl`
  Inference log: `/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/lossless-full-kernel-integration-r1/runs/lossless-fix-20261102-r1-candidate/lossless-full-forkserver-on-2update/logs/attempt_1/inference.log`
