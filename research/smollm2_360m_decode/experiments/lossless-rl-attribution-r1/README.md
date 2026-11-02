# Lossless decoder: RL workload and cache-refresh attribution

Follow-up diagnostic to Experiment 225; no new speedup or verified RL wall-time point.

The three recorded two-update runs each ship eight usable traces, but sample 444/370/392 rollouts respectively. 436/362/384 are no-signal discards. The audit reconciles cumulative totals against step deltas and keeps shipped-batch token counts distinct from total sampled tokens. GRPO scoring and zero-advantage pruning require resampling: an eight-rollout admission cap could prevent training from completing. Rewards, task sampling and training dtypes were not changed.

The standalone refresh probe uses the same 360M snapshot and actual 32 GU tensors. Ten warm synchronous refresh calls have median 3.439 ms and range 3.323–7.125 ms; statistics median 0.193 ms. Cold construction is 29.209 ms including existing library initialization; it is not a clean encoding-only measurement. CUDA events enclose the synchronous host call and include interlaunch/idle intervals. These are diagnostic timings from one process, not a before/after optimization comparison or an in-vLLM refresh measurement.

All 32 layers decode bit-exactly after an in-place bit mutation and after restoration; the descriptor pointer stays stable. Both guards pass, sampled GPU peak 743 MiB, zero job swap; measured Torch peak allocated 595,433,472 bytes and reserved 643,825,664 bytes. Source-load time and model/package hashes are in the report. No model weights or compiled libraries are archived.

The measured standalone refresh cost cannot explain multi-second batch waits. Next measure fixed-work generation using the same policy snapshot and request workload, recording actual input/output tokens. Full RL still needs useful-trace yield and sampled-work accounting; do not claim wall-time gain from unequal workloads or remove filtering to force one.
