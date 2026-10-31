# R28 versus R30 serving comparison

Experiment 193 uses four fresh vLLM processes in R28/R30/R30/R28 order, six shapes, 64 greedy tokens, three warmups and seven repeats. The plain JSON summary contains timestamp-derived request ITL, all eight successful guard records, route/hash evidence, exact sequence/logprob comparisons and clock ranges. Full raw per-arm outputs are gzip-compressed; `raw-output-manifest.json` pins both compressed and original hashes. Clock CSVs are included.

R30 improves ITL by 2.56–2.77% at B1 and 0.35–0.63% at B2. Both candidate arms beat both controls at all six shapes. Every cross-arm pairing matches all 63 generated sequences, sampled logprobs and top-five entries exactly. GPU peaks are 2,475 MiB; all median SM clocks are 2610 MHz, with unlocked clock ranges recorded in the summary.

The scripts are archived unchanged from the measured work directory and retain its absolute paths. To rerun the CPU analyzer, decompress the raw JSONs and place the clock logs and guard records in the original `prime-rl/outputs/megakernel-20261025/r30-phase-prefetch/serving/` layout. The summary embeds the guard records. The full local telemetry is preserved there; it is not all duplicated in this archive.

This preserves R28's outputs and adds serving speed; it does not resolve R28's stock-relative numerical misses or establish an RL wall-time gain. No package promotion is made.
