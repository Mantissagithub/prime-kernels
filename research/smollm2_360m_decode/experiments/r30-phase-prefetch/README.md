# R30 phase-ahead prefetch: decoder-component improvement

This is the measured R28-derived candidate in tiny-prl Experiment 192, November 1, 2026. PTX L2 hints fetch the next projection phase's weight rows before cooperative barriers. Arithmetic, workspace and ABI are unchanged.

B1 decoder medians improve by 3.03–3.32% at contexts 32/128/1024, with both candidate placements beating both controls in all three blocks per shape. B2 changes are 0.69–0.98%; context 1024 passes two of three blocks. All six cases match R28's final hidden state and all-layer K/V bitwise. See `RESULTS.md`, `result_summary.json` and raw CUDA-event `results/`.

The source is byte-identical to `prime-rl/outputs/megakernel-20261025/r30-phase-prefetch/candidate/`. `archive-sha256.json` pins the archived files. `build.sh` builds SM89, retaining the original work-directory output default; override it with `R30_BUILD_DIR`. No binary is committed.

Serving validation in Experiment 193 confirms a further 2.56–2.77% ITL gain at B1 and 0.35–0.63% at B2, with exact R28/R30 generated sequences and sampled/top-five logprobs. See `serving/`. This does not establish an RL wall-time gain or resolve R28's numerical acceptance misses; the packaged 360M implementation remains R25.
