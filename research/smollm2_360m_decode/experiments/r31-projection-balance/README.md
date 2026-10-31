# R31 projection-row balance: not adopted

Tiny-prl Experiment 195, November 1, 2026. This source is derived from R30 and changes only B1 QKV/gate-up row assignment and matching prefetch indices. It preserves per-row arithmetic, attention, O/down and B2 mapping.

Six three-block interleaved direct comparisons show changes below 0.11%, mixed across B1 contexts. All final hidden states and all-layer K/V match R30 bitwise, and all guards return zero. See `result_summary.json` and compressed raw event outputs in `results/`, pinned by `raw-output-manifest.json`. No serving or RL gain is established; R30 remains the best verified serving candidate.

The source and corrected build script are copied unchanged from `prime-rl/outputs/megakernel-20261025/r31-projection-balance/candidate/`. Building retains its original work-directory output default, overridable with `R31_BUILD_DIR`. No binary or package promotion is included.
