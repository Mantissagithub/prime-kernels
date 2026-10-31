# r29-proj-prefetch: measured R28 follow-up, not adopted

Source and measured results from tiny-prl Experiment 191, November 1, 2026. See `RESULTS.md` and `result_summary.json`; raw per-case CUDA-event samples and exact hidden/KV comparisons are in `results/`. The baseline is Opus's fast R28, archived in sibling `r28-flash-decode/`.

The source files are byte-identical to the measured private candidate at `prime-rl/outputs/megakernel-20261025/r29-proj-prefetch/candidate/`. `archive-sha256.json` records the archived file hashes. `build.sh` builds SM89 and retains the original work-directory output default, overridable with `R29_BUILD_DIR`. Building does not select this kernel in the package.

No package promotion or serving/RL improvement follows from this experiment. The full benchmark harness and guard telemetry remain in the original work directory.
