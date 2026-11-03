## Experiment 239: ten-update comparison is inconclusive; profile rollout waiting

The accepted two-sequence 360M invariant-serving/mixed-repair profile completes ten updates with full final checkpoints in both startup modes. Both use the same task, FP32 optimization/reduction, reward/admission logic, kernel and private request-seed hook. Only early_trainer_start differs. This is one process per arm, not a repeated speed comparison.

| Startup | Whole RL seconds | Successful requests | Generated tokens | Finalized rollouts | Finalized tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 158.865067 | 968 | 29126 | 958 | 105453 |
| early | 160.390165 | 1040 | 31677 | 1030 | 113760 |

Trainer loss/gradient/KL metrics match through update 4 and differ from update 5. Finalized rollout work is already not exact from update 2. Early startup generates 8.755% more tokens overall, so these times establish neither a speedup nor a regression. The five-per-arm sweep stops at the work gate. The reason for the later training divergence is not isolated. The verified two-update Experiment 238 result remains scoped to that shorter workload.

Native trainer wall timers locate the main wait. Control/early totals are 93.496/101.062 s waiting for batches, 9.406/9.876 s in the forward/backward bucket (which includes optimizer, gradient handling and annotations), and 13.478/12.635 s publishing weights. Load-data totals are about 0.004 s. Final checkpoint log intervals are 6/5 s. Startup-to-training-loop intervals are approximately 32/26 s from one-second logs measured against the runner start clock; remaining time includes metric logging, final synchronization, shutdown and log rounding. Publication includes handshake and may include outstanding GPU completion, not full engine reload. Orchestrator timers overlap this lane and are not added. Guard GPU-util samples aligned approximately to control steps 2-10 average 70.7% during trainer wait; this is GPU utilization, not MFU or memory bandwidth.

A separate instrumented two-update run times the actual V2 model runner, fixing the earlier research counter's unused V1 target. It records 6,810 model/sample calls. Pure-decode execution spans total 10.272 s (70.43% of measured serving stream spans); mixed execution 1.692 s (11.60%); prefill execution 0.797 s (5.46%); sampling 1.824 s (12.51%). B1/B2 pure-decode runner median spans are 2.887/3.140 ms. These are CUDA-stream spans including launch gaps, not isolated kernel durations; CPU spans overlap them and must not be summed. The diagnostic is excluded from wall-time comparisons, and flushes at weight refresh can omit the shutdown tail. No hardware-counter bandwidth/cache claim is made.

This directs the next screen toward pure decode rather than assuming mixed metadata is the largest cost. All three full-RL guards pass with zero job swap, including full checkpoint saves. Existing broadcast retention keeps only the current and previous snapshots, so no cleanup-policy change was needed for ten updates. Evidence: `prime-kernels/research/smollm2_360m_decode/experiments/rl-10step-startup-r1/`.
