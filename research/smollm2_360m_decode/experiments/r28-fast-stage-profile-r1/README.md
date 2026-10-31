# FAST R28 per-stage profile preparation

Status: private instrumentation clone compiled and profiled in six bounded cases. The target is canonical FAST R28 (`r28-flash-decode`), not the older same-number RMS experiment. The workspace is ignored under `prime-rl/outputs`.

## Measurement design

The private clone adds only clock reads and a profile-only pointer API. `decode360` keeps its canonical 28 arguments and its model math, layout, barriers, and launch geometry are copied from FAST R28. Timing data goes to a separate 135,168-byte device buffer, never through the BF16 workspace/debug tensors. A single thread per CTA reads `clock64()` at stage boundaries, holds the elapsed values in 176 bytes of shared memory, and flushes the stage rows once per layer. This avoids cross-CTA timestamp subtraction and keeps global writes out of each timed interval.

Each global-barrier stage records two deltas: work until thread 0 reaches the existing `grid.sync()`, and time from that observation until thread 0 exits it. The second interval includes CTA arrival skew, intra-CTA arrival skew, and the grid wait; it does not identify the cause of skew. Local stages record their interval after their existing CTA-local synchronization. Work cycles include memory stalls, instructions, and local synchronization. They are elapsed CTA-local cycles, not pure compute time or hardware performance counters.

| Slot | FAST R28 region | Boundary / notes |
|---|---|---|
| norm1 | First RMSNorm per layer | Local work through its existing CTA barriers; repeated for each batch row. |
| qkv | Fused Q/K/V projection | Work and the wait at the existing post-projection grid barrier. |
| rope_cache | RoPE, V staging, current K/V append | Work and wait at the existing cache-visibility barrier. |
| split_kv | Split-KV online attention | Work and wait at the barrier publishing partials. Some warps may have no chunk work for short contexts. |
| merge | Merge partials and form attention output | Local work through existing `__syncthreads`; ends before O projection. |
| o_residual | O projection and first residual | Work and wait at the existing grid barrier. |
| norm2 | Second RMSNorm | Local work through its existing CTA barriers. |
| gate_up_silu | Fused gate/up projection plus SwiGLU | Aggregate phase; work and wait at its existing grid barrier. |
| down_residual | Activation staging, down projection, residual | Work and wait at the existing layer-completion barrier. |
| layer_capture | Optional output capture | Disabled in this harness; zero and marked absent. |
| final_norm | Final RMSNorm | Captured only for CTA 0, after all decoder layers. |

The runner uses the same direct decoder harness and FAST R28 DSO as the uninstrumented arm, with the four frozen seed-17 fixtures at contexts 32/128 and deterministic synthetic seed prompts at context 1024, for both B1 and B2. Each case runs in its own process. The canonical `runtime/run_gpu.sh` supplies the GPU flock, idle-GPU check, systemd `MemoryMax`, zero-swap, 600-second limit, and pinned no-project interpreter; the nested case guard adds the 3072 MiB GPU and 2 GiB host/disk limits. Within each case, it interleaves uninstrumented and instrumented R28 in alternating ABBA/BAAB event blocks. CUDA-event timing brackets only the decoder call; KV restore, profile-buffer clearing, prefill, and parity checks stay outside the event interval. A separate instrumented invocation captures stage rows plus a CUDA-event duration and SM-clock samples. The runner asserts bitwise equality of final hidden and every layer's key and value cache between the two R28 arms before it emits profile cycles. The analyzer repeats this exact-parity requirement before calculating phase fractions.

The CPU analyzer reports distributions across CTA-local cycle totals and per-CTA phase fractions. It never sums durations across CTAs. It compares each CTA's accumulated stage cycles with the complete kernel event using bracketing SM clock samples as a rough coverage cross-check; this estimate is approximate because small instrumentation and launch intervals are not represented by stage slots. It does not claim cycle-to-time equivalence. For a grid-barrier phase, per-CTA work plus wait is useful for comparing where the time is spent; it is not a replacement for CUDA-event step time. The current unprivileged environment reports `RmProfilingAdminOnly`, so this method does not request Nsight hardware counters or change driver permissions.

## Pinned inputs and scope

- FAST R28 source SHA-256: `1d121b7f89561a955b1fcc665daa7535a13816821920fe4030c4040e7bf61313`.
- FAST R28 include SHA-256: `0ac2ba97ed2bfc4684c476987635e98e06affe2323349e4f9c4d9a27ed1247f0`.
- Uninstrumented R28 DSO SHA-256: `df3f03f7d680719b2f282fe7ad0a41cc24ff20f04d14809756b9b5e63a7f6603`.
- Private instrumentation clone hashes and the source ABI check are pinned by `preflight.py`.
- No source outside this ignored directory was edited. The instrumented variant does not change BF16 weights or trainer dtypes.

## Review/run sequence

1. Run CPU-only preflight with `/home/pradheep/.local/bin/uv run --offline --no-project --python /home/pradheep/.venv/bin/python python outputs/megakernel-20261025/r28-fast-stage-profile-r1/preflight.py`.
2. The private DSO was compiled after source review under the canonical GPU lock, CUDA-hidden, with `MemoryMax=2G`, zero swap, and a 120-second systemd deadline. See `build/ptxas.log` and `result_summary.json` for resource usage.
3. All six cases ran as separate, serial processes through `runtime/run_gpu.sh` and the per-case 3072 MiB GPU guard. The canonical lock and host guard were active; every case returned zero with no stop reason and no swap.
4. The summary in `results/stage-profile-summary.json` was generated with the pinned no-project interpreter. Two initial cases met the pre-agreed <=5% instrumentation-overhead condition before the remaining four were run.

## Build and bounded profiling result

The instrumented DSO SHA-256 is `97da2f8eba6e3531fa89d39c6c82a3e61b4b61041ca348c0ffa7b6c4fea7d107`. Ptxas reports B1 at 250 registers and 36,824 bytes shared memory, B2 at 244 registers and 43,232 bytes; both have zero spill stores, zero spill loads, and zero stack frame. Against the uninstrumented FAST R28 build (B1 249 registers/36,644 bytes; B2 236 registers/43,056 bytes), instrumentation adds 1 register/180 bytes for B1 and 8 registers/176 bytes for B2. It does not introduce spills.

For the six cases (B1/B2 at contexts 32/128/1024), instrumentation overhead from the pooled 30 CUDA-event samples per arm ranges from 0.19% to 1.60%. The R28 uninstrumented and profiled arms are bitwise equal for final hidden state and every layer's K/V cache in all six cases. Peak sampled GPU memory ranges from 1,315 to 1,991 MiB; all inner and outer guards returned zero, with no stop reason or swap.

The cycle fractions are descriptive, same-CTA totals across decoder layers. Median per-CTA fractions attribute roughly 36.5–44.0% to fused gate/up/SwiGLU, 19.4–23.0% to down projection/residual, and 12.6–14.3% to QKV at these six shapes; split-KV rises to 13.8% for synthetic B2/context 1024. These fractions include local stalls and local barriers, and named grid waits include arrival skew. They are not hardware-counter compute utilization or independent wall-time fractions. The event-bracketing clock estimate is only an approximate coverage cross-check; do not sum CTA durations or interpret its small over/under-shoot as a performance result.

Machine-readable hashes, ptxas values, six arm medians, per-case parity, and guard telemetry are in `result_summary.json`; raw samples and phase distributions remain under `results/`.

These data can direct a follow-up optimization, but do not establish a numerical gate, serving quality, a speedup, or the benefit of any future prefetch change. Keep the existing quality floors and RL gate unchanged.
