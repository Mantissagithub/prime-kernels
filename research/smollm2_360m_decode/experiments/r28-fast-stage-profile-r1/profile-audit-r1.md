# FAST R28 stage-profile audit

Independent CPU recalculation from the six one-case JSON captures. This is diagnostic profiling, not a new numerical or serving-quality gate.

## Integrity and run envelope

- All six instrumented-vs-uninstrumented comparisons are bitwise exact for final hidden state and K/V at all 32 layers. Each arm has 30 CUDA-event samples per case (three alternating ABBA/BAAB blocks, two placements per block, five repetitions per placement).
- Every per-case 3072 MiB guard returned 0 with no stop reason and zero swap. Peak sampled GPU memory was 1315–1991 MiB. The canonical `runtime/run_gpu.sh` envelope supplied the shared GPU lock, idle check, systemd memory/swap/runtime controls, and pinned no-project interpreter; each shape was a separate process.
- Instrumented DSO SHA-256: `97da2f8eba6e3531fa89d39c6c82a3e61b4b61041ca348c0ffa7b6c4fea7d107`; uninstrumented FAST R28 DSO: `df3f03f7d680719b2f282fe7ad0a41cc24ff20f04d14809756b9b5e63a7f6603`. The private clone added timing only; the canonical 28-argument decode ABI is retained.
- `ptxas` reports B1: 250 registers, 36,824 B shared memory; B2: 244 registers, 43,232 B shared memory; both have zero spills and zero stack bytes. The shared-memory increase is 176 B for profile scratch.

## Instrumentation overhead

The percentage compares pooled raw CUDA-event medians for instrumented B against uninstrumented A in the same case's alternating blocks. Clearing profile storage is outside the event window. The separate profile-capture event is shown later and is not used for these overhead percentages.

| Shape | R28 A median (ms) | Instrumented B median (ms) | Instrumentation delta |
|---|---:|---:|---:|
| B1 / 32 | 3.3481 | 3.4016 | +1.60% |
| B1 / 128 | 3.0356 | 3.0563 | +0.68% |
| B1 / 1024 synthetic | 3.3426 | 3.3602 | +0.53% |
| B2 / 32 | 3.1565 | 3.1698 | +0.42% |
| B2 / 128 | 3.2435 | 3.2572 | +0.42% |
| B2 / 1024 synthetic | 3.6916 | 3.6987 | +0.19% |

The profile adds a small measured event-time cost on these cases, largest at B1/context 32. This is why the cycle shares below are diagnostic and should not be presented as unperturbed timing.

## Per-CTA phase shares

Each row is the median across 24 CTAs of that CTA's own phase fraction, after summing its serial per-layer intervals. Phase work includes memory and local-sync stalls. Projection, attention, norm, and RoPE/cache shares include their associated phase work plus any grid wait within those phases. `Grid wait` is a component of those rows and must not be added to them.

| Shape | Projections | Split-KV + merge | Norms | RoPE/cache | All grid waits, overlapping component |
|---|---:|---:|---:|---:|---:|
| B1 / 32 | 90.2% | 4.3% | 3.4% | 2.2% | 11.5% |
| B1 / 128 | 89.1% | 5.4% | 3.3% | 2.2% | 11.6% |
| B1 / 1024 synthetic | 82.1% | 12.8% | 3.0% | 2.0% | 8.9% |
| B2 / 32 | 87.6% | 4.9% | 4.7% | 2.8% | 8.6% |
| B2 / 128 | 86.3% | 6.4% | 4.5% | 2.8% | 8.7% |
| B2 / 1024 synthetic | 76.3% | 17.3% | 4.0% | 2.4% | 7.1% |

These fractions come from same-CTA totals only. No CTA durations are summed together. Short-context attention is a small share; at B2/context 1024 it grows to 17.3%, while projections remain the largest family at 76.3%.

## Work and grid-wait distributions

Values below are millions of `clock64` cycles per CTA, median across CTAs after summing each CTA's intervals over 32 layers. Wait p90 is also shown for split-KV because its CTA distribution is notably skewed at B2/context 128. These are elapsed intervals, not arithmetic-only costs.

| Shape | QKV work / wait | Split-KV work / wait | Merge work | O+residual work / wait | Gate+up+SiLU work / wait | Down+residual work / wait |
|---|---:|---:|---:|---:|---:|---:|
| B1 / 32 | 0.968 / 0.153 | 0.004 / 0.234 | 0.096 | 0.600 / 0.090 | 3.265 / 0.167 | 1.606 / 0.188 |
| B1 / 128 | 0.978 / 0.155 | 0.004 / 0.288 | 0.139 | 0.601 / 0.096 | 3.267 / 0.195 | 1.630 / 0.185 |
| B1 / 1024 synthetic | 0.983 / 0.162 | 0.707 / 0.073 | 0.333 | 0.591 / 0.106 | 3.282 / 0.194 | 1.616 / 0.191 |
| B2 / 32 | 1.082 / 0.096 | 0.004 / 0.252 | 0.146 | 0.665 / 0.090 | 3.361 / 0.081 | 1.661 / 0.184 |
| B2 / 128 | 1.093 / 0.100 | 0.283 / 0.050 | 0.210 | 0.676 / 0.086 | 3.398 / 0.086 | 1.660 / 0.210 |
| B2 / 1024 synthetic | 1.100 / 0.103 | 1.180 / 0.142 | 0.327 | 0.663 / 0.089 | 3.401 / 0.086 | 1.656 / 0.198 |

At short B1 contexts, `split_kv` thread 0 often has little or no chunk work while other warps/CTAs perform the active chunks. Its large wait field includes arrival skew and the work still in flight elsewhere; it is not pure barrier overhead. B2/context 128 also has a long tail (split-KV wait p90 about 0.330 M cycles versus median 0.050 M), consistent with a skewed CTA distribution. At B1/context 1024 and B2/context 1024 the split-KV work median rises substantially (0.707 M and 1.180 M cycles). Those long-context changes align with the split-KV+merge family growing to 12.8% and 17.3%.

The largest single measured interval is fused gate/up projection plus SiLU at 3.26–3.40 M work cycles per CTA; down projection plus residual is next at 1.61–1.66 M. This identifies where the elapsed work concentrates, but does not show whether those phases are compute-, load-, or instruction-limited.

## Stage totals against the full event

The separate profile-capture CUDA event measured 3.074–3.741 ms. For each case, the median CTA's accumulated named-stage total was 7.82–9.57 M cycles and the maximum was 7.83–9.58 M cycles. Dividing the maximum same-CTA total by the separate event gives an effective diagnostic ratio of 2.545–2.561 million recorded cycles per event millisecond. This is not a calibrated `clock64` frequency: the named intervals omit instrumentation, setup, and flush gaps, and the profile capture is a separate invocation. External `clocks.sm` snapshots were taken outside the kernel; they varied from 2250 to 2610 MHz and are not used to convert cycles. This ratio is only a coarse cross-check. No cross-CTA timestamps or sums are used.

The per-case values and raw CTA vectors are in `results/stage-profile-summary.json` and the six source JSON files. All profile values were captured after exact hidden/KV parity checks.

## Follow-up hypothesis

The profile points to the fused gate/up phase as the highest-payoff target; split-KV wait at short contexts is not the leading cost. The existing FAST R28→R30 phase-ahead prefetch comparison already reports exact parity and about 3.0–3.3% lower event medians at B1, versus about 0.7–1.0% at B2 (six-case artifact: `../r30-phase-prefetch/RESULTS.md`). A focused next experiment should isolate the O→gate/up prefetch site, which currently issues hints after O projection and immediately before its grid barrier, then measure the same B1/B2 cases against R30. Moving that site's hints earlier into the O projection loop could provide more lead time for the 3.3–3.4 M-cycle gate/up phase, but may contend with O-weight reads; use a one-site variant and exact hidden/all-layer-KV parity before attributing any gain. This is a hypothesis only; the stage clocks do not prove memory stalls or guarantee improvement.
