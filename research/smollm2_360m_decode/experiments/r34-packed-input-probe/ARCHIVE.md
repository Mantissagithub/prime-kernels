# Packed shared activation probe

The measured working directory was `/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r34-packed-input-probe`. The archived harness retains those measured checkpoint and build paths. Run it with the pinned uv environment under the repository's existing GPU/resource guard. No model weights or compiled libraries are included here.

Valid runs 2 and 3 use identical source, harness, checkpoint and DSO hashes, recorded in their JSON. Each uses five alternating placements per arm and five CUDA graph replays per placement. Replay must write every NaN-poisoned output before timing. All FP32 outputs and BF16 output bits match across both arms for all 32 real checkpoint layers. Both inner guards pass with zero swap and 791 MiB sampled GPU peaks.

| Projection | Batch | Packed latency change, run 2 | Run 3 |
|---|---:|---:|---:|
| QKV | 2 | -2.87% | -2.71% |
| O | 2 | -4.85% | -4.47% |
| Gate/up | 2 | -0.39% | -0.38% |
| Down | 2 | +0.29% | +0.41% |
| Down | 1 | -1.03% | -1.03% |

QKV/O at batch two and down at batch one separate across all placement medians in both processes. Gate/up overlaps in run 3. Other batch-one cases are neutral or slower. This is an isolated projection signal, not a full-kernel adoption or serving/RL result. Staging occurs inside each projection call; the full decoder reuses a shared activation buffer across phases and has additional shared allocations. Full-kernel behavior requires a separate experiment.

Run 1 is invalid. A stale stream handle launched work outside graph capture, leaving an empty timed graph. Its raw report and guards are retained to document the failure, and its timings are excluded. The archived harness fixes stream lookup and checks writes from replay itself. The original invalid report contains its own earlier harness/source hashes; they differ from the corrected archived sources.

The corrected harness and measurements were independently reviewed by a Luna agent. Timings imply roughly 184–234 GB/s of weight reads, below the separately measured approximately 250 GB/s laptop bandwidth. This arithmetic supports plausibility and is not a hardware-counter bandwidth measurement.
