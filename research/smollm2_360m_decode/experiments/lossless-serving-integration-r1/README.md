# Lossless SmolLM2-360M serving integration evidence

This archive records the codec proof, strict R30 parity check, and one guarded two-update RL pair for the lossless gate/up decoder route. It contains source scripts, configs, compact metrics, compressed JSON evidence and a source-hash manifest. It does not contain model weights or compiled shared libraries.

## Results

The codec proof checked all 65,536 BF16 bit patterns. It also exercised overflow fallback and recovery: all 32 layers became inactive on overflow and active after recovery, while descriptor and encoded-buffer pointers stayed stable. The complete proof is `proof-r2.json.gz`.

The parity run covered six cases: batch 1/2 at contexts 32/128 plus synthetic context 1024 at batch 1/2. In the direct R30-versus-lossless comparison, candidate final hidden states and every key/value cache tensor across all 32 layers matched R30 bitwise in all six cases. The report also stores each arm’s native-reference records separately; those are not the basis for the R30-to-candidate bitwise claim. In the batch-2/context-32 graph-refresh check, a gate/up-only in-place mutation changed graph output and graph replay matched eager BF16 output. Parity checks, KV restoration and prefill are outside the timing intervals. See `parity-r2.json.gz` and `parity-validation-script.py`.

One real-RL pair completed two updates per arm on the RTX 4060 Laptop GPU. The baseline used the R30 fork-server route; the candidate used the lossless full decoder. Baseline worker traces verified that the R30 route loaded and first-decoded. Candidate logs report generations 1 through 4 over the initial capture and three checkpoint refreshes, all with 32 active layers. The two arms completed different work, so wall times are observations only and do not establish an RL speedup:

| Arm | Recorded elapsed | Guarded wall | GPU peak | Updates | Tasks | Rollouts | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R30 baseline | 83.795956 s | 86.567222 s | 7,222 MiB | 2 | 222 | 444 | 48,871 |
| Lossless candidate | 67.474434 s | 67.726367 s | 7,398 MiB | 2 | 185 | 370 | 40,784 |

Every update has finite loss, gradient norm and mismatch KL. Per-update values are in `trainer-metrics-summary.json.gz`. The two configurations have the same training, task, sampling and resource settings after excluding run/output identity and inference environment variables. Elapsed scopes differ: the baseline records the actual uv RL child; the candidate wrapper includes post-run log validation in its timed invocation. The elapsed values are not directly comparable. This is one pair only: no variance estimate or RL performance claim is made.

A separate full-decoder timing run used three alternating blocks, six placements and 20 CUDA-event repeats per arm/case (120 timed events per arm/case). Each value below is the median of the six placement medians:

| Batch | Context | R30 | Lossless | Change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 2.885200 ms | 2.885632 ms | +0.01% |
| 1 | 128 | 2.936320 ms | 2.936832 ms | +0.02% |
| 1 | 1024 | 3.237936 ms | 3.238912 ms | +0.03% |
| 2 | 32 | 3.121824 ms | 3.452416 ms | +10.59% |
| 2 | 128 | 3.207504 ms | 3.534768 ms | +10.20% |
| 2 | 1024 | 3.658752 ms | 3.982336 ms | +8.84% |

Batch 1 is effectively neutral; batch 2 is slower at every context. The candidate did not beat R30 across all three blocks in any case, so the route remains opt-in research and the default is unchanged. Timing guards passed with zero swap and a 2,251 MiB GPU peak. See `timing-r1.json.gz` and the timing guard JSON files.

The first candidate integration attempt stopped during checkpoint weight reload with an inference-tensor write outside `InferenceMode`. The cache refresh path was decorated with `torch.inference_mode`; the passing retry completed both updates and three refreshes, generations 1 through 4, with 32 active layers. The concise original error is retained in `candidate-r1-error-excerpt.txt`.

## Component-only microprobe

Experiment 223 is separate from the RL pair. In the fixed-work staged-CSR gate/up projection probe, six placements and five graph replays per placement measured 960 calls per arm. Median latency was 43.132 us/layer dense and 41.584 us/layer staged CSR, 3.59% lower by median. This is a component result, not an RL or full-decoder speedup. Exact output/decoded-weight evidence and guards are in `staged-csr-result.json.gz`, the two guard JSON files, and the source/arithmetic proof files.

## Reproduction pointers

The archived Python files, TOML recipes and staged-CSR CUDA source are snapshots from the original run directories. Commands, complete logs, resolved configs, route traces and original guard samples remain under `prime-rl/outputs/megakernel-20261025/{r30-full-forkserver-r1,lossless-full-kernel-integration-r1,gateup-lossless-staged-csr-r1}` in the development workspace. Reuse the original scripts for controlled reproduction; no GPU run is part of this archive step.

The parity and full-RL scripts reference their original source/build paths. SHA-256 identities for baseline/candidate source, includes, DSO, configs, metrics and the runtime refresh path are recorded in `source-hashes.json`. DSO and model weights are intentionally not copied.
