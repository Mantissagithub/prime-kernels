# B2 reproducibility and checkpointed RL startup

## Experiment 237: stabilize native projections and mixed decode routing

The first divergent native prefill operation is the first-layer gate/up projection. At the same prompt row, singleton and paired prefills have identical BF16 input, but 2,028 of 5,120 output words differ when the matrix grows from 80 to 160 rows. The preceding QKV and output projections match at the inspected row. This is a fixed-weight diagnostic, not a training or performance comparison.

With the actual RL FP32 LM head and processed-logprob settings, 3/16 fixed seeded requests change token streams between singleton and paired execution; all 16 sampled-logprob streams differ. Per-row FP32 head evaluation does not fix it; disabling reduced-precision cuBLAS accumulation also fails. Two earlier BF16-head probes are exploratory only because that head does not match the RL preset. An initial source search missed the batch-invariant implementation; the retained record corrects that conclusion.

The installed vLLM `VLLM_BATCH_INVARIANT=1` path uses invariant native linear and attention operations. All 80 requests across singleton, paired, reversed, cross-paired and repeated groupings then match their tokens and sampled logprobs exactly. Training still uses FP32 optimization/reduction. This does not yet fix live mixed batches: staggering a new prefill beside an ongoing decode produces query lengths [1,80] and sends the whole batch through native execution. Two of four ongoing requests change token streams; all four change logprobs. Newly prefilling requests match.

The integrated opt-in repair runs native forward for the mixed batch, then recomputes only its decode rows with the existing lossless decoder. It replaces those hidden-state rows and their current-token KV entries, keeping the decode arithmetic path consistent. Pure-decode fast paths are unchanged. Unsupported metadata or mixed graph capture falls back to native. The repair runs only for lossless 360M when the invariant flag is enabled. It adds duplicate work to mixed steps, so no speedup is inferred from this correctness change.

The private repair and the actual integrated path both make all eight staggered requests exactly match singleton tokens/logprobs; the integrated counter confirms four mixed decode rows. Existing decode geometry/routing tests pass 16/16, and focused Ruff checks pass. The opt-in `smollm2-360m-batch-invariant.toml` overlay resolves successfully through the packaged CLI with two sequence slots, FP32 training and final checkpoints. These short-context probes do not establish batch invariance for every long-context kernel shape, prefix-cache state, training schedule or seedless sampling. Defaults are unchanged. Evidence: `prime-kernels/research/smollm2_360m_decode/experiments/b2-reproducibility-r1/`.

## Experiment 238: checkpointed B2-capable RL startup gain confirmed

Five serial-start and five early-start full-parameter 360M runs use two vLLM sequence slots, default max_inflight=4, two alphabet-sort updates, stream64 weights, full final checkpoints and FP32 optimization/reduction. Both arms use the accepted kernel, batch-invariant native serving and Experiment 237's mixed-decode repair. Only early_trainer_start differs. Distinct deterministic per-prompt request seeds remain a private benchmark hook; the normal CLI is still stochastic.

Whole-RL median 62.548471 to 55.202267 seconds (-11.745%). Serial range 60.910289-71.590824 s; early range 54.492884-57.824186 s. Every early process beats every control. Both strict ABBA blocks and the final pair pass; block-mean changes are -14.995%, -10.689%, -13.916%.

| Order | Startup | Whole RL seconds | Requests | Generated tokens |
| --- | --- | ---: | ---: | ---: |
| c1 | control | 71.590824 | 174 | 5113 |
| e1 | early | 57.824186 | 175 | 5158 |
| e2 | early | 54.808486 | 174 | 5113 |
| c2 | control | 60.910289 | 173 | 5102 |
| c3 | control | 62.548471 | 173 | 5102 |
| e3 | early | 55.202267 | 175 | 5158 |
| e4 | early | 55.327365 | 174 | 5113 |
| c4 | control | 61.209609 | 172 | 5047 |
| c5 | control | 63.302306 | 174 | 5113 |
| e5 | early | 54.492884 | 174 | 5113 |

All ten runs match finalized work exactly at 166 rollouts, 83 tasks and 18,138 tokens, with identical loss, gradient norm and mismatch KL at both trainer steps. Common prompt/seed requests return identical token IDs/counts. The first 83 tasks' full token/mask multisets also match across all runs. Raw stored logprobs still vary on some all-zero-reward groups discarded for no signal, up to an observed 0.045; this is not universal logprob determinism. Timing includes startup, actual rollout work, training, weight refresh, full checkpoint I/O and shutdown. Post-run checkpoint hashing/cleanup is excluded. One-second log intervals from inference launch to startup policy broadcast have medians 27 s serial and 21 s early; final checkpoint I/O takes 5-8 s. This locates the startup saving while explaining why total runtime cannot halve.

The repaired confirmation gate was frozen after the diagnostic c1/e1 pair and before four additional runs per arm. It requires exact finalized work and trainer metrics, matching common-request outputs, at most four differing shutdown requests, generated-token work within 5% of c1, and clean resource/error checks. Initial c1/e1 are included in the five-per-arm statistics. Actual shutdown-tail work spans 172-175 requests and 5,047-5,158 generated tokens, about 2.2% between extremes; these tails are timed, not subtracted. Each checkpoint is about 4.53 GB with sizes/SHA256 recorded; the final e5 checkpoint remains local. All guards pass, zero job swap, peak GPU 7,228 MiB, minimum host availability 4.37 GB.

The invariant-only precursor completed three runs but failed its original exact-request/comparable-work gate at e2 (159 requests/148 finalized rollouts versus 174/166). It remains separate and is not pooled with repaired results. Its failed gate is retained. The later repair resolves a different, reproduced mixed-route cause, and the revised gate allows bounded shutdown tails while tightening finalized work and trainer metrics to exact equality.

This confirms overlapping startup on this opt-in two-update profile. It does not isolate a kernel speedup, establish a gain over the unchanged default preset, or establish long-run learning throughput. The packaged CLI resolves the batch-invariant overlay plus --deployment.early-trainer-start. Evidence: `prime-kernels/research/smollm2_360m_decode/experiments/b2-reproducibility-r1/`.

The probe uses 16 real initial-policy prompt/seed requests. BF16-head exploratory runs use probe-bf16-head.py; the corrected probe uses an explicit FP32 head. decode_backend.before.py preserves the pre-change implementation. Source hashes, all failed precursor records, compressed training traces, guard records and checkpoint manifests are retained. Compiled libraries, model weights and checkpoint payloads are not included in the research archive. The installed vLLM is pinned by the existing laptop environment; no venv file was edited.
