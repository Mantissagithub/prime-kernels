# Experiment 226: fixed-work vLLM serving

Five alternating fresh processes per arm, five measured repeats per case after two warmups. Same 360M policy snapshot, seeded token-ID prompts, greedy 64-token output, graph configuration and 256 KV blocks. Full batch wall time includes prefill, LM head, scheduler and output handling; startup and untimed logprob parity are excluded. This is not an RL wall-time comparison.

| Context | Batch | R30 median [range], ms | Lossless median [range], ms | Change | Verdict |
|---:|---:|---|---|---:|---|
|32|1|217.805 [217.375, 219.200]|219.113 [217.789, 219.661]|+0.60%|inconclusive_overlapping_process_ranges|
|32|2|234.027 [233.376, 235.936]|232.031 [231.707, 234.363]|-0.85%|inconclusive_overlapping_process_ranges|
|128|1|220.493 [220.295, 221.706]|221.231 [220.232, 222.078]|+0.33%|inconclusive_overlapping_process_ranges|
|128|2|240.531 [240.226, 240.622]|237.774 [237.572, 238.012]|-1.15%|candidate_lower_all_process_medians|
|1024|1|263.371 [262.890, 263.398]|263.354 [263.142, 263.567]|-0.01%|inconclusive_overlapping_process_ranges|
|1024|2|321.758 [321.231, 322.203]|319.934 [319.387, 320.140]|-0.57%|candidate_lower_all_process_medians|

Ranges span the five process medians, not individual repeats. Every prompt and output token ID matches across all processes and timed repeats, with 2,880 timed generated tokens per process. Maximum sampled-token logprob difference is zero. B2 contexts 128/1024 have nonoverlapping process ranges; other cases remain inconclusive. No default promotion or verified RL curve point.

All 20 inner/outer guard records pass, zero job swap, sampled GPU peak 1,405 MiB, minimum available host memory 8,577,482,752 bytes. The interrupted A1 invocation has partial guard records but no completion JSON or output; it is excluded and preserved locally. Completed sequence: A2,B1,A3,B2,A4,B3,A5,B4,A6,B5.

Baseline private R30 preload comes from `outputs/megakernel-20261025/r30-full-rl-r1/python/sitecustomize.py`, validated by pinned source/include/library hashes. The candidate is actual packaged fixed lossless code from prime-kernels f891acb. Libraries were prebuilt; no weights or compiled libraries are archived. Current host scripts retain original workspace paths. `manifest.json` hashes originals and deterministic gzip payloads.
