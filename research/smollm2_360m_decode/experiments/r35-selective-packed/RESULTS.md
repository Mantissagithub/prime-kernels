# Selective packing is not adopted

R35 R2 packs only B2 QKV/O inputs in a separate typed shared array. This avoids the earlier exploratory aliasing concern and preserves the FP32 gate/up and down paths. B2 uses 243 registers and 46,896 bytes of static shared memory, with zero spills. B1 remains 241 registers and 36,644 bytes.

Two fresh-process direct decoder comparisons report small apparent B2 latency reductions of 0.064–0.279%. They do not establish a packing speedup. The unchanged B1 path shows reductions of similar size, and B2/context 128 misses strict ordering in one repeated block. R35 is not adopted, and no serving or RL gain is claimed.

`audit-r2-r3/reproduce_sass.py` compares both encoded machine words of each SM89 instruction. B1 has exactly the same 7,360 instructions and 117,760 encoded bytes in both libraries, SHA256 `912d14f46ffe8b591a79f7fa26a2e13d22cbb516000c4015874b0c3343919858`. B2 differs. This identifies an unchanged-code timing control; it does not establish whole-library identity or explain the timing variation's exact cause.

All final hidden states and all 32 layers' K/V match R30 bitwise in all six cases of both measured processes. Both inner/outer guards pass, with 1,955 MiB sampled GPU peaks and zero sampled cgroup swap. Native numerical-floor acceptance remains unresolved for the baseline and is not established here.

The original measured work directory is `prime-rl/outputs/megakernel-20261025/r35-selective-packed`. Archived source and scripts retain its paths. The report contains raw pooled-event medians, ranges, work counts and limits. Raw JSON reports are deterministic gzip files; `raw-output-manifest.json` records hashes before and after compression. No model weights or compiled binaries are included. The next experiment profiles the integrated phases rather than inferring a decoder win from isolated GEMV timing.
