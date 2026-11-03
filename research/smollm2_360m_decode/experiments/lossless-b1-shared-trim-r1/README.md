# B1 shared scratch sizing and explicit L1 preference

## Experiment 240: smaller B1 shared scratch and L1 preference rejected

A private candidate sizes the vector, attention-merge and inverse-denominator shared arrays to template batch B instead of maximum batch 2. B1 static shared memory falls 38,440 to 23,576 bytes (-38.67%); registers/instructions/spills stay 253/8,720/0. B2 resource use and all instruction words, including control bits, remain byte-identical at 246 registers, 44,848 shared bytes, 9,048 instructions and no spills. Arithmetic, reduction order, weight format and launch dimensions are unchanged. Lower shared usage does not remove the B1 register limit, so no occupancy gain is assumed.

The first screen passes exact hidden/KV bits at both batches and contexts 32/128/1024 plus captured poisoned fallback/GU-only refresh. Across three alternating blocks with 120 events per placement, B1 median latency changes are -0.036/+0.272/+0.018%, with strict 0/3 at all contexts. Rejected for speed.

A second private control requests 25% shared-memory carveout once before timing, testing explicit L1 preference without a per-token setter call. Its GPU instruction words exactly match the shared-only candidate. This is a requested driver preference, not a measured physical cache allocation. Exact and graph checks pass again, but B1 changes -0.012/+0.243/-0.013%, strict 1/3,0/3,1/3. Rejected too. B2 timing shifts are a byte-identical placebo, not a gain.

Both builds hide CUDA and are bounded to one CPU, 2 GiB memory, zero job swap and 120 s. Both inner/outer GPU screens pass resource guards with zero job swap. No serving sweep, RL timing sweep or production change follows the failed first screens. The memory reduction is retained as research evidence only. Evidence: `prime-kernels/research/smollm2_360m_decode/experiments/lossless-b1-shared-trim-r1/`.

shared-only/ preserves the first candidate source/library identity; the root source adds only a host cache-preference setter. GPU words match between them. Libraries and build objects are local only and excluded from the archive.
