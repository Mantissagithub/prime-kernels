# R28 flash-decode (SmolLM2-360M, SM89): R27 with split-KV attention

R28 is R27 with the attention phase replaced by a FlashDecoding-style split-KV pass. The design is the
R6 decode megakernel's attention (`prime-kernels/prime_kernels/decode_megakernel/csrc/megakernel.cu`,
MK_ATTN_V2 + MK_ADAPT_NC), adapted to the 360M shapes: 15 query heads, 5 KV heads (GQA group 3),
head_dim 64, block size 16, B in {1, 2}, contexts up to 8192.

Sources: `block360.cu` (helpers, constants, workspace layout) and `block360_kernel.inc` (phases, kernel,
host ABI). Build: `bash build.sh`, which writes `libdecode360_r28_flash.so`, `ptxas.log` and
`library.sha256` to `outputs/megakernel-20261025/r28-flash-decode/build/`. Nothing is written into this
directory. The R27 directory is unchanged and was only read.

## What changed from R27

Unchanged: RMSNorm, the warp-GEMV projections (QKV, O, gate/up, down), RoPE, the CTA-0 K/V cache write,
the 24-CTA x 256-thread cooperative launch, the 28-argument `decode360` ABI and `decode360_status_string`,
and the paged FLASH_ATTN KV addressing (`block_table[b * bt_stride + pos / 16]`, `k_off`, `v_off`,
`block_stride`, `token_stride`, `head_stride`).

Replaced: the R27 QK, softmax and PV phases (their wmma staging, the BF16 score and probability scratch,
and the three grid barriers between them).

New attention, per layer:

1. Phase A (split-KV). One warp per item (request, kv head, chunk). Chunks per (request, kv head) are
   `NC = 192 / (5 B)` at most (38 at B=1, 19 at B=2), and adaptive: `min(NC, ceil(seq / 16))`, so each chunk
   has at least 16 positions. A lane group of 8 lanes reads one 128-byte K row (or V row) with 16-byte
   loads; 4 lane groups cover 4 positions per step, so a batch is 16 positions. The three query heads of
   the GQA group reuse each K/V load. Scores are reduced over the 8 lanes, rounded to BF16 as the native
   matmul output, then scaled by 0.125 (exact). The online softmax is FP32 with one rescale per 4-position
   step; the next batch's K/V loads are issued before the current batch's math. The item writes m, l and
   the unnormalized 3 x 64 P.V accumulator to FP32 scratch (`192 x 3 x 64` floats). Empty chunks write
   m = -inf, l = 0.
2. Grid barrier (replaces R27's QK, softmax and PV barriers, so each layer has 6 barriers instead of 8).
3. Phase B (merge, in every CTA). Stats are staged in shared memory, each (request, query head) computes
   M = max over its chunks and den = sum l_c exp(m_c - M), then each (request, query head, 4-dim slice) job
   computes sum_c exp(m_c - M) acc_c and writes `bf16(sum / den)` into the shared attention vector. The
   O projection then runs exactly as in R27. No barrier is needed between the merge and the O projection,
   because every CTA merges the same partials for its own copy of the vector.

K/V are read once per (request, kv head) per step, and each K/V row serves all three query heads of its
group. No score matrix is materialized.

Attention-internal debug stages (QK, scaled scores, probabilities) are no longer written. Their regions in
the debug buffer keep their R27 offsets and are left as the caller provided them. The Q, the attention
output (PV) and every per-layer stage are still written.

## Resources (ptxas -v, sm_89, `build.sh`, CUDA 13.0)

| Kernel | Registers | Spill stores / loads | Stack | Static smem | Dynamic smem | Threads x regs per CTA |
|---|---:|---:|---:|---:|---:|---:|
| R28 B=1 (`mk_kernel<1>`) | 249 | 0 / 0 | 0 B | 36,644 B | 0 | 256 x 249 = 63,744 |
| R28 B=2 (`mk_kernel<2>`) | 236 | 0 / 0 | 0 B | 43,056 B | 0 | 256 x 236 = 60,416 |
| R27 B=1 (reference) | 152 | 0 / 0 | 8 B | 6,944 B | 6,944 B | |
| R27 B=2 (reference) | 166 | 0 / 0 | 16 B | 13,344 B | | |

No spills. Residency: 63,744 registers at B=1 is under the 65,536-register SM file, so at most one CTA per
SM. The 24-CTA cooperative launch is accepted because `decode360`'s runtime check (`active * sm >= 24`,
from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) passed on the RTX 4060 Laptop GPU, and every guarded
launch in this session returned success. The workspace is 193,024 bytes (R27: 536,320 bytes).

## Numerics

Two checks were run. Both are relaxed-gate checks, not bit-exact ones.

1. Synthetic 32-layer check against an fp32 torch reference (`selfcheck_r28.py`, `compare_selfcheck.py`).
   Random weights (one set, shared by all layers), per-layer caches, shuffled physical block tables, and
   independent positions and lengths for B=2. 23 cases from 1 to 8192 tokens. R28 vs fp32 reference:
   mean of per-case max|err| 0.1035, max 0.164 (R27: 0.1037 and 0.164). R28 vs R27: max 0.141,
   mean of per-case mean|err| about 0.02. No NaN, no growth with context. Relative error is about 2 to 4%,
   which is bf16 output rounding on a scale of about 3.5.
2. Native HF BF16 fixtures (copies of the R27 validators, `validate_r28.py`, `validate_heldout30_r28.py`,
   `validate_wide12_r28.py`). Relaxed RL gate, as in R27: each case's max|dlogit| and mean|dlogprob| of the
   native argmax must not exceed the native B1-vs-B2 floor on the same prompt. Results are in
   `outputs/megakernel-20261025/r28-flash-decode/README.md`. Summary, 46 cases:
   - greedy agreement with native: 100% (R27: 98.9%);
   - mean max|dlogit| 0.206 (R27 0.192, floor 0.220); mean mean|dlogprob| 0.0345 (R27 0.0362, floor 0.0402);
   - per-case within floor: max|dlogit| 29/46 (R27 32/46), mean|dlogprob| 28/46 (R27 30/46).
   The strict ULP gates (R25 criterion) fail for R28 on every case, as they did for R27 (the R27 README
   explains why these gates are not applicable to the relaxed projections).

Caveat for attention: R27 was bitwise to native at layer 0 for B=1 context 8. R28 is not (layer-0
post-block max absolute error 0.016, about one bf16 ulp at magnitude 2 to 4), because the softmax and PV
accumulation order differ from the native kernels. That is the expected relaxed-parity change, not a bug
signature.

## Speed (direct decoder, A-B-B-A, R27 control vs R28)

3 blocks x 20 repeats per placement, 120 samples per arm per case, decoder step only (LM head excluded).
Medians in ms. The full table with IQRs is in the work directory README.

| Batch | Context | R27 | R28 | Speedup |
|---|---:|---:|---:|---:|
| 1 | 32 | 3.259 | 2.986 | 1.09x |
| 1 | 128 | 3.346 | 3.037 | 1.10x |
| 1 | 1024 | 5.772 | 3.338 | 1.73x |
| 1 | 4096 | 12.725 | 3.915 | 3.25x |
| 1 | 8000 | 22.408 | 4.598 | 4.87x |
| 2 | 32 | 3.258 | 3.155 | 1.03x |
| 2 | 128 | 3.493 | 3.242 | 1.08x |
| 2 | 1024 | 5.762 | 3.687 | 1.56x |
| 2 | 4096 | 10.591 | 4.742 | 2.23x |
| 2 | 8000 | 17.981 | 6.105 | 2.95x |

## Serving (vLLM, A-B-B-A vs stock)

Stock arms A1 and A2 bracket the two R28 arms (B1, B2), 7 repeats per shape, 64 greedy tokens. Pooled
median ITL in ms (the full table is in the work directory README):

| Prompt/batch | Stock | R28 | R28 vs stock | Both R28 arms < both stock |
|---|---:|---:|---:|:---:|
| 32/1 | 4.288 | 3.509 | -18.2% | yes |
| 32/2 | 3.993 | 3.683 | -7.7% | yes |
| 128/1 | 4.301 | 3.552 | -17.4% | yes |
| 128/2 | 3.999 | 3.741 | -6.5% | yes |
| 1024/1 | 4.502 | 3.816 | -15.2% | yes |
| 1024/2 | 4.407 | 4.154 | -5.7% | yes |

## Caveats

- Relaxed parity only. Strict ULP gates fail. Per-case relaxed-floor misses are concentrated at contexts of
  32 tokens or fewer, where R27 also misses. Only four cases above 1024 tokens were checked against the
  floor: max|dlogit| is above the floor in three (by 4 to 16%), and mean|dlogprob| in three. One of these
  is a clear outlier: wide-set B=1 context 2048 has mean|dlogprob| 0.057 against a floor of 0.026 and R27's
  0.013. The synthetic check at 2048 shows normal kernel error (relative 2.2%), so this is most likely
  single-prompt noise, but it should be re-run on more prompts at 2048 before the gate is called passed.
- The attention-internal debug stages (QK, scaled scores, probabilities) are no longer produced.
- Barrier count per layer: 6 (R27: 8). The KV-write barrier after CTA 0's cache write is kept, because the
  attention warps read the current token from the cache.
- Limits: B in {1, 2}, head_dim 64, 15/5 heads, block_size 16, max_seq_len 8192 (the same as R27).
- Harness memory: the direct and validator harnesses (native prefill plus per-arm KV copies) peak at
  4 to 6.5 GB of GPU memory at 8000 tokens. The kernel's own scratch is 193 KB. R27's wide-context
  validator run shows the same 4.3 GB peak.
