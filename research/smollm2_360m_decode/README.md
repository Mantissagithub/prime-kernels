# SmolLM2-360M full-decoder research snapshot

This directory archives the exact R9 source and include used by the recorded
full-decoder checks. The kernel performs one single-launch decode step across
all 32 layers, including embedding lookup, paged KV reads/writes, and final
RMSNorm. Its 34 fixture checks are single-step numerical validation only; they
do not establish multi-step serving behavior.

## R9 change

R9 keeps R8's CUDA source and updates the PV dispatch in the include. For B1
medium contexts K=17..32, it uses y2/chunk16 with two round-robin FP32
accumulators per lane. B1 K<=16 retains WMMA, and K>32 retains y4/chunk32
with one accumulator. B2 retains R8's PV paths. Products use separate RN
multiply/add operations, accumulator and y partials fold serially, and the
result is rounded once to BF16. Other decoder arithmetic is unchanged.

## Validation

All four initial B1/B2 context-8/32 fixtures pass bitwise. The frozen 30-case
held-out suite (seeds 17/101/2027, batches 1/2, contexts 1/8/16/32/128) also
passes all cases under the original gates. Across all 34 single-step cases,
recorded per-layer block outputs and KV, final hidden state, and final logits
are bitwise equal to native (0 BF16 ULP), and greedy IDs match exactly.

This is a numerical result for one full 32-layer decode step at those fixture
shapes. It is not multi-step, long-context, serving, speed, reward, or quality
validation. The decoder remains research code and is not marked accepted for
production.

R8 is retained in `validation-summary.json`: it passed the four initial
fixtures and 29/30 held-out cases, with one failure at `seed17-b1-ctx16`.
R7 passed 24/30, R6 passed 22/30, and R4 passed 20/30; their compact history
and hashes remain in the summary.

## Wider-context and wrapper checks

A separate seed-777 R9 set passes 5/8 medium-context cases (31/63/255/511)
and 0/4 long-context cases (2048/8000). The three medium and four long
failures remain under the unchanged numerical gates even though all greedy
IDs match. At the medium failures, Q/QK/scores/probabilities are exact and a
single PV output differs by one BF16 ULP; the native softmax recompute's
singleton-query shape comparison is not a probability mismatch. These results
mean the original 34-case pass does not generalize to the wider tested shapes.
Evidence and guard data are in the `wide_context_r9` entry of
`validation-summary.json` and the linked Prime-RL run outputs.

The public package wrapper also passes the four original B1/B2 context-8/32
fixtures for int32 and int64 IDs, with bitwise-equal hidden states, logits,
all layer KV, and int64 CUDA Graph replay. This is single-step wrapper
validation; end-to-end serving remains unverified. No speed or quality claim
is made.

## Build

With CUDA 13.0 and SM89 support, build the shared library with:

```sh
mkdir -p build
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  block360.cu -o build/libdecode360_r9.so
```

The archived R9 source/include/library hashes and validation provenance are in
`validation-summary.json` and `prime-rl/outputs/megakernel-20261025/decode-360-cuda-r9/provenance.json`.

## Evidence

- Initial four: `prime-rl/outputs/megakernel-20261025/decode-360-validation-r10/results-r4-r9/comparison.json` and `decode-360-validation-r10/guard-r9/result.json`.
- Full held-out 30: `prime-rl/outputs/megakernel-20261025/block-360-heldout-r16/results-r1/comparison.json` and `block-360-heldout-r16/guard/result.json`.

The numerical gates are unchanged: per-layer output/KV <=1 ULP, final hidden
<=1 ULP, final logits <=2 ULP, and exact greedy IDs. No speed, reward, quality,
multi-step, long-context, or production claim follows from these checks.
