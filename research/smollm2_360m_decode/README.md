# SmolLM2-360M full-decoder research checkpoint

This folder contains the exact R4 CUDA source and include used by the recorded R4 validations. The candidate is a single cooperative launch for one decode step across all 32 SmolLM2-360M layers, including embedding lookup, paged KV writes/reads, and final RMSNorm. It remains research code; passing the initial fixtures does not amount to held-out acceptance, production readiness, multi-step decode, or vLLM integration.

## R4 attention PV schedule

R4 keeps the batch-one WMMA probability/value (PV) path. For batch two it uses a coalesced SIMT layout: each 64-thread group handles one 32-dimension task, with two 32-thread y partitions accumulating contiguous 16-token chunks. The two FP32 partials are added once and rounded to BF16. Tasks map across the fixed 24-CTA cooperative grid. Every CTA reaches both B2 CTA barriers, followed by the existing grid barrier; the launch still requires the full grid to satisfy cooperative residency checks. This is the tested R4 schedule, not a general reconstruction of native cuBLAS arithmetic.

The recorded PTXAS build reports 173 registers/thread and no spills for B1; B2 uses 168 registers/thread, 12,864 bytes static shared memory, one hardware barrier, and no spills. The build is standalone and uses no fast-math flag.

## Numerical validation

The four initial B1/B2, context-8/32 fixtures pass all original gates under R4. Every recorded per-layer block output and KV tensor, final hidden state, and final logits are bitwise equal to native (0 BF16 ULP); greedy IDs match exactly.

The wider 30-case held-out run passes 20 and fails 10. Greedy IDs match in all 30, but the ten numerical failures still reject the full candidate under the unchanged gates. The compact failure list and first divergent post-block / following KV layers are in `validation-summary.json`.

R9 completed a guarded diagnostic of those ten failures (20.49 s, 1,359 MiB peak GPU, zero swap, guard return code 0). At each R8 first-post-block divergence layer, the first captured differing stage is `attn_o_input` in eight cases and `norm2` in two. Layer indices in the result files are zero-based. In two attention cases, the earliest captured difference occurs several layers before the first post-block divergence; an earlier difference may be attenuated or erased and is not enough to assign cause. R10 then captured rotated Q, QK product, scaled scores, softmax probabilities, and PV output at the R9-selected layers, using R5 diagnostic instrumentation (guard 0, 8.10 s, 1,359 MiB peak GPU, zero swap). Five cases first differ at PV output, three at QK product, and the same two cases differ at norm2. The R9-to-R10 final norm, logits, greedy IDs, and per-layer comparisons are identical across the ten cases, so the instrumentation did not change those outcomes. These observations localize captured differences, but do not establish the full upstream cause or a production fix. R10 is supplemental diagnosis only; R4 acceptance remains 4/4 initial and 20/30 held out. Per-case layers and R10 provenance are in `validation-summary.json`.

Next diagnostic: compare native and candidate QK, PV, and reduction arithmetic directly for the three QK-first and five PV-first cases, retaining the two norm2 cases separately. R10 used R5 instrumentation and is not a production candidate validation.

No speed, reward, quality, or production-acceptance claim follows from these numerical checks. The R4 guard results, source/include/library hashes, exact case IDs, and earlier R1 result are recorded in `validation-summary.json`.

## Optional debug buffer

With a non-null debug pointer, the source can capture 12 stage tensors for all 32 layers: norm1, QKV, attention output, O projection, post-attention residual, norm2, gate, up, SiLU, SwiGLU/down input, down projection, and post-block output. At layer 15 it also captures Q, attention scores, probabilities, and PV output. `decode360_debug_bytes()` reports **3,366,400 bytes** (1,683,200 BF16 elements) for the maximum B=2 layout. The debug buffer is optional; no-debug calls use the normal output path.

## Standalone build

From this folder, build the library with CUDA 13.0 and SM89 support:

```sh
mkdir -p build
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  block360.cu -o build/libdecode360_r4.so
```

This omits `-use_fast_math`. The archived `.so` was built from these exact source files; its hash is in `validation-summary.json`. Rebuilding is not part of this checkpoint update.

## Evidence

- Initial four fixtures: `prime-rl/outputs/megakernel-20261025/decode-360-validation-r8/results-r1/` and `decode-360-validation-r8/guard/result.json`.
- Thirty held-out fixtures: `prime-rl/outputs/megakernel-20261025/block-360-heldout-r8/results-r1/` and `block-360-heldout-r8/guard/result.json`.
- Source, compile, PTXAS, and shared-library provenance: `prime-rl/outputs/megakernel-20261025/decode-360-cuda-r4/provenance.json`.
- The earlier R1 three-pass / one-fail result is retained as historical context in `validation-summary.json`.
