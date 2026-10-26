# SmolLM2-360M decoder block research prototype

This folder contains an isolated research kernel for one SmolLM2-360M decoder
block. It is not registered, installed, wired into vLLM, or production code.

The kernel targets BF16 weights and activations with H=960, I=2560, 32 layers,
15 query heads, 5 KV heads, and head dimension 64. It supports batch sizes 1 and
2 on SM89 with one cooperative launch of 24 CTAs × 256 threads. The launch checks
the device's opt-in shared-memory limit and CTA residency before running.

The host ABI takes a six-pointer layer table (norm1, fused QKV, O, norm2, fused
gate/up, down), input hidden states, rotary cos/sin values, and a 16-token paged
K/V cache. The implementation follows the captured HF BF16 rounding boundaries
for RMSNorm, RoPE, eager attention, residuals, and SwiGLU. See the comments in
`block360.cu` and `block360_kernel.inc` for layouts and scratch sizes.

## Validation

R5 uses a 16-lane strided reduction for batch-one gate/up and down projections,
matching the observed native half-warp GEMV schedule. Q/K/V/O and batch-two WMMA
retain their existing paths. The native reference uses SDPA prefill and eager
decode. The four initial fixtures (B1/B2, contexts 8/32) and a frozen held-out set
of 30 cases (seeds 17/101/2027, B1/B2, contexts 1/8/16/32/128) all pass.

The strict gates remain block output <=1 BF16 ULP, full layer-0 K/V <=1 ULP,
final logits <=2 ULP, and exact greedy tokens. Block output, K/V, final norm and
hybrid logits are bitwise identical in all 30 held-out cases. One internal B2
attention value retains a 1-ULP difference also present before R5. The remaining
31 model layers, final norm and output head run natively for this validation.

R4 passed the initial four fixtures but failed three held-out B1 cases; those
three now pass with R5. These results validate a single custom block followed by
31 native layers. They do not validate a full custom decoder, multi-step decode,
vLLM integration, training or performance. No speedup claim is made.

## Build

Requires CUDA 13.0 and C++17. From this directory, the shared library build
command is:

```sh
nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo --ptxas-options=-v \
  -Xcompiler=-fPIC --shared block360.cu -o libblock360.so
```

There is no `--use_fast_math` flag. The recorded ptxas resource use is 80
registers/thread and 6,464 static shared bytes for B1, and 78 registers/thread
and 12,864 static shared bytes for B2; neither variant spills. The kernel also
requests 57,344 bytes dynamic shared memory, so runtime opt-in limits apply.
