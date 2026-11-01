# Lossless gate/up projection probe

This harness compares the frozen dense gate/up projection with its losslessly
encoded representation on all 32 SmolLM2-360M layers. Each layer combines its
BF16 gate and up matrices into `[5120, 960]`. It measures the raw projection
microkernel only. It does not include SwiGLU or the rest of the MLP.

The harness reads the cached `model.safetensors` checkpoint. It encodes each
layer on CPU with OMP and MKL limited to one thread, then transfers one
concatenated tensor for each of the dense weights, byte planes, headers, raw
escapes, and CSR row offsets. It saves no weights. The report counts the row
offset array as encoded metadata and includes hashes for the model, harness,
CUDA source, and compiled library.

Build without importing PyTorch or touching the GPU:

```bash
uv run python harness.py --build-only
```

Build with ptxas resource output captured under `build/ptxas.log`:

```bash
uv run python harness.py --build-only --ptxas-verbose
```

Run the comparison after the shared library exists:

```bash
uv run python harness.py --skip-build
```

Or compile and run in one invocation:

```bash
uv run python harness.py
```

The run GPU-decodes all encoded weights and compares every BF16 word with the
dense copy. It captures dense and lossless graphs over the 32 distinct layer
weights, poisons outputs with NaNs after capture, and verifies finite,
bit-identical FP32 and BF16 projection results after replay. Timing alternates
five placements per arm, with five graph replays per placement. Results default
to `results.json`; `--output PATH` selects a new path. Existing libraries,
ptxas logs, and result files are never overwritten.

The harness checks PyTorch's peak allocated GPU memory against a strict 3 GiB
limit. It requires the CUDA 13.0 compiler by default, an sm_89 GPU, and the
cached checkpoint at the path embedded in `harness.py`.
