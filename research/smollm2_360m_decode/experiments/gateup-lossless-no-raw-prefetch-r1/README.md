# Lossless gate/up: remove raw range prefetch setup

This trial compares the frozen compressed kernel with a candidate that removes
only the three CSR-dependent raw escape range hint setup lines. The original
kernel and its `run_gateup_lossless` ABI remain in the source unchanged. Both
arms keep the CSR offsets and raw escape allocation; the candidate retains the
plane and header L2 hints and identical decode/math/row mapping.

`source_deletion_proof.py` anchors the control source at SHA-256
`b0218ea40e2f9012c802dde20adfb428d0361ad952761c07a698a7e60e7cf09c` and proves
the candidate kernel is exactly the control kernel renamed with that one
three-line deletion. Run it with:

```bash
uv run python source_deletion_proof.py
```

The harness uses the cached SmolLM2-360M checkpoint. Dense weights are retained
for exact decoded-word and output verification. Timed arms are the compressed
control and no-raw-prefetch candidate, in one session, alternating five
placements with five graph replays per placement. It checks NaN-poisoned output
writes and bit-exact FP32/BF16 outputs for dense, compressed control, and
candidate. CSR offsets and raw escapes remain allocated and are included in the
encoded-byte count. The scope is the raw projection microbenchmark, excluding
SwiGLU and surrounding MLP work.

Build without loading weights or touching a GPU:

```bash
uv run python harness.py --build-only
```

Run after the library exists:

```bash
uv run python harness.py --skip-build --output results-r1.json
```

The harness enforces the 3 GiB peak allocated GPU-memory limit and records
source, harness, shared-library, and model hashes. It does not save weights.
