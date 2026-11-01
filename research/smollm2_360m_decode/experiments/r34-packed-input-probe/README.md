# R34 packed activation input probe

This source-only probe compares the R30 projection's FP32 shared activation staging with BF16 shared staging. Each case reads the cached SmolLM2-360M BF16 projection weights for all 32 layers and uses deterministic synthetic BF16 activation rows. It does not run full RL or modify production code.

The data shapes are QKV `[1600,960]`, O `[960,960]`, paired gate/up `[2560,960]` rows in each of two contiguous G and U planes, and down `[960,2560]`. The paired gate/up case launches one warp task for each output-row pair, reading G row `r` and U row `r`, as R30 does. B is 1 or 2. Weights are streamed from the single cached `model.safetensors` file by layer and copied to GPU; no full model object is materialized. The four projection weight sets occupy 629,145,600 bytes across 32 layers. Synthetic activation values are generated directly as BF16.

The kernel uses 24 CTAs of 256 threads. Both arms use the R30 weight-vector loads, batch-specific output-row mapping, four FP32 FMA accumulators, FMA operand order, reduction order, and output layout. Only shared activation representation and its producer/consumer load differ. The FP32 arm expands BF16 input to FP32 while staging; the packed arm copies eight BF16 values as one `uint4` and expands them at the point of use. Each timed graph includes activation staging and invokes the projection once per each of 32 distinct layer weight sets.

Before timing each shape and batch, the harness captures both graphs, poisons both output buffers with NaN, then replays each graph and checks that every output element became finite. This verifies that graph replay writes every layer and output element before the harness accepts parity or records timings. It then compares every FP32 output value and every BF16 output bit. Timing alternates arms for ten placements, with five graph replays per placement. `results.json` retains all five event samples within every placement and each placement median. It also records timed and untimed projection call counts, checked output elements, graph-write verification, validation errors, and graph replays.

`results-invalid-empty-graph-r1.json` preserves the first run. Its graph-capture stream bug left the timed graphs empty, so its timings are invalid.

The report includes SHA-256 hashes for the checkpoint, CUDA source, harness, and loaded shared library. Use `python harness.py --build-only` to compile without loading the model or using a GPU. Add `--ptxas-verbose` to request and print ptxas register, spill, and shared-memory output. Use `--skip-build` to run an existing shared library without invoking nvcc. Build output is included in `results.json` when ptxas reporting is enabled.

The probe makes no throughput, cache-residency, shared-bank, or SASS claim before a later approved GPU run and inspection.

`build_argv_preflight.py` sends the exact compiler argv assembled by the harness to a Python stub. It does not invoke nvcc. The harness itself builds only when explicitly run; the build is not part of source preparation.
