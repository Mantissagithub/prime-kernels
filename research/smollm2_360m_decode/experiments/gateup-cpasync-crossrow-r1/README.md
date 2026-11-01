# Cross-row cp.async gate/up microprobe

This arm tests one scheduling change against the R30 B2 gate/up projection. The earlier row-local cp.async probe restarted its pipeline for every output row and waited for the first tile of each row. This candidate flattens each warp's row sequence and four K-tile iterations into one task stream. While it consumes a row's final K tile and reduces that row, it can copy the next row's first tile into the other buffer.

Each task `t` maps to row `gw + (t >> 2) * 192` and iteration `t & 3`. The kernel commits task zero, then commits task `t + 1` before consuming task `t`. It waits with `cp.async.wait_group 1` while a newer task remains and waits with `wait_group 0` for the final task. Iteration zero resets the four FP32 accumulators for that row. Iteration three reduces and stores the row before the next row's task is consumed. The warp synchronizes and applies a compiler memory fence before each shared-memory read. There are no CTA barriers inside the task stream.

The baseline `gateup_kernel` and shared helper prefix are byte-for-byte identical to `gateup-cpasync-probe-r1/projection_probe.cu`. The CPU preflight checks those source slices, task coverage for all 192 warp starts and all 2,560 rows, two-buffer pending-group transitions, and the final K tile's 24 valid lanes and eight zero-fill lanes. The baseline source hash is recorded in the harness report.

Both arms retain the same 24 CTAs, 256 threads, eight warps per CTA, R30 row mapping, full-row L2 prefetch policy, paired G/U loads, input expansion, FP32 FMA order, reduction order, and output layout. The candidate uses the same warp-private two-buffer, 16 KiB cp.async tile as the prior probe. The source keeps the four-tile loop bounds and uses shift/mask task decoding, so the compiler can lower division and remainder by four to bit operations.

The harness retains output poisoning and replay checks, requires exact FP32 and BF16-bit parity, records file hashes and work counts, and protects existing build and result outputs. It uses ten alternating placements, which gives five placements per arm, with five raw CUDA-event samples per placement. It is still a projection microprobe. It does not reproduce complete decoder scheduling or cache competition, and no speedup is implied by the hypothesis.

## Commands

From `prime-rl/`, run `UV_CACHE_DIR=/tmp/uv-cache uv run --no-project python outputs/megakernel-20261025/gateup-cpasync-crossrow-r1/source_preflight.py` for the CPU-only scheduling and source checks. Run `UV_CACHE_DIR=/tmp/uv-cache uv run --no-project python outputs/megakernel-20261025/gateup-cpasync-crossrow-r1/build_argv_preflight.py` to validate the compiler arguments with a stub that does not invoke nvcc.

`UV_CACHE_DIR=/tmp/uv-cache uv run --no-project python outputs/megakernel-20261025/gateup-cpasync-crossrow-r1/harness.py --build-only` builds without importing torch, loading weights, or using a GPU. Add `--ptxas-verbose` to retain ptxas diagnostics. Existing outputs are protected unless `--force` is supplied. `--skip-build` runs the GPU comparison using an existing shared library. Those CUDA operations are left for the serialized review and run.
