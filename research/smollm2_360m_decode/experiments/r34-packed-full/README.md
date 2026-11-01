# R34 packed B2 shared activations

R34 keeps the R30 kernel and stores B2's shared projection input as BF16. B1 keeps R30's FP32 shared representation. The candidate changes no public decode arguments or workspace sizes.

Each shared-vector producer already crosses a BF16 boundary. RMS norm and merged attention round before storing, and SwiGLU writes BF16 to workspace. R34 writes those same values directly to BF16 shared memory. The projection loads expand them to their exact FP32 values before the existing FMA order. The down projection copies BF16 workspace values directly, with no extra conversion pass.

The source preflight writes `source-preflight-r1.json`. It checks the source regions that must stay fixed and records the source diffs. It does not compile CUDA or access a GPU.

Build the candidate with `bash candidate/build.sh`. The default output is the separate `r34build/` directory. Set `R34_BUILD_DIR` to override it.

After the root compiles the candidate, `bench_r30_r34.py` compares it with R30 on the four frozen short-context cases and seeded synthetic 1024-token cases. The benchmark checks exact final-hidden and KV parity outside the timed region. `run_case_cap3g.sh <unit> <guard-dir> <case-id>` runs one case under the existing 3072 MiB GPU memory guard.

This package contains no post-change timing result. Source parity does not establish a speedup.
