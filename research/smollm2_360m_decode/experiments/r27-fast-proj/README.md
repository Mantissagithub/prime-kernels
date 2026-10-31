# R27 FAST projections (R25 + bandwidth-oriented GEMV)

R27 is R25 with every projection (qkv, o, gate_up, down) at B=1 and B=2 replaced by a warp GEMV
in the same cooperative kernel. Everything else is copied from R25: attention, RoPE, RMSNorm,
KV handling, barriers, debug stages, workspace and the 28-argument `decode360` ABI.

Sources: `block360.cu` (helpers) and `block360_kernel.inc` (phases). Diff against
`../r25-pv64/` is confined to the four projection phases and the helper block.
Build: `bash build.sh` (writes the library, `ptxas.log` and `library.sha256` to
`outputs/megakernel-20261025/r27-fast-proj/build/`, not into this directory).

## Design

- `warp_gemv_rows<K, B, NR>`: one warp computes NR rows of row-major BF16 `W` against B FP32
  activation vectors in shared memory. Each lane reads 16-byte (8 BF16) chunks with `__ldg`,
  keeps four independent FP32 `fmaf` accumulators per (row, request), reduces with a 32-lane xor
  tree, and applies each weight to all B requests (weights read once). The K loop is fully
  unrolled.
- Configs follow the FAST best entries at 192 warps (grid=24 in the diagnosis, `results.json`):
  R=1 row per warp with U=1 for qkv, o and down; for gate_up, NR=2 means gate row r and up row
  kI+r are computed by the same warp, so SwiGLU is formed in-warp and no gate scratch is staged.
- Row parallelism only, no split-K: rows are `gw + k * 192` for the 192 global warps, so no extra
  grid barrier is needed. Per warp: qkv 9 rows (8 or 9), o 5, gate_up 13-14 pairs, down 5.
- Rounding boundaries are unchanged from R25: projection outputs to BF16 (qkv, o, gate, up, silu,
  act, down), o-residual add in FP32 on the BF16-rounded projection, down residual into `out_hidden`.
  Debug stage writes are kept.
- Unchanged launch: 24 CTAs x 256 threads, `__launch_bounds__(256,1)`, dynamic smem layout
  (wmma buffers kept so attention smem and `dynamic_smem_bytes` are unchanged).

## Resources (ptxas -v, sm_89, `build.sh`)

| Kernel | Registers | Spills | Stack | Shared (static + dynamic) |
|---|---:|---:|---:|---:|
| R27 B=1 (`mk_kernel<1>`) | 152 | 0 | 8 B | 6,944 B |
| R27 B=2 (`mk_kernel<2>`) | 166 | 0 | 16 B | 13,344 B |
| R25 B=1 (reference) | 147 | 0 | 8 B | 6,944 B |
| R25 B=2 (reference) | 152 | 0 | 16 B | 13,344 B |

Residency: 256 threads x 166 registers = 42.5k of 64k per SM, so one CTA per SM stays feasible
for the 24-CTA cooperative launch. No spills.

## Numerics

R27 is not bitwise to native. The projections use FP32 FMA in a 4-accumulator order instead of
R25's serial `add.rn`/`mul.rn` order, so BF16 outputs can flip by one ULP. The gate is the
relaxed RL gate declared before measurement: R27's errors vs native must not exceed the native
B1-vs-B2 kernel-selection floor on the same prompts. Results and the per-case table are in
`outputs/megakernel-20261025/r27-fast-proj/README.md`.

## Caveats

- Not bit-exact; the R25 strict ULP gates fail for R27 by design and are reported, not relaxed.
- Attention, QK, softmax and PV are the R25 code paths; their numerics are unchanged.
- Projection timings from the standalone diagnosis were used as the design reference only.
