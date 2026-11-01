# R35 selective B2 packed inputs

R35 starts from R30's shared FP32 vector footprint and packs only B2 inputs to QKV and O. A separate, explicitly BF16 shared array receives B2 norm1's and attention's already BF16-rounded outputs; the existing barriers separate each producer from its consumer. B1 stays on the R30 FP32 path. Norm2, gate/up, down, and final norm also keep the R30 FP32 path. The R34 generic norm and warp GEMV helpers handle either vector type without changing workspace sizes or decode arguments. The extra BF16 allocation is 3,840 bytes for B2; the statically sized shared arrays total 46,896 bytes before compiler padding, below 48 KiB.

R1's source, build script, DSO, compiler logs, hash, and result are preserved under `revision-r1/`. Build the revised source with `bash candidate/build.sh`; the default library goes into `r35build-r2/`. `bench_r30_r35_r2.py` compares R35 against R30 on four frozen short-context cases and seeded synthetic 1024-token cases. Its pre-timing gate requires exact final-hidden and all-layer KV parity. `run_case_cap3g.sh <unit> <guard-dir> <case-id>` runs one case under the existing 3072 MiB GPU memory guard.

R1 exploratory timing is recorded in `revision-r1/results/all-six-r1.json`; R2's union overlay has not been rebuilt or benchmarked yet. The selective projection hypothesis has no verified gain so far.
