#!/usr/bin/env bash
# Builds the R28 decoder library (sm_89). Outputs go to the work dir, not the research dir.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${R28_BUILD_DIR:-/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r28-flash-decode/build}"
mkdir -p "$OUT"
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  block360.cu -o "$OUT/libdecode360_r28_flash.so" \
  >"$OUT/nvcc.stdout" 2>"$OUT/ptxas.log"
sha256sum "$OUT/libdecode360_r28_flash.so" > "$OUT/library.sha256"
