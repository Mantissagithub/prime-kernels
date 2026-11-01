#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OUT="${R36_BUILD_DIR:-/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r36-gateup-cta-map/r36build}"
mkdir -p "$OUT"
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  block360.cu -o "$OUT/libdecode360_r36_gateup_cta_map.so" \
  >"$OUT/nvcc.stdout" 2>"$OUT/ptxas.log"
sha256sum "$OUT/libdecode360_r36_gateup_cta_map.so" > "$OUT/library.sha256"
