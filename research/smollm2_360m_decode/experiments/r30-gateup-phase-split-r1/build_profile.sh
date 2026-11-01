#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ "${1:-r30}" != r30 ]]; then echo "this profile clone is R30 only" >&2; exit 2; fi
SRC="$ROOT/r30-profile"
OUT="$ROOT/r30-build/libdecode360_r30_gateup_phase_split.so"
mkdir -p "$(dirname "$OUT")" "$ROOT/build"
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  "$SRC/block360.cu" -o "$OUT" \
  >"$ROOT/build/r30-nvcc.stdout" 2>"$ROOT/build/r30-ptxas.log"
sha256sum "$OUT" >"$ROOT/build/r30-library.sha256"
