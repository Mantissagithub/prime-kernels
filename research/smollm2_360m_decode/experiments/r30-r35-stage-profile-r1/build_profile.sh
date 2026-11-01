#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PAIR="${1:?usage: build_profile.sh r30|r35}"
case "$PAIR" in
  r30) SRC="$ROOT/r30-profile"; OUT="$ROOT/r30-build/libdecode360_r30_stage_profile.so" ;;
  r35) SRC="$ROOT/r35-profile"; OUT="$ROOT/r35-build/libdecode360_r35_stage_profile.so" ;;
  *) echo "pair must be r30 or r35" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$OUT")"
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  "$SRC/block360.cu" -o "$OUT" \
  >"$ROOT/build/${PAIR}-nvcc.stdout" 2>"$ROOT/build/${PAIR}-ptxas.log"
sha256sum "$OUT" >"$ROOT/build/${PAIR}-library.sha256"
