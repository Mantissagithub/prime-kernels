#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
uv run --directory "$HERE/../../.." --offline --no-sync python "$HERE/preflight.py"
mkdir -p "$HERE/build"
timeout --signal=TERM --kill-after=5s 180s /usr/local/cuda-13.0/bin/nvcc \
  -std=c++17 -O3 -arch=sm_89 -lineinfo --ptxas-options=-v -Xcompiler=-fPIC --shared \
  "$HERE/src/block360.cu" -o "$HERE/build/libdecode360_r28_stage_profile.so" \
  >"$HERE/build/nvcc.stdout" 2>"$HERE/build/ptxas.log"
