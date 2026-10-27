#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  block360.cu -o build/libdecode360_r16.so \
  >build/nvcc.stdout 2>build/ptxas.log
