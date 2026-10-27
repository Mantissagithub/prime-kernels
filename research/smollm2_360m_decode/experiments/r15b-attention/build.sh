#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build-r15b
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  block360.cu -o build-r15b/libdecode360_r15b_qk_wmma.so \
  >build-r15b/nvcc.stdout 2>build-r15b/ptxas.log
