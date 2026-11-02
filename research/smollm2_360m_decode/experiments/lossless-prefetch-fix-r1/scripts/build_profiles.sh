#!/usr/bin/env bash
set -euo pipefail
cd /home/pradheep/tiny-prl/outputs/megakernel-20261025/lossless-prefetch-fix-r1
for variant in r30 lossless; do
  /usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo -shared -Xcompiler -fPIC -Xptxas=-v "$variant/block360.cu" -o "$variant/profile.so" > "$variant/build.log" 2>&1
done
