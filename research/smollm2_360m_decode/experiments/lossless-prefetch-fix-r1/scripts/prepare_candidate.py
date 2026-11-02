import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'outputs/megakernel-20261025/lossless-full-kernel-integration-r1'
PACK = ROOT / 'prime-kernels/prime_kernels/decode_megakernel_360m_lossless/csrc'
OUT = HERE / 'phase-ahead'
OUT.mkdir(exist_ok=False)
cu = (PACK / 'block360.cu').read_text()
cu = cu.replace('#include "../../lossless_bf16/csrc/encoded_gu.h"', '#include "' + str(PACK.parents[1] / 'lossless_bf16/csrc/encoded_gu.h') + '"')
inc = (PACK / 'block360_kernel.inc').read_text()
anchor = '  if (!encoded_active) {\n'
assert inc.count(anchor) == 1
early = '''  if (encoded_active) {
    for (int r = gw; r < kI; r += kGlobalWarps) {
      for (int matrix = 0; matrix < 2; ++matrix) {
        const int row = matrix * kI + r;
        prefetch_range_l2(encoded_layer->plane + size_t(row) * kH, kH);
        prefetch_range_l2(encoded_layer->headers + size_t(row) * (kH / 8), (kH / 8) * sizeof(uint32_t));
        const uint32_t first = __ldg(encoded_layer->row_offsets + row);
        const uint32_t last = __ldg(encoded_layer->row_offsets + row + 1);
        if (last > first) prefetch_range_l2(encoded_layer->raw + size_t(first) * 8, size_t(last - first) * 16);
      }
    }
  } else {
'''
inc = inc.replace(anchor, early, 1)
start = inc.index('    __shared__ __align__(8) uint2 csr_ranges')
end = inc.index('      float acc[2][B][4] = {};', start)
inc = inc[:start] + '''    const uint8_t* plane = encoded_layer->plane;
    const uint32_t* headers = encoded_layer->headers;
    const uint16_t* raw = encoded_layer->raw;
    for (int r = gw; r < kI; r += kGlobalWarps) {
''' + inc[end:]
inc = inc.replace('  // B2 uses the staged-CSR lossless decoder only when the published layer flag is enabled.',
                  '  // Encoded row hints were issued before the preceding grid barrier.')
(OUT / 'block360.cu').write_text(cu)
(OUT / 'block360_kernel.inc').write_text(inc)
shutil.copy2(OLD / 'build/prime_decode_megakernel_360m_lossless.so', HERE / 'frozen-lossless.so')
bench = (OLD / 'validate_r30_parity.py').read_text()
bench = bench.replace('HERE = Path(__file__).resolve().parent', 'HERE = Path(' + repr(str(OLD)) + ').resolve()', 1)
marker = 'A, B = "A-R30", "B-Lossless"'
bench = bench.replace(marker, 'ARMS["B-Lossless"] = {"source": Path(' + repr(str(OUT / 'block360.cu')) + '), "include": Path(' + repr(str(OUT / 'block360_kernel.inc')) + '), "library": Path(' + repr(str(OUT / 'candidate.so')) + ')}\n' + marker, 1)
(HERE / 'candidate_bench.py').write_text(bench)
(HERE / 'candidate-source-hashes.json').write_text(json.dumps({f: hashlib.sha256((PACK / f).read_bytes()).hexdigest() for f in ('block360.cu', 'block360_kernel.inc')}, indent=2) + '\n')
