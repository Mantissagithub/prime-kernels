from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'outputs/megakernel-20261025/lossless-full-kernel-integration-r1'
PACK = ROOT / 'prime-kernels/prime_kernels/decode_megakernel_360m_lossless/csrc'
OUT = HERE / 'early-staged'
OUT.mkdir(exist_ok=False)
cu = (PACK / 'block360.cu').read_text().replace('#include "../../lossless_bf16/csrc/encoded_gu.h"', '#include "' + str(PACK.parents[1] / 'lossless_bf16/csrc/encoded_gu.h') + '"')
inc = (PACK / 'block360_kernel.inc').read_text()
start = inc.index('    __shared__ __align__(8) uint2 csr_ranges')
loop = inc.index('    for (int r = gw; r < kI; r += kGlobalWarps) {', start)
setup = inc[start:loop]
declaration, setup = setup.split('\n', 1)
setup = setup.replace('    __syncthreads();', '    __syncwarp();')
late_pointers = '''    const uint8_t* plane = encoded_layer->plane;
    const uint32_t* headers = encoded_layer->headers;
    const uint16_t* raw = encoded_layer->raw;
'''
inc = inc[:start] + late_pointers + inc[loop:]
anchor = '  if (!encoded_active) {\n'
assert inc.count(anchor) == 1
inc = inc.replace(anchor, declaration + '\n  if (encoded_active) {\n' + setup + '  } else {\n', 1)
(OUT / 'block360.cu').write_text(cu)
(OUT / 'block360_kernel.inc').write_text(inc)
bench = (OLD / 'validate_r30_parity.py').read_text().replace('HERE = Path(__file__).resolve().parent', 'HERE = Path(' + repr(str(OLD)) + ').resolve()', 1)
marker = 'A, B = "A-R30", "B-Lossless"'
bench = bench.replace(marker, 'ARMS["B-Lossless"] = {"source": Path(' + repr(str(OUT / 'block360.cu')) + '), "include": Path(' + repr(str(OUT / 'block360_kernel.inc')) + '), "library": Path(' + repr(str(OUT / 'candidate.so')) + ')}\n' + marker, 1)
(HERE / 'early_staged_bench.py').write_text(bench)
