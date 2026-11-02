from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'outputs/megakernel-20261025/lossless-full-kernel-integration-r1'
PACK = ROOT / 'prime-kernels/prime_kernels/decode_megakernel_360m_lossless/csrc'
OUT = HERE / 'row-local'
OUT.mkdir(exist_ok=False)
cu = (PACK / 'block360.cu').read_text().replace('#include "../../lossless_bf16/csrc/encoded_gu.h"', '#include "' + str(PACK.parents[1] / 'lossless_bf16/csrc/encoded_gu.h') + '"')
inc = (PACK / 'block360_kernel.inc').read_text()
anchor = '  if (encoded_active) {\n    __shared__'
assert inc.count(anchor) == 1
inc = inc.replace(anchor, '  if (encoded_active) {\n    const int gw = blockIdx.x * kWarps + warp;\n    __shared__', 1)
rows = [b * 8 + w + j * 192 for b in range(24) for w in range(8) for j in range(14) if b * 8 + w + j * 192 < 2560]
assert sorted(rows) == list(range(2560))
(OUT / 'block360.cu').write_text(cu)
(OUT / 'block360_kernel.inc').write_text(inc)
bench = (OLD / 'validate_r30_parity.py').read_text().replace('HERE = Path(__file__).resolve().parent', 'HERE = Path(' + repr(str(OLD)) + ').resolve()', 1)
marker = 'A, B = "A-R30", "B-Lossless"'
bench = bench.replace(marker, 'ARMS["B-Lossless"] = {"source": Path(' + repr(str(OUT / 'block360.cu')) + '), "include": Path(' + repr(str(OUT / 'block360_kernel.inc')) + '), "library": Path(' + repr(str(OUT / 'candidate.so')) + ')}\n' + marker, 1)
(HERE / 'row_local_bench.py').write_text(bench)
