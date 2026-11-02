from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'outputs/megakernel-20261025/lossless-full-kernel-integration-r1'
PACK = ROOT / 'prime-kernels/prime_kernels/decode_megakernel_360m_lossless/csrc'
OUT = HERE / 'register-epilogue'
OUT.mkdir(exist_ok=False)
cu = (PACK / 'block360.cu').read_text().replace('#include "../../lossless_bf16/csrc/encoded_gu.h"', '#include "' + str(PACK.parents[1] / 'lossless_bf16/csrc/encoded_gu.h') + '"')
inc = (PACK / 'block360_kernel.inc').read_text()
start = inc.index('#pragma unroll\n      for (int i = 0; i < 2; ++i)', inc.index('  if (encoded_active) {'))
end = inc.index('  } else {\n    for (int r = gw; r < kI; r += kGlobalWarps)', start)
inc = inc[:start] + '''      float res[2][B];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int b = 0; b < B; ++b) {
          float sum = (acc[i][b][0] + acc[i][b][1]) + (acc[i][b][2] + acc[i][b][3]);
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
          res[i][b] = sum;
        }
      if (lane == 0) {
        for (int b = 0; b < B; ++b) {
          const float g = rnd(res[0][b]), u = rnd(res[1][b]);
          const float silu = rnd(g / (1.f + expf(-g)));
          if (p.debug) {
            debug_layer[kDbgGate + b * kI + r] = __float2bfloat16_rn(g);
            debug_layer[kDbgUp + b * kI + r] = __float2bfloat16_rn(u);
            debug_layer[kDbgSilu + b * kI + r] = __float2bfloat16_rn(silu);
          }
          ws[kOffAct + b * kI + r] = __float2bfloat16_rn(mul_rn(silu, u));
          if (p.debug) debug_layer[kDbgAct + b * kI + r] = ws[kOffAct + b * kI + r];
        }
      }
    }
''' + inc[end:]
(OUT / 'block360.cu').write_text(cu)
(OUT / 'block360_kernel.inc').write_text(inc)
bench = (OLD / 'validate_r30_parity.py').read_text().replace('HERE = Path(__file__).resolve().parent', 'HERE = Path(' + repr(str(OLD)) + ').resolve()', 1)
marker = 'A, B = "A-R30", "B-Lossless"'
bench = bench.replace(marker, 'ARMS["B-Lossless"] = {"source": Path(' + repr(str(OUT / 'block360.cu')) + '), "include": Path(' + repr(str(OUT / 'block360_kernel.inc')) + '), "library": Path(' + repr(str(OUT / 'candidate.so')) + ')}\n' + marker, 1)
(HERE / 'register_epilogue_bench.py').write_text(bench)
