import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'outputs/megakernel-20261025/lossless-full-kernel-integration-r1'
PACK = ROOT / 'prime-kernels/prime_kernels/decode_megakernel_360m_lossless/csrc'


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


manifest = {}
for name, source in [('r30', ROOT / 'outputs/megakernel-20261025/r30-phase-prefetch/candidate'), ('lossless', PACK)]:
    out = HERE / name
    out.mkdir(exist_ok=True)
    cu = (source / 'block360.cu').read_text()
    inc = (source / 'block360_kernel.inc').read_text()
    manifest[name] = {f: hashlib.sha256((source / f).read_bytes()).hexdigest() for f in ('block360.cu', 'block360_kernel.inc')}
    cu = cu.replace('#include "../../lossless_bf16/csrc/encoded_gu.h"', '#include "' + str(PACK.parents[1] / 'lossless_bf16/csrc/encoded_gu.h') + '"')
    cu = replace_once(cu, 'using bf16 = __nv_bfloat16;', 'using bf16 = __nv_bfloat16;\n__device__ unsigned long long* gu_profile = nullptr;')
    cu += '''
extern "C" int set_gu_profile(void* pointer) {
  auto* p = static_cast<unsigned long long*>(pointer);
  return static_cast<int>(cudaMemcpyToSymbol(gu_profile, &p, sizeof(p)));
}
'''
    begin = '  const EncodedGULayer* encoded_layer' if name == 'lossless' else '  for (int k = 0; k < (kI + kGlobalWarps - 1) / kGlobalWarps; ++k) {'
    inc = replace_once(inc, begin, '  unsigned long long gu_start = t == 0 ? clock64() : 0;\n' + begin)
    norm = '  for (int b = 0; b < B; ++b)\n    rms_norm_to_smem(ws + kOffH1'
    inc = replace_once(inc, norm, '  if (t == 0) { gu_profile[(layer_index * 24 + blockIdx.x) * 3] = clock64() - gu_start; gu_start = clock64(); }\n' + norm)
    gate = '  // B2 uses the staged-CSR' if name == 'lossless' else '  // R27 FAST gate/up:'
    inc = replace_once(inc, gate, '  if (t == 0) { gu_profile[(layer_index * 24 + blockIdx.x) * 3 + 1] = clock64() - gu_start; gu_start = clock64(); }\n' + gate)
    end = '  for (int idx = t; idx < B * kI; idx += kThreads) vec_s[idx] = bf2f(ws[kOffAct + idx]);'
    inc = replace_once(inc, end, '  if (t == 0) gu_profile[(layer_index * 24 + blockIdx.x) * 3 + 2] = clock64() - gu_start;\n' + end)
    (out / 'block360.cu').write_text(cu)
    (out / 'block360_kernel.inc').write_text(inc)

bench = (OLD / 'validate_r30_parity.py').read_text()
bench = replace_once(bench, 'HERE = Path(__file__).resolve().parent', 'HERE = Path(' + repr(str(OLD)) + ').resolve()')
for name, label in [('r30', 'A-R30'), ('lossless', 'B-Lossless')]:
    marker = 'A, B = "A-R30", "B-Lossless"'
    location = HERE / name
    bench = replace_once(bench, marker, 'ARMS[' + repr(label) + '] = {"source": Path(' + repr(str(location / 'block360.cu')) + '), "include": Path(' + repr(str(location / 'block360_kernel.inc')) + '), "library": Path(' + repr(str(location / 'profile.so')) + ')}\n' + marker)
bench = replace_once(bench, '    from prime_kernels.lossless_bf16 import EncodedGUCache', '''    profile = {name: torch.zeros((32, 24, 3), dtype=torch.int64, device=device) for name in libs}
    for name, entry in libs.items():
        entry[0].set_gu_profile.argtypes = [ctypes.c_void_p]
        entry[0].set_gu_profile.restype = ctypes.c_int
        assert entry[0].set_gu_profile(profile[name].data_ptr()) == 0
    from prime_kernels.lossless_bf16 import EncodedGUCache''')
anchor = '"events_ms": elapsed, "median_ms_per_step": sorted(elapsed)[len(elapsed) // 2]})'
bench = replace_once(bench, anchor, anchor + '\n                    block_rows[-1]["profile_cycles"] = profile[name].cpu().tolist()')
(HERE / 'profile_bench.py').write_text(bench)
(HERE / 'profile-source-hashes.json').write_text(json.dumps(manifest, indent=2) + '\n')
