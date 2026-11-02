from pathlib import Path
import json
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE / 'sign-prmt'
OUT.mkdir(exist_ok=False)
shutil.copy2(HERE / 'early-register/block360.cu', OUT / 'block360.cu')
inc = (HERE / 'early-register/block360_kernel.inc').read_text()
start = inc.index('    const uint32_t bytes = __byte_perm(')
end = inc.index('    words[j] = sign_mant', start)
inc = inc[:start] + '''    uint32_t expanded;
    const uint32_t packed = j < 2 ? p.x : p.y;
    if (j % 2)
      asm("prmt.b32 %0, %1, 0, 0xb3a2;" : "=r"(expanded) : "r"(packed));
    else
      asm("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(expanded) : "r"(packed));
    const uint32_t sign_mant = expanded & 0x807f807fu;
''' + inc[end:]
(OUT / 'block360_kernel.inc').write_text(inc)
(HERE / 'sign_prmt_bench.py').write_text((HERE / 'early_register_bench.py').read_text().replace(str(HERE / 'early-register'), str(OUT)))

def prmt(value, selector):
    result = 0
    for i in range(4):
        choice = (selector >> (4 * i)) & 15
        byte = (value >> (8 * (choice & 7))) & 255
        if choice & 8:
            byte = 255 if byte & 128 else 0
        result |= byte << (8 * i)
    return result

for pair in range(65536):
    old = (pair & 127) | ((pair & 128) << 8) | ((pair & 0x7f00) << 8) | ((pair & 0x8000) << 16)
    assert prmt(pair, 0x9180) & 0x807f807f == old
    assert prmt(pair << 16, 0xb3a2) & 0x807f807f == old
(HERE / 'sign-prmt-cpu-proof.json').write_text(json.dumps({'pair_values': 65536, 'selectors': [0x9180, 0xb3a2], 'exact': True, 'ptx_spec': 'https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt'}, indent=2) + '\n')
