from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE / 'paired-loads'
OUT.mkdir(exist_ok=False)
shutil.copy2(HERE / 'early-register/block360.cu', OUT / 'block360.cu')
inc = (HERE / 'early-register/block360_kernel.inc').read_text()
start = inc.index('__device__ __forceinline__ uint4 decode_chunk_predicated(')
end = inc.index('__device__ __forceinline__ void prefetch_range_l2', start)
helper = inc[start:end]
body_start = helper.index('  const uint32_t h = __ldg(headers + chunk);')
split = helper.index('  const uint2 p = make_uint2(plane0, plane1);')
decode_start = helper.index('  const uint32_t base_packed', split)
load = helper[body_start:split].replace('  const uint32_t h = __ldg(headers + chunk);\n', '', 1)
replacement = '''struct EncodedChunkRegisters {
  uint32_t header;
  uint2 plane;
  uint4 raw;
};

__device__ __forceinline__ EncodedChunkRegisters load_encoded_chunk(
    const uint8_t* plane, uint32_t h, const uint16_t* raw, int chunk) {
''' + load + '''  return {h, make_uint2(plane0, plane1), make_uint4(raw0, raw1, raw2, raw3)};
}

__device__ __forceinline__ uint4 reconstruct_encoded_chunk(const EncodedChunkRegisters& v) {
  const uint32_t h = v.header, base = h >> 24;
  const uint2 p = v.plane;
  const uint32_t raw0 = v.raw.x, raw1 = v.raw.y, raw2 = v.raw.z, raw3 = v.raw.w;
''' + helper[decode_start:]
inc = inc[:start] + replacement + inc[end:]
first = inc.index('          const uint4 wg = decode_chunk_predicated(')
last = inc.index('          const uint32_t ug[4]', first)
inc = inc[:first] + '''          const uint32_t hg = __ldg(headers + size_t(r) * (kH / 8) + c);
          const uint32_t hu = __ldg(headers + size_t(kI + r) * (kH / 8) + c);
          const auto vg = load_encoded_chunk(plane + size_t(r) * kH, hg, raw, c);
          const auto vu = load_encoded_chunk(plane + size_t(kI + r) * kH, hu, raw, c);
          const uint4 wg = reconstruct_encoded_chunk(vg);
          const uint4 wu = reconstruct_encoded_chunk(vu);
''' + inc[last:]
(OUT / 'block360_kernel.inc').write_text(inc)
(HERE / 'paired_loads_bench.py').write_text((HERE / 'early_register_bench.py').read_text().replace(str(HERE / 'early-register'), str(OUT)))
