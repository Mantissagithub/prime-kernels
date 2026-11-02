from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE / 'parallel-plane'
OUT.mkdir(exist_ok=False)
shutil.copy2(HERE / 'shared-gemv/block360.cu', OUT / 'block360.cu')
inc = (HERE / 'shared-gemv/block360_kernel.inc').read_text()
start = inc.index('  const uint32_t h = __ldg(headers + chunk);')
end = inc.index('  const uint32_t base_packed', start)
inc = inc[:start] + '''  const uint2 p = __ldg(reinterpret_cast<const uint2*>(plane) + chunk);
  const uint32_t h = __ldg(headers + chunk);
  const uint32_t base = h >> 24;
  uint32_t raw0 = 0, raw1 = 0, raw2 = 0, raw3 = 0;
  asm volatile(
      "{ .reg .pred escape; .reg .b32 raw_index; .reg .b64 raw_address;\\n\\t"
      "setp.eq.u32 escape, %4, 255;\\n\\t"
      "selp.b32 raw_index, %5, 0, escape;\\n\\t"
      "mad.wide.u32 raw_address, raw_index, 16, %6;\\n\\t"
      "@escape ld.global.nc.v4.u32 {%0, %1, %2, %3}, [raw_address];\\n\\t"
      "}\\n\\t"
      : "+r"(raw0), "+r"(raw1), "+r"(raw2), "+r"(raw3)
      : "r"(base), "r"(h & 0xFFFFFFu), "l"(raw)
      : "memory");

''' + inc[end:]
(OUT / 'block360_kernel.inc').write_text(inc)
(HERE / 'parallel_plane_bench.py').write_text((HERE / 'shared_gemv_bench.py').read_text().replace(str(HERE / 'shared-gemv'), str(OUT)))
