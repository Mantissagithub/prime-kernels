from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE / 'bulk-cooperative'
OUT.mkdir(exist_ok=False)
shutil.copy2(HERE / 'register-epilogue/block360.cu', OUT / 'block360.cu')
inc = (HERE / 'register-epilogue/block360_kernel.inc').read_text()
anchor = '  if (!encoded_active) {\n'
assert inc.count(anchor) == 1
inc = inc.replace(anchor, '''  if (encoded_active) {
    const size_t first_line = size_t(blockIdx.x * kThreads + t) * 128;
    const size_t line_stride = size_t(kCTAs * kThreads) * 128;
    for (size_t offset = first_line; offset < size_t(2 * kI * kH); offset += line_stride)
      asm volatile("prefetch.global.L2 [%0];" : : "l"(encoded_layer->plane + offset));
    for (size_t offset = first_line; offset < size_t(2 * kI * (kH / 8)) * sizeof(uint32_t); offset += line_stride)
      asm volatile("prefetch.global.L2 [%0];" : : "l"(reinterpret_cast<const uint8_t*>(encoded_layer->headers) + offset));
    const size_t raw_bytes = size_t(__ldg(encoded_layer->row_offsets + 2 * kI)) * 16;
    for (size_t offset = first_line; offset < raw_bytes; offset += line_stride)
      asm volatile("prefetch.global.L2 [%0];" : : "l"(reinterpret_cast<const uint8_t*>(encoded_layer->raw) + offset));
  } else {
''', 1)
start = inc.index('    __shared__ __align__(8) uint2 csr_ranges')
end = inc.index('      float acc[2][B][4] = {};', start)
inc = inc[:start] + '''    const uint8_t* plane = encoded_layer->plane;
    const uint32_t* headers = encoded_layer->headers;
    const uint16_t* raw = encoded_layer->raw;
    for (int r = gw; r < kI; r += kGlobalWarps) {
''' + inc[end:]
inc = inc.replace('  // B2 uses the staged-CSR lossless decoder only when the published layer flag is enabled.',
                  '  // Encoded buffers were cooperatively prefetched before the preceding grid barrier.')
(OUT / 'block360_kernel.inc').write_text(inc)
(HERE / 'bulk_cooperative_bench.py').write_text((HERE / 'register_epilogue_bench.py').read_text().replace(str(HERE / 'register-epilogue'), str(OUT)))
