#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <stdint.h>

using bf16 = __nv_bfloat16;
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kCTAs = 24;
constexpr int kGlobalWarps = kCTAs * kWarps;
constexpr int kN = 2560;
constexpr int kK = 960;
constexpr int kChunks = kK / 8;
constexpr int kIters = (kChunks + 31) / 32;
constexpr int kB = 2;
constexpr int kDoubleBufferBytes = 2 * kWarps * 2 * 32 * 16;
static_assert(kDoubleBufferBytes == 16 * 1024, "cp.async weight buffers must use 16 KiB");

__device__ __forceinline__ float lo_bf16(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_bf16(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, int bytes) {
  const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(smem), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void prefetch_row_l2(const void* ptr, int bytes) {
  const char* p = reinterpret_cast<const char*>(ptr);
  for (int off = (threadIdx.x & 31) * 128; off < bytes; off += 32 * 128)
    asm volatile("prefetch.global.L2 [%0];" : : "l"(p + off));
}

__device__ __forceinline__ void prefetch_pair(const bf16* row_g, const bf16* row_u,
                                               uint4 (&tile)[2][2][32], int buffer,
                                               int lane, int iter) {
  const int chunk = lane + 32 * iter;
  const bool valid = chunk < kChunks;
  // For src_size=0, PTX still requires a valid global address operand.
  const int safe_chunk = valid ? chunk : (kChunks - 1);
  const int bytes = valid ? 16 : 0;
  cp_async_16(&tile[buffer][0][lane], row_g + safe_chunk * 8, bytes);
  cp_async_16(&tile[buffer][1][lane], row_u + safe_chunk * 8, bytes);
}

template <bool AsyncWeights>
__global__ void gateup_kernel(const bf16* W, const bf16* x, float* y) {
  __shared__ __align__(16) float xf[kB * kK];
  __shared__ __align__(16) uint4 weight_tiles[kWarps][2][2][32];
  for (int i = threadIdx.x; i < kB * kK; i += kThreads) xf[i] = __bfloat162float(x[i]);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gw = warp * kCTAs + blockIdx.x;
  // Match R30's L2 prefetch policy for every output row owned by this warp.
  for (int k = 0; k < (kN + kGlobalWarps - 1) / kGlobalWarps; ++k) {
    const int r = gw + k * kGlobalWarps;
    if (r < kN) {
      prefetch_row_l2(W + static_cast<size_t>(r) * kK, kK * sizeof(bf16));
      prefetch_row_l2(W + static_cast<size_t>(kN + r) * kK, kK * sizeof(bf16));
    }
  }
  for (int r = gw; r < kN; r += kGlobalWarps) {
    const bf16* row_g = W + static_cast<size_t>(r) * kK;
    const bf16* row_u = W + static_cast<size_t>(kN + r) * kK;
    float acc[2][2][4] = {};

    if constexpr (AsyncWeights) {
      prefetch_pair(row_g, row_u, weight_tiles[warp], 0, lane, 0);
      asm volatile("cp.async.commit_group;");
    }

#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      if constexpr (AsyncWeights) {
        const bool has_next = it + 1 < kIters;
        if (has_next) {
          prefetch_pair(row_g, row_u, weight_tiles[warp], (it + 1) & 1, lane, it + 1);
          asm volatile("cp.async.commit_group;");
          asm volatile("cp.async.wait_group 1;");
        } else {
          asm volatile("cp.async.wait_group 0;");
        }
        __syncwarp();
        asm volatile("" ::: "memory");
      }
      const int c = lane + 32 * it;
      if (c < kChunks) {
        uint4 wg, wu;
        if constexpr (AsyncWeights) {
          const int buffer = it & 1;
          wg = weight_tiles[warp][buffer][0][lane];
          wu = weight_tiles[warp][buffer][1][lane];
        } else {
          wg = __ldg(reinterpret_cast<const uint4*>(row_g) + c);
          wu = __ldg(reinterpret_cast<const uint4*>(row_u) + c);
        }
        const uint32_t ug[4] = {wg.x, wg.y, wg.z, wg.w};
        const uint32_t uu[4] = {wu.x, wu.y, wu.z, wu.w};
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          const float4 a = *reinterpret_cast<const float4*>(xf + b * kK + c * 8);
          const float4 d = *reinterpret_cast<const float4*>(xf + b * kK + c * 8 + 4);
          const float xv[8] = {a.x, a.y, a.z, a.w, d.x, d.y, d.z, d.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float wg_f = (e & 1) ? hi_bf16(ug[e >> 1]) : lo_bf16(ug[e >> 1]);
            const float wu_f = (e & 1) ? hi_bf16(uu[e >> 1]) : lo_bf16(uu[e >> 1]);
            acc[0][b][e & 3] = fmaf(wg_f, xv[e], acc[0][b][e & 3]);
            acc[1][b][e & 3] = fmaf(wu_f, xv[e], acc[1][b][e & 3]);
          }
        }
      }
    }

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int b = 0; b < kB; ++b) {
        float s = (acc[i][b][0] + acc[i][b][1]) + (acc[i][b][2] + acc[i][b][3]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) y[b * kN * 2 + i * kN + r] = s;
      }
  }
}

// Lossless stream layout: one byte per BF16 weight, one uint32 header per
// eight weights, and 16 raw bytes per escaped chunk. raw_offsets has one
// start index per row plus a final sentinel; it is both counted metadata and
// the range used by L2 prefetches. Header escape indices address raw globally.
__device__ __forceinline__ uint4 decode_chunk(const uint8_t* plane,
                                               const uint32_t* headers,
                                               const uint16_t* raw,
                                               int chunk) {
  const uint32_t h = __ldg(headers + chunk);
  const uint32_t base = h >> 24;
  uint16_t w[8];
  if (base == 255) {
    const uint4 packed = __ldg(reinterpret_cast<const uint4*>(raw + (h & 0xFFFFFFu) * 8));
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = uint16_t(words[i >> 1] >> ((i & 1) * 16));
  } else {
    const uint2 packed = __ldg(reinterpret_cast<const uint2*>(plane + chunk * 8));
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t delta = (h >> (3 * i)) & 7u;
      const uint32_t word = (i < 4) ? packed.x : packed.y;
      const uint32_t p = (word >> ((i & 3) * 8)) & 255u;
      w[i] = uint16_t(((p & 128u) << 8) | ((base + delta) << 7) | (p & 127u));
    }
  }
  return make_uint4(uint32_t(w[0]) | (uint32_t(w[1]) << 16),
                    uint32_t(w[2]) | (uint32_t(w[3]) << 16),
                    uint32_t(w[4]) | (uint32_t(w[5]) << 16),
                    uint32_t(w[6]) | (uint32_t(w[7]) << 16));
}

__device__ __forceinline__ uint4 decode_chunk_simd(const uint8_t* plane,
                                                    const uint32_t* headers,
                                                    const uint16_t* raw,
                                                    int chunk) {
  const uint32_t h = __ldg(headers + chunk);
  if ((h >> 24) == 255) {
    return __ldg(reinterpret_cast<const uint4*>(raw + (h & 0xFFFFFFu) * 8));
  }
  const uint2 p = __ldg(reinterpret_cast<const uint2*>(plane + chunk * 8));
  const uint32_t base_packed = (h >> 24) * 0x00010001u;
  uint32_t words[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t d = (h >> (6 * j)) & 63u;
    const uint32_t delta_packed = (d * 0x2001u) & 0x00070007u;
    const uint32_t bytes = __byte_perm(j < 2 ? p.x : p.y, 0,
                                       j % 2 ? 0x4342 : 0x4140);
    const uint32_t sign_mant = (bytes & 0x007f007fu) |
                               ((bytes & 0x00800080u) << 8);
    words[j] = sign_mant | ((base_packed + delta_packed) << 7);
  }
  return make_uint4(words[0], words[1], words[2], words[3]);
}

__device__ __forceinline__ uint4 decode_chunk_predicated(const uint8_t* plane,
                                                          const uint32_t* headers,
                                                          const uint16_t* raw,
                                                          int chunk) {
  const uint32_t h = __ldg(headers + chunk);
  const uint32_t base = h >> 24;
  uint32_t plane0 = 0, plane1 = 0;
  uint32_t raw0 = 0, raw1 = 0, raw2 = 0, raw3 = 0;
  const uint8_t* plane_address = plane + chunk * 8;
  asm volatile(
      "{ .reg .pred p; .reg .b32 raw_index; .reg .b64 raw_address;\n\t"
      "setp.eq.u32 p, %6, 255;\n\t"
      "selp.b32 raw_index, %7, 0, p;\n\t"
      "mad.wide.u32 raw_address, raw_index, 16, %8;\n\t"
      "@!p ld.global.nc.v2.u32 {%0, %1}, [%9];\n\t"
      "@p ld.global.nc.v4.u32 {%2, %3, %4, %5}, [raw_address];\n\t"
      "}\n\t"
      : "+r"(plane0), "+r"(plane1), "+r"(raw0), "+r"(raw1), "+r"(raw2), "+r"(raw3)
      : "r"(base), "r"(h & 0xFFFFFFu), "l"(raw), "l"(plane_address)
      : "memory");

  const uint2 p = make_uint2(plane0, plane1);
  const uint32_t base_packed = base * 0x00010001u;
  uint32_t words[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t d = (h >> (6 * j)) & 63u;
    const uint32_t delta_packed = (d * 0x2001u) & 0x00070007u;
    const uint32_t bytes = __byte_perm(j < 2 ? p.x : p.y, 0,
                                       j % 2 ? 0x4342 : 0x4140);
    const uint32_t sign_mant = (bytes & 0x007f007fu) |
                               ((bytes & 0x00800080u) << 8);
    words[j] = sign_mant | ((base_packed + delta_packed) << 7);
  }
  asm volatile(
      "{ .reg .pred p; .reg .b32 selected0, selected1, selected2, selected3;\n\t"
      "setp.eq.u32 p, %4, 255;\n\t"
      "selp.b32 selected0, %5, %0, p;\n\t"
      "selp.b32 selected1, %6, %1, p;\n\t"
      "selp.b32 selected2, %7, %2, p;\n\t"
      "selp.b32 selected3, %8, %3, p;\n\t"
      "mov.b32 %0, selected0;\n\t"
      "mov.b32 %1, selected1;\n\t"
      "mov.b32 %2, selected2;\n\t"
      "mov.b32 %3, selected3;\n\t"
      "}\n\t"
      : "+r"(words[0]), "+r"(words[1]), "+r"(words[2]), "+r"(words[3])
      : "r"(base), "r"(raw0), "r"(raw1), "r"(raw2), "r"(raw3)
      : "memory");
  return make_uint4(words[0], words[1], words[2], words[3]);
}

__global__ void decode_weights_kernel(const uint8_t* plane, const uint32_t* headers,
                                      const uint16_t* raw, uint16_t* output) {
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= 2 * kN * kChunks) return;
  const int row = chunk / kChunks;
  const int row_chunk = chunk - row * kChunks;
  const uint4 decoded = decode_chunk(plane + size_t(row) * kK,
                                     headers + size_t(row) * kChunks, raw, row_chunk);
  const uint32_t words[4] = {decoded.x, decoded.y, decoded.z, decoded.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    output[size_t(row) * kK + row_chunk * 8 + i * 2] = uint16_t(words[i]);
    output[size_t(row) * kK + row_chunk * 8 + i * 2 + 1] = uint16_t(words[i] >> 16);
  }
}

__device__ __forceinline__ void prefetch_range_l2(const void* ptr, size_t bytes) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t(127);
  const uintptr_t end = (reinterpret_cast<uintptr_t>(ptr) + bytes + 127) & ~uintptr_t(127);
  for (uintptr_t p = begin + (threadIdx.x & 31) * 128; p < end; p += 32 * 128)
    asm volatile("prefetch.global.L2 [%0];" : : "l"(p));
}

__global__ void gateup_lossless_simd_kernel(const uint8_t* plane, const uint32_t* headers,
                                       const uint16_t* raw, const uint32_t* raw_offsets,
                                       const bf16* x, float* y) {
  __shared__ __align__(16) float xf[kB * kK];
  for (int i = threadIdx.x; i < kB * kK; i += kThreads) xf[i] = __bfloat162float(x[i]);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gw = warp * kCTAs + blockIdx.x;
  for (int k = 0; k < (kN + kGlobalWarps - 1) / kGlobalWarps; ++k) {
    const int r = gw + k * kGlobalWarps;
    if (r < kN) {
      for (int matrix = 0; matrix < 2; ++matrix) {
        const int row = matrix * kN + r;
        prefetch_range_l2(plane + size_t(row) * kK, kK);
        prefetch_range_l2(headers + size_t(row) * kChunks, kChunks * sizeof(uint32_t));
        const uint32_t first = __ldg(raw_offsets + row);
        const uint32_t last = __ldg(raw_offsets + row + 1);
        if (last > first) prefetch_range_l2(raw + size_t(first) * 8, size_t(last - first) * 16);
      }
    }
  }
  for (int r = gw; r < kN; r += kGlobalWarps) {
    float acc[2][2][4] = {};
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int c = lane + 32 * it;
      if (c < kChunks) {
        const uint4 wg = decode_chunk_simd(plane + size_t(r) * kK, headers + size_t(r) * kChunks, raw, c);
        const uint4 wu = decode_chunk_simd(plane + size_t(kN + r) * kK,
                                      headers + size_t(kN + r) * kChunks, raw, c);
        const uint32_t ug[4] = {wg.x, wg.y, wg.z, wg.w};
        const uint32_t uu[4] = {wu.x, wu.y, wu.z, wu.w};
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          const float4 a = *reinterpret_cast<const float4*>(xf + b * kK + c * 8);
          const float4 d = *reinterpret_cast<const float4*>(xf + b * kK + c * 8 + 4);
          const float xv[8] = {a.x, a.y, a.z, a.w, d.x, d.y, d.z, d.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float wg_f = (e & 1) ? hi_bf16(ug[e >> 1]) : lo_bf16(ug[e >> 1]);
            const float wu_f = (e & 1) ? hi_bf16(uu[e >> 1]) : lo_bf16(uu[e >> 1]);
            acc[0][b][e & 3] = fmaf(wg_f, xv[e], acc[0][b][e & 3]);
            acc[1][b][e & 3] = fmaf(wu_f, xv[e], acc[1][b][e & 3]);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int b = 0; b < kB; ++b) {
        float s = (acc[i][b][0] + acc[i][b][1]) + (acc[i][b][2] + acc[i][b][3]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) y[b * kN * 2 + i * kN + r] = s;
      }
  }
}


__global__ void gateup_lossless_predicated_kernel(const uint8_t* plane, const uint32_t* headers,
                                       const uint16_t* raw, const uint32_t* raw_offsets,
                                       const bf16* x, float* y) {
  __shared__ __align__(16) float xf[kB * kK];
  for (int i = threadIdx.x; i < kB * kK; i += kThreads) xf[i] = __bfloat162float(x[i]);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gw = warp * kCTAs + blockIdx.x;
  for (int k = 0; k < (kN + kGlobalWarps - 1) / kGlobalWarps; ++k) {
    const int r = gw + k * kGlobalWarps;
    if (r < kN) {
      for (int matrix = 0; matrix < 2; ++matrix) {
        const int row = matrix * kN + r;
        prefetch_range_l2(plane + size_t(row) * kK, kK);
        prefetch_range_l2(headers + size_t(row) * kChunks, kChunks * sizeof(uint32_t));
        const uint32_t first = __ldg(raw_offsets + row);
        const uint32_t last = __ldg(raw_offsets + row + 1);
        if (last > first) prefetch_range_l2(raw + size_t(first) * 8, size_t(last - first) * 16);
      }
    }
  }
  for (int r = gw; r < kN; r += kGlobalWarps) {
    float acc[2][2][4] = {};
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int c = lane + 32 * it;
      if (c < kChunks) {
        const uint4 wg = decode_chunk_predicated(plane + size_t(r) * kK, headers + size_t(r) * kChunks, raw, c);
        const uint4 wu = decode_chunk_predicated(plane + size_t(kN + r) * kK,
                                      headers + size_t(kN + r) * kChunks, raw, c);
        const uint32_t ug[4] = {wg.x, wg.y, wg.z, wg.w};
        const uint32_t uu[4] = {wu.x, wu.y, wu.z, wu.w};
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          const float4 a = *reinterpret_cast<const float4*>(xf + b * kK + c * 8);
          const float4 d = *reinterpret_cast<const float4*>(xf + b * kK + c * 8 + 4);
          const float xv[8] = {a.x, a.y, a.z, a.w, d.x, d.y, d.z, d.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float wg_f = (e & 1) ? hi_bf16(ug[e >> 1]) : lo_bf16(ug[e >> 1]);
            const float wu_f = (e & 1) ? hi_bf16(uu[e >> 1]) : lo_bf16(uu[e >> 1]);
            acc[0][b][e & 3] = fmaf(wg_f, xv[e], acc[0][b][e & 3]);
            acc[1][b][e & 3] = fmaf(wu_f, xv[e], acc[1][b][e & 3]);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int b = 0; b < kB; ++b) {
        float s = (acc[i][b][0] + acc[i][b][1]) + (acc[i][b][2] + acc[i][b][3]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) y[b * kN * 2 + i * kN + r] = s;
      }
  }
}

__global__ void gateup_lossless_kernel(const uint8_t* plane, const uint32_t* headers,
                                       const uint16_t* raw, const uint32_t* raw_offsets,
                                       const bf16* x, float* y) {
  __shared__ __align__(16) float xf[kB * kK];
  for (int i = threadIdx.x; i < kB * kK; i += kThreads) xf[i] = __bfloat162float(x[i]);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gw = warp * kCTAs + blockIdx.x;
  for (int k = 0; k < (kN + kGlobalWarps - 1) / kGlobalWarps; ++k) {
    const int r = gw + k * kGlobalWarps;
    if (r < kN) {
      for (int matrix = 0; matrix < 2; ++matrix) {
        const int row = matrix * kN + r;
        prefetch_range_l2(plane + size_t(row) * kK, kK);
        prefetch_range_l2(headers + size_t(row) * kChunks, kChunks * sizeof(uint32_t));
        const uint32_t first = __ldg(raw_offsets + row);
        const uint32_t last = __ldg(raw_offsets + row + 1);
        if (last > first) prefetch_range_l2(raw + size_t(first) * 8, size_t(last - first) * 16);
      }
    }
  }
  for (int r = gw; r < kN; r += kGlobalWarps) {
    float acc[2][2][4] = {};
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int c = lane + 32 * it;
      if (c < kChunks) {
        const uint4 wg = decode_chunk(plane + size_t(r) * kK, headers + size_t(r) * kChunks, raw, c);
        const uint4 wu = decode_chunk(plane + size_t(kN + r) * kK,
                                      headers + size_t(kN + r) * kChunks, raw, c);
        const uint32_t ug[4] = {wg.x, wg.y, wg.z, wg.w};
        const uint32_t uu[4] = {wu.x, wu.y, wu.z, wu.w};
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          const float4 a = *reinterpret_cast<const float4*>(xf + b * kK + c * 8);
          const float4 d = *reinterpret_cast<const float4*>(xf + b * kK + c * 8 + 4);
          const float xv[8] = {a.x, a.y, a.z, a.w, d.x, d.y, d.z, d.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float wg_f = (e & 1) ? hi_bf16(ug[e >> 1]) : lo_bf16(ug[e >> 1]);
            const float wu_f = (e & 1) ? hi_bf16(uu[e >> 1]) : lo_bf16(uu[e >> 1]);
            acc[0][b][e & 3] = fmaf(wg_f, xv[e], acc[0][b][e & 3]);
            acc[1][b][e & 3] = fmaf(wu_f, xv[e], acc[1][b][e & 3]);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int b = 0; b < kB; ++b) {
        float s = (acc[i][b][0] + acc[i][b][1]) + (acc[i][b][2] + acc[i][b][3]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) y[b * kN * 2 + i * kN + r] = s;
      }
  }
}

extern "C" int run_gateup(int async_weights, const void* weights, const void* input,
                           void* output, cudaStream_t stream) {
  const auto* W = static_cast<const bf16*>(weights);
  const auto* x = static_cast<const bf16*>(input);
  auto* y = static_cast<float*>(output);
  if (async_weights == 0) gateup_kernel<false><<<kCTAs, kThreads, 0, stream>>>(W, x, y);
  else if (async_weights == 1) gateup_kernel<true><<<kCTAs, kThreads, 0, stream>>>(W, x, y);
  else return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int run_gateup_lossless(const void* plane, const void* headers,
                                    const void* raw, const void* raw_offsets,
                                    const void* input, void* output, cudaStream_t stream) {
  gateup_lossless_kernel<<<kCTAs, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<const uint32_t*>(raw_offsets),
      static_cast<const bf16*>(input), static_cast<float*>(output));
  return static_cast<int>(cudaGetLastError());
}

__global__ void decode_weights_simd_kernel(const uint8_t* plane, const uint32_t* headers,
                                           const uint16_t* raw, uint16_t* output) {
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= 2 * kN * kChunks) return;
  const int row = chunk / kChunks;
  const int row_chunk = chunk - row * kChunks;
  const uint4 decoded = decode_chunk_simd(plane + size_t(row) * kK,
                                          headers + size_t(row) * kChunks, raw, row_chunk);
  const uint32_t words[4] = {decoded.x, decoded.y, decoded.z, decoded.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    output[size_t(row) * kK + row_chunk * 8 + i * 2] = uint16_t(words[i]);
    output[size_t(row) * kK + row_chunk * 8 + i * 2 + 1] = uint16_t(words[i] >> 16);
  }
}

extern "C" int run_gateup_lossless_simd(const void* plane, const void* headers,
                                        const void* raw, const void* raw_offsets,
                                        const void* input, void* output, cudaStream_t stream) {
  gateup_lossless_simd_kernel<<<kCTAs, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<const uint32_t*>(raw_offsets),
      static_cast<const bf16*>(input), static_cast<float*>(output));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int run_gateup_lossless_predicated(const void* plane, const void* headers,
                                               const void* raw, const void* raw_offsets,
                                               const void* input, void* output, cudaStream_t stream) {
  gateup_lossless_predicated_kernel<<<kCTAs, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<const uint32_t*>(raw_offsets),
      static_cast<const bf16*>(input), static_cast<float*>(output));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int decode_weights_simd(const void* plane, const void* headers,
                                   const void* raw, void* output_uint16,
                                   cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int chunks = 2 * kN * kChunks;
  decode_weights_simd_kernel<<<(chunks + threads - 1) / threads, threads, 0, stream>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<uint16_t*>(output_uint16));
  return static_cast<int>(cudaGetLastError());
}

__global__ void decode_weights_predicated_kernel(const uint8_t* plane, const uint32_t* headers,
                                                const uint16_t* raw, uint16_t* output) {
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= 2 * kN * kChunks) return;
  const int row = chunk / kChunks;
  const int row_chunk = chunk - row * kChunks;
  const uint4 decoded = decode_chunk_predicated(plane + size_t(row) * kK,
                                                 headers + size_t(row) * kChunks, raw, row_chunk);
  const uint32_t words[4] = {decoded.x, decoded.y, decoded.z, decoded.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    output[size_t(row) * kK + row_chunk * 8 + i * 2] = uint16_t(words[i]);
    output[size_t(row) * kK + row_chunk * 8 + i * 2 + 1] = uint16_t(words[i] >> 16);
  }
}

extern "C" int decode_weights_predicated(const void* plane, const void* headers,
                                          const void* raw, void* output_uint16,
                                          cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int chunks = 2 * kN * kChunks;
  decode_weights_predicated_kernel<<<(chunks + threads - 1) / threads, threads, 0, stream>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<uint16_t*>(output_uint16));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int decode_weights(const void* plane, const void* headers,
                               const void* raw, void* output_uint16,
                               cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int chunks = 2 * kN * kChunks;
  decode_weights_kernel<<<(chunks + threads - 1) / threads, threads, 0, stream>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<uint16_t*>(output_uint16));
  return static_cast<int>(cudaGetLastError());
}

__global__ void gateup_lossless_packed_ranges_kernel(const uint8_t* plane, const uint32_t* headers,
                                       const uint16_t* raw, const uint32_t* packed_ranges,
                                       const bf16* x, float* y) {
  __shared__ __align__(16) float xf[kB * kK];
  for (int i = threadIdx.x; i < kB * kK; i += kThreads) xf[i] = __bfloat162float(x[i]);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gw = warp * kCTAs + blockIdx.x;
  for (int k = 0; k < (kN + kGlobalWarps - 1) / kGlobalWarps; ++k) {
    const int r = gw + k * kGlobalWarps;
    if (r < kN) {
      for (int matrix = 0; matrix < 2; ++matrix) {
        const int row = matrix * kN + r;
        prefetch_range_l2(plane + size_t(row) * kK, kK);
        prefetch_range_l2(headers + size_t(row) * kChunks, kChunks * sizeof(uint32_t));
        const uint32_t range = __ldg(packed_ranges + row);
        const uint32_t first = range & 0xFFFFFFu;
        const uint32_t count = range >> 24;
        if (count) prefetch_range_l2(raw + size_t(first) * 8, size_t(count) * 16);
      }
    }
  }
  for (int r = gw; r < kN; r += kGlobalWarps) {
    float acc[2][2][4] = {};
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int c = lane + 32 * it;
      if (c < kChunks) {
        const uint4 wg = decode_chunk_predicated(plane + size_t(r) * kK, headers + size_t(r) * kChunks, raw, c);
        const uint4 wu = decode_chunk_predicated(plane + size_t(kN + r) * kK,
                                      headers + size_t(kN + r) * kChunks, raw, c);
        const uint32_t ug[4] = {wg.x, wg.y, wg.z, wg.w};
        const uint32_t uu[4] = {wu.x, wu.y, wu.z, wu.w};
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          const float4 a = *reinterpret_cast<const float4*>(xf + b * kK + c * 8);
          const float4 d = *reinterpret_cast<const float4*>(xf + b * kK + c * 8 + 4);
          const float xv[8] = {a.x, a.y, a.z, a.w, d.x, d.y, d.z, d.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float wg_f = (e & 1) ? hi_bf16(ug[e >> 1]) : lo_bf16(ug[e >> 1]);
            const float wu_f = (e & 1) ? hi_bf16(uu[e >> 1]) : lo_bf16(uu[e >> 1]);
            acc[0][b][e & 3] = fmaf(wg_f, xv[e], acc[0][b][e & 3]);
            acc[1][b][e & 3] = fmaf(wu_f, xv[e], acc[1][b][e & 3]);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int b = 0; b < kB; ++b) {
        float s = (acc[i][b][0] + acc[i][b][1]) + (acc[i][b][2] + acc[i][b][3]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) y[b * kN * 2 + i * kN + r] = s;
      }
  }
}


extern "C" int run_gateup_lossless_packed_ranges(const void* plane, const void* headers,
                                                  const void* raw, const void* packed_ranges,
                                                  const void* input, void* output, cudaStream_t stream) {
  gateup_lossless_packed_ranges_kernel<<<kCTAs, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<const uint32_t*>(packed_ranges),
      static_cast<const bf16*>(input), static_cast<float*>(output));
  return static_cast<int>(cudaGetLastError());
}
