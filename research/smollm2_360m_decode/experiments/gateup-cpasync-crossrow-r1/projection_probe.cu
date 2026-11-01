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

__global__ void gateup_crossrow_kernel(const bf16* W, const bf16* x, float* y) {
  __shared__ __align__(16) float xf[kB * kK];
  __shared__ __align__(16) uint4 weight_tiles[kWarps][2][2][32];
  for (int i = threadIdx.x; i < kB * kK; i += kThreads) xf[i] = __bfloat162float(x[i]);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gw = warp * kCTAs + blockIdx.x;
  for (int k = 0; k < (kN + kGlobalWarps - 1) / kGlobalWarps; ++k) {
    const int r = gw + k * kGlobalWarps;
    if (r < kN) {
      prefetch_row_l2(W + static_cast<size_t>(r) * kK, kK * sizeof(bf16));
      prefetch_row_l2(W + static_cast<size_t>(kN + r) * kK, kK * sizeof(bf16));
    }
  }

  const int rows = (kN - gw + kGlobalWarps - 1) / kGlobalWarps;
  const int tasks = rows * kIters;
  float acc[2][2][4];
  const bf16* row_g = W + static_cast<size_t>(gw) * kK;
  const bf16* row_u = W + static_cast<size_t>(kN + gw) * kK;
  prefetch_pair(row_g, row_u, weight_tiles[warp], 0, lane, 0);
  asm volatile("cp.async.commit_group;");

  for (int task = 0; task < tasks; ++task) {
    const int row = gw + (task >> 2) * kGlobalWarps;
    const int iter = task & 3;
    row_g = W + static_cast<size_t>(row) * kK;
    row_u = W + static_cast<size_t>(kN + row) * kK;

    if (task + 1 < tasks) {
      const int next = task + 1;
      const int next_row = gw + (next >> 2) * kGlobalWarps;
      const int next_iter = next & 3;
      const bf16* next_g = W + static_cast<size_t>(next_row) * kK;
      const bf16* next_u = W + static_cast<size_t>(kN + next_row) * kK;
      prefetch_pair(next_g, next_u, weight_tiles[warp], next_iter & 1, lane, next_iter);
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncwarp();
    asm volatile("" ::: "memory");

    if (iter == 0) {
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int b = 0; b < kB; ++b)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[i][b][e] = 0.0f;
    }

    const int c = lane + 32 * iter;
    if (c < kChunks) {
      const int buffer = iter & 1;
      const uint4 wg = weight_tiles[warp][buffer][0][lane];
      const uint4 wu = weight_tiles[warp][buffer][1][lane];
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

    if (iter == kIters - 1) {
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          float s = (acc[i][b][0] + acc[i][b][1]) + (acc[i][b][2] + acc[i][b][3]);
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
          if (lane == 0) y[b * kN * 2 + i * kN + row] = s;
        }
    }
  }
}

extern "C" int run_gateup(int async_weights, const void* weights, const void* input,
                           void* output, cudaStream_t stream) {
  const auto* W = static_cast<const bf16*>(weights);
  const auto* x = static_cast<const bf16*>(input);
  auto* y = static_cast<float*>(output);
  if (async_weights == 0) gateup_kernel<false><<<kCTAs, kThreads, 0, stream>>>(W, x, y);
  else if (async_weights == 1) gateup_crossrow_kernel<<<kCTAs, kThreads, 0, stream>>>(W, x, y);
  else return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(cudaGetLastError());
}
