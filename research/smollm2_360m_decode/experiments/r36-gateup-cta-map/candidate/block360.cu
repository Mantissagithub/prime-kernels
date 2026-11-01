// SM89 research kernel R28: R27 (FAST warp-GEMV projections) with a FlashDecoding-style split-KV
// attention phase (R6 design adapted to 360M). One SmolLM2-360M decoder block in one cooperative launch.
// This is isolated under prime-rl/outputs and is not production code.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

constexpr int kH = 960;
constexpr int kI = 2560;
constexpr int kQH = 15;
constexpr int kKVH = 5;
constexpr int kD = 64;
constexpr int kMaxB = 2;
constexpr int kLayers = 32;
constexpr int kMaxSeq = 8192;
constexpr int kCTAs = 24;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;         // 8
constexpr int kGlobalWarps = kCTAs * kWarps;  // 192
constexpr float kEps = 1e-5f;
constexpr float kAttnScale = 0.125f;

// Global BF16 scratch (elements), each region has a fixed kMaxB row stride.
constexpr int kOffQKV = 0;      // fused projection [4][1600]
constexpr int kOffH1 = 6400;    // post-attention residual [4][960]
constexpr int kOffAct = 10240;  // SwiGLU output [4][2560]
constexpr int kScratchBf16 = 20480;
// FP32 split-KV partials after the BF16 region (byte offset 40960, 16-byte aligned). One item per
// (request, kv head, chunk): m and l for the 3 query heads of the GQA group, and the unnormalized
// 3 x 64 P.V accumulator. kMaxItems covers 5 * 2 * 19 and 5 * 1 * 38.
constexpr int kMaxItems = kCTAs * kWarps;  // 192
constexpr int kFOffPM = 0;                 // [192][3]
constexpr int kFOffPL = kMaxItems * 3;     // [192][3]
constexpr int kFOffPACC = 2 * kMaxItems * 3;  // [192][3][64]
constexpr int kScratchFloat = kFOffPACC + kMaxItems * 3 * kD;
constexpr size_t kWorkspaceBytes = size_t(kScratchBf16) * sizeof(uint16_t) + size_t(kScratchFloat) * sizeof(float);

// Host status codes.
constexpr int kErrB = 9101;
constexpr int kErrMaxSeq = 9102;
constexpr int kErrNull = 9103;
constexpr int kErrCoop = 9104;
constexpr int kErrResidency = 9105;
constexpr int kErrSm89 = 9106;
constexpr int kErrAlign = 9107;
constexpr int kErrScratch = 9108;
constexpr int kErrBlock = 9109;

// Weight pointers for the single decoder block.
struct LayerW {
  const bf16* norm1;   // [960]
  const bf16* qkv;     // [1600, 960]: q rows 0..959, k 960..1279, v 1280..1599
  const bf16* o;       // [960, 960]
  const bf16* norm2;   // [960]
  const bf16* gate_up; // [5120, 960]: gate rows 0..2559, up 2560..5119
  const bf16* down;    // [960, 2560]
};

struct Params {
  const LayerW* layers;      // device table [32]
  const bf16* embed;        // [vocab, 960]
  const bf16* final_norm;   // [960]
  const int32_t* input_ids; // [B]
  void* const* kv;          // device table [32]
  const int64_t* positions; // [B] absolute position of the new token
  const int* seq_lens;      // [B] tokens in context including the new one
  const int* block_table;   // [B, bt_stride]
  int bt_stride;
  const int64_t* slot_mapping; // [B]
  const void* cos_sin;      // [max_pos, 64]: cos 0..31, sin 32..63
  int cs_fp32;
  bf16* out_hidden;         // [B, 960] final-normalized output
  bf16* layer_outputs;      // optional [32, B, 960] post-block captures
  bf16* debug;              // optional stage dump, fixed kMaxB strides
  bf16* ws;                 // bf16 scratch
  long long k_off, v_off, block_stride, token_stride, head_stride;  // elements
  int block_size;
  int max_seq_len;
  int debug_layer_index;
};

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rnd(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// Halves tree (shfl_down 16..1); lane 0 holds the sum, broadcast to all lanes.
__device__ __forceinline__ float warp_sum(float v) {
  constexpr unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) v = add_rn(v, __shfl_down_sync(mask, v, off));
  return __shfl_sync(mask, v, 0);
}

// CTA-wide sum of one value per thread. All threads must call.
__device__ float block_sum(float v, float* red) {
  const int warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[warp] = v;
  __syncthreads();
  float total = red[0];
  for (int w = 1; w < kWarps; ++w) total = add_rn(total, red[w]);
  __syncthreads();
  return total;
}

// out[i] = bf16(bf16(src[i] * rsqrt(mean(src^2) + eps)) * w[i]), expanded to fp32.
__device__ void rms_norm_to_smem(const bf16* src, const bf16* w, float* out, float* red) {
  constexpr int kReduceThreads = 128;
  constexpr int kInputVector = 4;
  const int lane = threadIdx.x;
  if (lane < kReduceThreads) {
    float values[kInputVector] = {0.f, 0.f, 0.f, 0.f};
    const int vector_count = kH / kInputVector;
    for (int vector_index = lane; vector_index < vector_count; vector_index += kReduceThreads) {
#pragma unroll
      for (int item = 0; item < kInputVector; ++item) {
        const float v = bf2f(src[vector_index * kInputVector + item]);
        values[item] = add_rn(values[item], mul_rn(v, v));
      }
    }
    for (int i = vector_count * kInputVector + lane; i < kH; i += kReduceThreads) {
      const float v = bf2f(src[i]);
      values[0] = add_rn(values[0], mul_rn(v, v));
    }
    float partial = values[0];
#pragma unroll
    for (int item = 1; item < kInputVector; ++item)
      partial = add_rn(partial, values[item]);
    red[lane] = partial;
  }
  __syncthreads();
  for (int offset = kReduceThreads / 2; offset >= 32; offset >>= 1) {
    if (lane < offset) red[lane] = add_rn(red[lane], red[lane + offset]);
    __syncthreads();
  }
  if (lane < 32) {
    float partial = red[lane];
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
      partial = add_rn(partial, __shfl_down_sync(mask, partial, offset));
    if (lane == 0) red[0] = partial;
  }
  __syncthreads();
  const float factor = 1.0f / static_cast<float>(kH);
  const float mean = mul_rn(red[0], factor);
  const float mean_eps = add_rn(mean, kEps);
  const float scale = rsqrtf(mean_eps);
  for (int i = threadIdx.x; i < kH; i += kThreads) {
    out[i] = rnd(mul_rn(rnd(mul_rn(bf2f(src[i]), scale)), bf2f(w[i])));
  }
  __syncthreads();
}

// R27 projection GEMV. One warp computes NR output rows of a row-major BF16 matrix against
// B FP32 activation vectors staged in shared memory (vs[b * K + k]). Each lane reads 16-byte
// (8 x BF16) weight chunks, keeps four independent FP32 FMA accumulators per (row, request),
// and folds them with a 32-lane xor tree. Every weight is read once and applied to all B
// requests. The accumulation order and FMA contraction differ from R25 dots/dots16, so outputs
// are relaxed-parity (see README), not bit-exact to native.
__device__ __forceinline__ float lo_bf16(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_bf16(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

template <int K, int B, int NR>
__device__ __forceinline__ void warp_gemv_rows(const bf16* const* rows, const float* vs, int lane,
                                               float (&res)[NR][B]) {
  static_assert(K % 8 == 0, "projection K must be a multiple of 8");
  constexpr int kChunks = K / 8;
  constexpr int kIters = (kChunks + 31) / 32;
  float acc[NR][B][4];
#pragma unroll
  for (int i = 0; i < NR; ++i)
#pragma unroll
    for (int b = 0; b < B; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][b][e] = 0.f;
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int c = lane + 32 * it;
    if (c < kChunks) {
      uint4 wv[NR];
#pragma unroll
      for (int i = 0; i < NR; ++i) wv[i] = __ldg(reinterpret_cast<const uint4*>(rows[i]) + c);
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const float4 xa = *reinterpret_cast<const float4*>(vs + b * K + c * 8);
        const float4 xb = *reinterpret_cast<const float4*>(vs + b * K + c * 8 + 4);
        const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (int i = 0; i < NR; ++i) {
          const uint32_t u[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float w = (e & 1) ? hi_bf16(u[e >> 1]) : lo_bf16(u[e >> 1]);
            acc[i][b][e & 3] = fmaf(w, xv[e], acc[i][b][e & 3]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < NR; ++i)
#pragma unroll
    for (int b = 0; b < B; ++b) {
      float s = (acc[i][b][0] + acc[i][b][1]) + (acc[i][b][2] + acc[i][b][3]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
      res[i][b] = s;
    }
}

// Prefetch a row into L2, one 128-byte line per lane step.
__device__ __forceinline__ void prefetch_row(const void* ptr, int bytes) {
  const char* c = reinterpret_cast<const char*>(ptr);
  for (int off = (threadIdx.x & 31) * 128; off < bytes; off += 32 * 128) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(c + off));
  }
}

// HF eager RoPE boundaries: round each multiply to BF16, then round the add to BF16.
__device__ __forceinline__ float rope_val(const bf16* head, int d, int pos, const void* cs, int cs_fp32) {
  const int ci = pos * kD + (d & 31);
  const int si = ci + 32;
  float c, s;
  if (cs_fp32) {
    c = reinterpret_cast<const float*>(cs)[ci];
    s = reinterpret_cast<const float*>(cs)[si];
  } else {
    c = bf2f(reinterpret_cast<const bf16*>(cs)[ci]);
    s = bf2f(reinterpret_cast<const bf16*>(cs)[si]);
  }
  const float x = bf2f(head[d]);
  const float y = bf2f(head[d ^ 32]);
  const float xc = rnd(mul_rn(x, c));
  const float ys = rnd(mul_rn(y, s));
  return rnd(add_rn(xc, d < 32 ? -ys : ys));
}

#include "block360_kernel.inc"
