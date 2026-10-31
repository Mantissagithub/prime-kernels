// SM89 research kernel: one SmolLM2-360M decoder block in one cooperative launch.
// This is isolated under prime-rl/outputs and is not production code.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

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
constexpr int kOffH1 = 6400;     // post-attention residual [4][960]
constexpr int kOffAct = 10240;  // SwiGLU output [4][2560]
constexpr int kScratchBf16 = 20480;
// Global BF16 attention scratch: logits become probabilities in place.
constexpr size_t kOffScores = kScratchBf16;
constexpr size_t kScoreElems = size_t(kMaxB) * kQH * kMaxSeq;
constexpr size_t kOffAttn = kOffScores + kScoreElems;
constexpr size_t kAttnElems = size_t(kMaxB) * kH;
constexpr size_t kWorkspaceElems = kOffAttn + kAttnElems;
constexpr size_t kWorkspaceBytes = kWorkspaceElems * sizeof(uint16_t);

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

// Batch-1 target path: lane L accumulates strided elements L+32j in increasing j.
// Broader projection probes still show rare one-ULP differences from native.
template <int K, int B>
__device__ __forceinline__ void dots(const bf16* __restrict__ row, const float* vs, int lane, float* out) {
  float s[B];
#pragma unroll
  for (int b = 0; b < B; ++b) s[b] = 0.f;
  static_assert(K == kH || K == kI, "unsupported projection K");
  constexpr int kIterations = K / 32;
  int j = 0;
#pragma unroll 1
  for (; j + 7 < kIterations; j += 8) {
    const int idx0 = lane + 32 * (j + 0);
    const int idx1 = lane + 32 * (j + 1);
    const int idx2 = lane + 32 * (j + 2);
    const int idx3 = lane + 32 * (j + 3);
    const int idx4 = lane + 32 * (j + 4);
    const int idx5 = lane + 32 * (j + 5);
    const int idx6 = lane + 32 * (j + 6);
    const int idx7 = lane + 32 * (j + 7);
    const float w0 = bf2f(row[idx0]);
    const float w1 = bf2f(row[idx1]);
    const float w2 = bf2f(row[idx2]);
    const float w3 = bf2f(row[idx3]);
    const float w4 = bf2f(row[idx4]);
    const float w5 = bf2f(row[idx5]);
    const float w6 = bf2f(row[idx6]);
    const float w7 = bf2f(row[idx7]);
#pragma unroll
    for (int b = 0; b < B; ++b) {
      s[b] = add_rn(s[b], mul_rn(w0, vs[b * K + idx0]));
      s[b] = add_rn(s[b], mul_rn(w1, vs[b * K + idx1]));
      s[b] = add_rn(s[b], mul_rn(w2, vs[b * K + idx2]));
      s[b] = add_rn(s[b], mul_rn(w3, vs[b * K + idx3]));
      s[b] = add_rn(s[b], mul_rn(w4, vs[b * K + idx4]));
      s[b] = add_rn(s[b], mul_rn(w5, vs[b * K + idx5]));
      s[b] = add_rn(s[b], mul_rn(w6, vs[b * K + idx6]));
      s[b] = add_rn(s[b], mul_rn(w7, vs[b * K + idx7]));
    }
  }
  if constexpr (kIterations % 8 == 6) {
    const int idx0 = lane + 32 * (j + 0);
    const int idx1 = lane + 32 * (j + 1);
    const int idx2 = lane + 32 * (j + 2);
    const int idx3 = lane + 32 * (j + 3);
    const int idx4 = lane + 32 * (j + 4);
    const int idx5 = lane + 32 * (j + 5);
    const float w0 = bf2f(row[idx0]);
    const float w1 = bf2f(row[idx1]);
    const float w2 = bf2f(row[idx2]);
    const float w3 = bf2f(row[idx3]);
    const float w4 = bf2f(row[idx4]);
    const float w5 = bf2f(row[idx5]);
#pragma unroll
    for (int b = 0; b < B; ++b) {
      s[b] = add_rn(s[b], mul_rn(w0, vs[b * K + idx0]));
      s[b] = add_rn(s[b], mul_rn(w1, vs[b * K + idx1]));
      s[b] = add_rn(s[b], mul_rn(w2, vs[b * K + idx2]));
      s[b] = add_rn(s[b], mul_rn(w3, vs[b * K + idx3]));
      s[b] = add_rn(s[b], mul_rn(w4, vs[b * K + idx4]));
      s[b] = add_rn(s[b], mul_rn(w5, vs[b * K + idx5]));
    }
  }
#pragma unroll
  for (int b = 0; b < B; ++b) out[b] = warp_sum(s[b]);
}

// Match the native MLP GEMV's 16-lane reduction group while retaining strided K order.
template <int K>
__device__ __forceinline__ float dots16(const bf16* __restrict__ row, const float* vec, int lane16) {
  static_assert(K == kH || K == kI, "unsupported projection K");
  float sum = 0.f;
  constexpr int kIterations = K / 16;
  int j = 0;
#pragma unroll 1
  for (; j + 7 < kIterations; j += 8) {
    const int idx0 = lane16 + 16 * (j + 0);
    const int idx1 = lane16 + 16 * (j + 1);
    const int idx2 = lane16 + 16 * (j + 2);
    const int idx3 = lane16 + 16 * (j + 3);
    const int idx4 = lane16 + 16 * (j + 4);
    const int idx5 = lane16 + 16 * (j + 5);
    const int idx6 = lane16 + 16 * (j + 6);
    const int idx7 = lane16 + 16 * (j + 7);
    const float w0 = bf2f(row[idx0]);
    const float w1 = bf2f(row[idx1]);
    const float w2 = bf2f(row[idx2]);
    const float w3 = bf2f(row[idx3]);
    const float w4 = bf2f(row[idx4]);
    const float w5 = bf2f(row[idx5]);
    const float w6 = bf2f(row[idx6]);
    const float w7 = bf2f(row[idx7]);
    sum = add_rn(sum, mul_rn(w0, vec[idx0]));
    sum = add_rn(sum, mul_rn(w1, vec[idx1]));
    sum = add_rn(sum, mul_rn(w2, vec[idx2]));
    sum = add_rn(sum, mul_rn(w3, vec[idx3]));
    sum = add_rn(sum, mul_rn(w4, vec[idx4]));
    sum = add_rn(sum, mul_rn(w5, vec[idx5]));
    sum = add_rn(sum, mul_rn(w6, vec[idx6]));
    sum = add_rn(sum, mul_rn(w7, vec[idx7]));
  }
  if constexpr (kIterations % 8 == 4) {
    const int idx0 = lane16 + 16 * (j + 0);
    const int idx1 = lane16 + 16 * (j + 1);
    const int idx2 = lane16 + 16 * (j + 2);
    const int idx3 = lane16 + 16 * (j + 3);
    const float w0 = bf2f(row[idx0]);
    const float w1 = bf2f(row[idx1]);
    const float w2 = bf2f(row[idx2]);
    const float w3 = bf2f(row[idx3]);
    sum = add_rn(sum, mul_rn(w0, vec[idx0]));
    sum = add_rn(sum, mul_rn(w1, vec[idx1]));
    sum = add_rn(sum, mul_rn(w2, vec[idx2]));
    sum = add_rn(sum, mul_rn(w3, vec[idx3]));
  }
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    sum = add_rn(sum, __shfl_xor_sync(0xffffffffu, sum, offset, 16));
  }
  return sum;
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

// B=2 projection path. One warp owns one 16-row output tile. A is row-major X[16,K]; only
// rows 0 and 1 are live. Each K16 slice is staged to the warp's shared A tile. The row-major
// weight tile is viewed as column-major W^T with leading dimension K.
template <int K>
__device__ __forceinline__ void wmma_project_tile(const bf16* weight, const float* vec_s,
                                                   int tile, int warp,
                                                   bf16* wmma_a, bf16* wmma_b,
                                                   bf16* wmma_b_next, float* wmma_c) {
  using namespace nvcuda;
  const int lane = threadIdx.x & 31;
  bf16* a_tile = wmma_a + warp * 16 * 16;
  bf16* current_b_tile = wmma_b + warp * 16 * 16;
  bf16* next_b_tile = wmma_b_next + warp * 16 * 16;
  float* c_tile = wmma_c + warp * 16 * 16;
  const int row0 = tile * 16;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.f);
  auto prefetch_weight_tile = [&](bf16* shared_tile, int k0) {
    const int output_col = lane >> 1;
    const int k_inner = (lane & 1) * 8;
    bf16* dst = shared_tile + output_col * 16 + k_inner;
    const bf16* src = weight + static_cast<size_t>(row0 + output_col) * K + k0 + k_inner;
    const unsigned int shared_addr = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared_addr), "l"(src) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  for (int idx = lane; idx < 16 * 16; idx += 32) {
    if (idx >= 32) a_tile[idx] = __float2bfloat16_rn(0.f);
  }
  prefetch_weight_tile(current_b_tile, 0);
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncwarp(0xffffffffu);
  for (int k0 = 0; k0 < K; k0 += 16) {
    const int next_k0 = k0 + 16;
    const bool has_next = next_k0 < K;
    if (has_next) prefetch_weight_tile(next_b_tile, next_k0);
    const int b = lane >> 4;
    const int d = lane & 15;
    a_tile[lane] = __float2bfloat16_rn(vec_s[b * K + k0 + d]);
    __syncwarp(0xffffffffu);
    wmma::load_matrix_sync(a_frag, a_tile, 16);
    wmma::load_matrix_sync(b_frag, current_b_tile, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    if (has_next) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncwarp(0xffffffffu);
      bf16* swap_tile = current_b_tile;
      current_b_tile = next_b_tile;
      next_b_tile = swap_tile;
    }
  }
  wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);
  __syncwarp(0xffffffffu);
}

#include "block360_kernel.inc"
