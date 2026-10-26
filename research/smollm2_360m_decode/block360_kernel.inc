constexpr int kDbgNorm1 = 0;
constexpr int kDbgQKV = kDbgNorm1 + kMaxB * kH;
constexpr int kDbgAttn = kDbgQKV + kMaxB * 1600;
constexpr int kDbgO = kDbgAttn + kMaxB * kH;
constexpr int kDbgH1 = kDbgO + kMaxB * kH;
constexpr int kDbgNorm2 = kDbgH1 + kMaxB * kH;
constexpr int kDbgGate = kDbgNorm2 + kMaxB * kH;
constexpr int kDbgUp = kDbgGate + kMaxB * kI;
constexpr int kDbgSilu = kDbgUp + kMaxB * kI;
constexpr int kDbgAct = kDbgSilu + kMaxB * kI;
constexpr int kDbgDown = kDbgAct + kMaxB * kI;
constexpr int kDbgOut = kDbgDown + kMaxB * kH;
constexpr int kDebugLayerBf16 = kDbgOut + kMaxB * kH;
constexpr size_t kDebugStagesBf16 = size_t(kLayers) * kDebugLayerBf16;
constexpr size_t kDbgAttnQ = kDebugStagesBf16;
constexpr size_t kDbgAttnQK = kDbgAttnQ + size_t(kMaxB) * kH;
constexpr size_t kDbgAttnScores = kDbgAttnQK + size_t(kMaxB) * kQH * kMaxSeq;
constexpr size_t kDbgAttnProbs = kDbgAttnScores + size_t(kMaxB) * kQH * kMaxSeq;
constexpr size_t kDbgAttnPv = kDbgAttnProbs + size_t(kMaxB) * kQH * kMaxSeq;
constexpr size_t kDebugBf16 = kDbgAttnPv + size_t(kMaxB) * kH;

template <int B>
constexpr size_t dynamic_smem_bytes() {
  constexpr size_t vectors = size_t(kMaxB) * kI * sizeof(float);
  constexpr size_t wmma_per_warp = size_t(16) * 16 * (2 * sizeof(bf16) + sizeof(float));
  return vectors + size_t(kWarps) * wmma_per_warp;
}

template <int B>
__global__ void __launch_bounds__(kThreads, 1) mk_kernel(const Params p) {
  static_assert(B == 1 || B == 2, "the block prototype covers only B=1 and B=2");

  cg::grid_group grid = cg::this_grid();
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int gw = blockIdx.x * kWarps + warp;

  extern __shared__ __align__(32) unsigned char dynamic_smem[];
  float* vec_s = reinterpret_cast<float*>(dynamic_smem);
  bf16* wmma_a = reinterpret_cast<bf16*>(dynamic_smem + size_t(kMaxB) * kI * sizeof(float));
  bf16* wmma_b = wmma_a + size_t(kWarps) * 16 * 16;
  float* wmma_c = reinterpret_cast<float*>(wmma_b + size_t(kWarps) * 16 * 16);

  __shared__ __align__(16) float q_s[B * kQH * kD];
  __shared__ __align__(16) float k_s[B * kKVH * kD];
  __shared__ __align__(16) float v_s[B * kKVH * kD];
  __shared__ float red[128];
  __shared__ int seq_s[B], pos_s[B];

  bf16* ws = p.ws;
  bf16* scores = ws + kOffScores;
  bf16* attn_out = ws + kOffAttn;
  auto sync_grid = [&]() { grid.sync(); };

  if (t < B) {
    seq_s[t] = max(static_cast<int>(p.seq_lens[t]), 1);
    pos_s[t] = max(static_cast<int>(p.positions[t]), 0);
  }
  __syncthreads();

  for (int layer_index = 0; layer_index < kLayers; ++layer_index) {
  bf16* debug_layer = p.debug ? p.debug + static_cast<size_t>(layer_index) * kDebugLayerBf16 : nullptr;
  const LayerW L = p.layers[layer_index];
  bf16* kvw = reinterpret_cast<bf16*>(p.kv[layer_index]);
  const bf16* hidden = p.out_hidden;
  for (int b = 0; b < B; ++b) {
    rms_norm_to_smem((layer_index == 0 ? p.embed + static_cast<size_t>(p.input_ids[b]) * kH : hidden + b * kH), L.norm1, vec_s + b * kH, red);
  }
  if (p.debug && blockIdx.x == 0) {
    for (int i = t; i < B * kH; i += kThreads)
      debug_layer[kDbgNorm1 + i] = __float2bfloat16_rn(vec_s[i]);
  }

  // QKV output is packed [Q(960), K(320), V(320)] per request.
  if constexpr (B == 1) {
    float out[1];
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const int r = gw + k * kGlobalWarps;
      if (r < 1600) {
        dots<kH, 1>(L.qkv + static_cast<size_t>(r) * kH, vec_s, lane, out);
        if (lane == 0) ws[kOffQKV + r] = __float2bfloat16_rn(out[0]);
      }
    }
  } else {
#pragma unroll 1
    for (int tile = gw; tile < 1600 / 16; tile += kGlobalWarps) {
      wmma_project_tile<kH>(L.qkv, vec_s, tile, warp, wmma_a, wmma_c);
      const int b = lane / 16, j = lane & 15;
      const int r = tile * 16 + j;
      ws[kOffQKV + b * 1600 + r] = __float2bfloat16_rn(wmma_c[warp * 256 + lane]);
    }
  }
  sync_grid();
  if (p.debug) {
    const int gi = blockIdx.x * kThreads + t;
    const int stride = kCTAs * kThreads;
    for (int i = gi; i < B * 1600; i += stride)
      debug_layer[kDbgQKV + i] = ws[kOffQKV + i];
  }

  // Rotate Q/K and retain V in shared memory. CTA 0 appends the current K/V cache entry.
  for (int idx = t; idx < B * kQH * kD; idx += kThreads) {
    const int b = idx / (kQH * kD), rem = idx % (kQH * kD);
    const int head = rem / kD, d = rem % kD;
    q_s[idx] = rope_val(ws + kOffQKV + b * 1600 + head * kD, d, pos_s[b], p.cos_sin, p.cs_fp32);
  }
  for (int idx = t; idx < B * kKVH * kD; idx += kThreads) {
    const int b = idx / (kKVH * kD), rem = idx % (kKVH * kD);
    const int head = rem / kD, d = rem % kD;
    const int qkv_k = 960 + head * kD;
    const int qkv_v = 1280 + head * kD;
    k_s[idx] = rope_val(ws + kOffQKV + b * 1600 + qkv_k, d, pos_s[b], p.cos_sin, p.cs_fp32);
    v_s[idx] = bf2f(ws[kOffQKV + b * 1600 + qkv_v + d]);
  }
  __syncthreads();
  if (p.debug && layer_index == p.debug_layer_index && blockIdx.x == 0) {
    bf16* attn_debug = p.debug + kDebugStagesBf16;
    for (int idx = t; idx < B * kH; idx += kThreads)
      attn_debug[kDbgAttnQ - kDebugStagesBf16 + idx] = __float2bfloat16_rn(q_s[idx]);
  }
  if (blockIdx.x == 0) {
    for (int idx = t; idx < B * kKVH * kD; idx += kThreads) {
      const int b = idx / (kKVH * kD), rem = idx % (kKVH * kD);
      const int head = rem / kD, d = rem % kD;
      const int64_t slot = p.slot_mapping[b];
      if (slot < 0) continue;
      const long long blk = slot / p.block_size, off = slot % p.block_size;
      const long long e = blk * p.block_stride + off * p.token_stride + head * p.head_stride + d;
      kvw[p.k_off + e] = __float2bfloat16_rn(k_s[idx]);
      kvw[p.v_off + e] = __float2bfloat16_rn(v_s[idx]);
    }
  }

  // Make CTA 0's current K/V cache write visible before attention reads it.
  sync_grid();

  using namespace nvcuda;
  constexpr unsigned kFullMask = 0xffffffffu;
  const int pages = (p.max_seq_len + 15) / 16;

  // QK SIMT path: each four-lane group computes one query-token dot product.
  // Each lane accumulates strided K dimensions in increasing order, then the
  // group reduces offsets 2 and 1 with explicit FP32 RN adds.
  constexpr int kQkLanes = 4;
  constexpr int kQkGroupsPerWarp = 32 / kQkLanes;
  const int qk_tasks = B * kQH * pages * 16;
  const int qk_group = lane / kQkLanes;
  const int qk_lane = lane % kQkLanes;
  for (int task_base = gw * kQkGroupsPerWarp; task_base < qk_tasks;
       task_base += kGlobalWarps * kQkGroupsPerWarp) {
    const int task = task_base + qk_group;
    const bool task_valid = task < qk_tasks;
    const int token_in_page = task % 16;
    const int page = (task / 16) % pages;
    const int row = task / (16 * pages);
    const int b = row / kQH;
    const int hq = row % kQH;
    const int pos = page * 16 + token_in_page;
    const int kvh = hq / 3;
    const int seqb = task_valid ? seq_s[b] : 0;
    float partial = 0.f;
    if (task_valid && pos < seqb) {
      const int physical_block = p.block_table[b * p.bt_stride + page];
      for (int j = 0; j < kD / kQkLanes; ++j) {
        const int d = qk_lane + j * kQkLanes;
        const long long e = static_cast<long long>(physical_block) * p.block_stride +
                            token_in_page * p.token_stride + kvh * p.head_stride + d;
        const float q = q_s[(b * kQH + hq) * kD + d];
        const float k = bf2f(kvw[p.k_off + e]);
        partial = add_rn(partial, mul_rn(q, k));
      }
    }
    for (int offset = kQkLanes / 2; offset > 0; offset >>= 1)
      partial = add_rn(partial, __shfl_down_sync(kFullMask, partial, offset, kQkLanes));
    if (task_valid && pos < seqb && qk_lane == 0) {
      const bf16 dot_bf16 = __float2bfloat16_rn(partial);
      const float scaled = mul_rn(bf2f(dot_bf16), kAttnScale);
      const int score_index = (b * kQH + hq) * kMaxSeq + pos;
      scores[score_index] = __float2bfloat16_rn(scaled);
      if (p.debug && layer_index == p.debug_layer_index) {
        bf16* attn_debug = p.debug + kDebugStagesBf16;
        attn_debug[kDbgAttnQK - kDebugStagesBf16 + score_index] = dot_bf16;
        attn_debug[kDbgAttnScores - kDebugStagesBf16 + score_index] = scores[score_index];
      }
    }
  }
  sync_grid();

  // Softmax is row-wise FP32 and writes BF16 probabilities over the score buffer.
  // The warp processes next-power-of-two rows in the persistent-softmax lane order.
  for (int row = gw; row < B * kQH; row += kGlobalWarps) {
    const int b = row / kQH;
    const int seqb = seq_s[b];
    int padded = 1;
    while (padded < seqb) padded <<= 1;
    const int warp_width = min(padded, 32);
    const int iterations = padded / warp_width;
    float local_max = -INFINITY;
    for (int i = 0; i < iterations; ++i) {
      const int pos = lane + i * warp_width;
      if (pos < seqb) local_max = fmaxf(local_max, bf2f(scores[row * kMaxSeq + pos]));
    }
    for (int offset = warp_width / 2; offset > 0; offset >>= 1)
      local_max = fmaxf(local_max, __shfl_xor_sync(kFullMask, local_max, offset));
    float local_sum = 0.f;
    for (int i = 0; i < iterations; ++i) {
      const int pos = lane + i * warp_width;
      if (pos < seqb) local_sum += expf(bf2f(scores[row * kMaxSeq + pos]) - local_max);
    }
    for (int offset = warp_width / 2; offset > 0; offset >>= 1)
      local_sum += __shfl_xor_sync(kFullMask, local_sum, offset);
    for (int i = 0; i < iterations; ++i) {
      const int pos = lane + i * warp_width;
      if (pos < seqb) {
        const float probability = expf(bf2f(scores[row * kMaxSeq + pos]) - local_max) / local_sum;
        scores[row * kMaxSeq + pos] = __float2bfloat16_rn(probability);
        if (p.debug && layer_index == p.debug_layer_index) {
          bf16* attn_debug = p.debug + kDebugStagesBf16;
          attn_debug[kDbgAttnProbs - kDebugStagesBf16 + row * kMaxSeq + pos] = scores[row * kMaxSeq + pos];
        }
      }
    }
  }
  sync_grid();

  // Keep short-context B1 WMMA and reuse two y-partitions for B2.
  if constexpr (B == 1) {
    if (seq_s[0] <= 16) {
      constexpr int kDimTiles = kD / 16;
      const int pv_tasks = B * kKVH * kDimTiles;
      for (int task = gw; task < pv_tasks; task += kGlobalWarps) {
        const int dtile = task % kDimTiles;
        const int head_item = task / kDimTiles;
        const int kvh = head_item % kKVH;
        const int b = head_item / kKVH;
        const int seqb = seq_s[b];
        bf16* a_tile = wmma_a + warp * 256;
        bf16* b_tile = wmma_b + warp * 256;
        float* c_tile = wmma_c + warp * 256;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.f);
        for (int k0 = 0; k0 < seqb; k0 += 16) {
          for (int idx = lane; idx < 256; idx += 32) {
            const int qrow = idx / 16;
            const int token = idx & 15;
            const int pos = k0 + token;
            if (qrow < 3 && pos < seqb) {
              const int hq = kvh * 3 + qrow;
              a_tile[idx] = scores[(b * kQH + hq) * kMaxSeq + pos];
            } else {
              a_tile[idx] = __float2bfloat16_rn(0.f);
            }
            const int vpos = k0 + idx / 16;
            const int d = dtile * 16 + (idx & 15);
            if (vpos < seqb) {
              const int physical_block = p.block_table[b * p.bt_stride + vpos / 16];
              const int token_off = vpos & 15;
              const long long e = static_cast<long long>(physical_block) * p.block_stride +
                                  token_off * p.token_stride + kvh * p.head_stride + d;
              b_tile[idx] = kvw[p.v_off + e];
            } else {
              b_tile[idx] = __float2bfloat16_rn(0.f);
            }
          }
          __syncwarp(kFullMask);
          wmma::load_matrix_sync(a_frag, a_tile, 16);
          wmma::load_matrix_sync(b_frag, b_tile, 16);
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);
        __syncwarp(kFullMask);
        for (int idx = lane; idx < 256; idx += 32) {
          const int qrow = idx / 16;
          if (qrow < 3) {
            const int hq = kvh * 3 + qrow;
            const int d = dtile * 16 + (idx & 15);
            attn_out[b * kH + hq * kD + d] = __float2bfloat16_rn(c_tile[idx]);
          }
        }
      }
    } else {
      constexpr int kDimTiles = kD / 32;
      constexpr int kPvTasks = B * kQH * kDimTiles;
      const int group = t >> 7;
      const int group_lane = t & 127;
      const int y = group_lane >> 5;
      const int x = group_lane & 31;
      const int task = blockIdx.x + group * kCTAs;
      float partial = 0.f;
      if (task < kPvTasks) {
        const int dtile = task % kDimTiles;
        const int qh = (task / kDimTiles) % kQH;
        const int b = task / (kDimTiles * kQH);
        const int kvh = qh / 3;
        const int seqb = seq_s[b];
        for (int base = 0; base < seqb; base += 128) {
          const int start = base + y * 32;
          for (int j = 0; j < 32 && start + j < seqb; ++j) {
            const int pos = start + j;
            const int physical_block = p.block_table[b * p.bt_stride + pos / 16];
            const int token_off = pos & 15;
            const long long e = static_cast<long long>(physical_block) * p.block_stride +
                                token_off * p.token_stride + kvh * p.head_stride + dtile * 32 + x;
            const float prob = bf2f(scores[(b * kQH + qh) * kMaxSeq + pos]);
            const float value = bf2f(kvw[p.v_off + e]);
            partial = add_rn(partial, mul_rn(prob, value));
          }
        }
      }
      float* pv_partials = wmma_c + group * 128;
      pv_partials[y * 32 + x] = partial;
      __syncthreads();
      if (task < kPvTasks && y == 0) {
        const int dtile = task % kDimTiles;
        const int qh = (task / kDimTiles) % kQH;
        const int b = task / (kDimTiles * kQH);
        float result = add_rn(pv_partials[x], pv_partials[32 + x]);
        result = add_rn(result, pv_partials[64 + x]);
        result = add_rn(result, pv_partials[96 + x]);
        attn_out[b * kH + qh * kD + dtile * 32 + x] = __float2bfloat16_rn(result);
      }
      __syncthreads();
    }
  } else {
    constexpr int kDimTiles = kD / 32;
    constexpr int kPvTasks = B * kQH * kDimTiles;
    const int group = t >> 6;
    const int group_lane = t & 63;
    const int y = group_lane >> 5;
    const int x = group_lane & 31;
    const int task = blockIdx.x + group * kCTAs;
    float partial = 0.f;
    if (task < kPvTasks) {
      const int dtile = task % kDimTiles;
      const int qh = (task / kDimTiles) % kQH;
      const int b = task / (kDimTiles * kQH);
      const int kvh = qh / 3;
      const int seqb = seq_s[b];
      const int y_chunk = seqb > 32 ? 32 : 16;
      for (int base = 0; base < seqb; base += 2 * y_chunk) {
        const int start = base + y * y_chunk;
        for (int j = 0; j < y_chunk && start + j < seqb; ++j) {
          const int pos = start + j;
          const int physical_block = p.block_table[b * p.bt_stride + pos / 16];
          const int token_off = pos & 15;
          const long long e = static_cast<long long>(physical_block) * p.block_stride +
                              token_off * p.token_stride + kvh * p.head_stride + dtile * 32 + x;
          const float prob = bf2f(scores[(b * kQH + qh) * kMaxSeq + pos]);
          const float value = bf2f(kvw[p.v_off + e]);
          partial = add_rn(partial, mul_rn(prob, value));
        }
      }
    }
    float* pv_partials = wmma_c + group * 64;
    pv_partials[y * 32 + x] = partial;
    __syncthreads();
    if (task < kPvTasks && y == 0) {
      const int dtile = task % kDimTiles;
      const int qh = (task / kDimTiles) % kQH;
      const int b = task / (kDimTiles * kQH);
      const float result = add_rn(pv_partials[x], pv_partials[32 + x]);
      attn_out[b * kH + qh * kD + dtile * 32 + x] = __float2bfloat16_rn(result);
    }
    __syncthreads();
  }
  sync_grid();
  if (p.debug && layer_index == p.debug_layer_index && blockIdx.x == 0) {
    bf16* attn_debug = p.debug + kDebugStagesBf16;
    for (int idx = t; idx < B * kH; idx += kThreads)
      attn_debug[kDbgAttnPv - kDebugStagesBf16 + idx] = attn_out[idx];
  }
  for (int idx = t; idx < B * kH; idx += kThreads)
    vec_s[idx] = bf2f(attn_out[idx]);
  __syncthreads();
  if (p.debug && blockIdx.x == 0) {
    for (int i = t; i < B * kH; i += kThreads)
      debug_layer[kDbgAttn + i] = __float2bfloat16_rn(vec_s[i]);
  }

  // O projection plus the first residual. The B=2 WMMA result is rounded before the residual add.
  if constexpr (B == 1) {
    float out[1];
#pragma unroll
    for (int k = 0; k < 5; ++k) {
      const int r = gw + k * kGlobalWarps;
      if (r < kH) {
        dots<kH, 1>(L.o + static_cast<size_t>(r) * kH, vec_s, lane, out);
        if (lane == 0) {
          const float h = add_rn(bf2f(layer_index == 0 ? p.embed[static_cast<size_t>(p.input_ids[0]) * kH + r] : hidden[r]), rnd(out[0]));
          if (p.debug) debug_layer[kDbgO + r] = __float2bfloat16_rn(rnd(out[0]));
          ws[kOffH1 + r] = __float2bfloat16_rn(h);
          if (p.debug) debug_layer[kDbgH1 + r] = ws[kOffH1 + r];
        }
      }
    }
  } else {
#pragma unroll 1
    for (int tile = gw; tile < kH / 16; tile += kGlobalWarps) {
      wmma_project_tile<kH>(L.o, vec_s, tile, warp, wmma_a, wmma_c);
      const int b = lane / 16, j = lane & 15, r = tile * 16 + j;
      const float h = add_rn(bf2f(layer_index == 0 ? p.embed[static_cast<size_t>(p.input_ids[b]) * kH + r] : hidden[b * kH + r]), rnd(wmma_c[warp * 256 + lane]));
      if (p.debug) debug_layer[kDbgO + b * kH + r] = __float2bfloat16_rn(rnd(wmma_c[warp * 256 + lane]));
      ws[kOffH1 + b * kH + r] = __float2bfloat16_rn(h);
      if (p.debug) debug_layer[kDbgH1 + b * kH + r] = ws[kOffH1 + b * kH + r];
    }
  }
  sync_grid();

  for (int b = 0; b < B; ++b)
    rms_norm_to_smem(ws + kOffH1 + b * kH, L.norm2, vec_s + b * kH, red);
  if (p.debug && blockIdx.x == 0) {
    for (int i = t; i < B * kH; i += kThreads)
      debug_layer[kDbgNorm2 + i] = __float2bfloat16_rn(vec_s[i]);
  }

  if constexpr (B == 1) {
    const int lane16 = lane & 15;
    const int row_group = lane >> 4;
#pragma unroll 1
    for (int r = 2 * gw + row_group; r < kI; r += 2 * kGlobalWarps) {
      const float go = dots16<kH>(L.gate_up + static_cast<size_t>(r) * kH, vec_s, lane16);
      const float uo = dots16<kH>(L.gate_up + static_cast<size_t>(kI + r) * kH, vec_s, lane16);
      if (lane16 == 0) {
        const float g = rnd(go), u = rnd(uo);
        const float silu = rnd(g / (1.f + expf(-g)));
        if (p.debug) {
          debug_layer[kDbgGate + r] = __float2bfloat16_rn(g);
          debug_layer[kDbgUp + r] = __float2bfloat16_rn(u);
          debug_layer[kDbgSilu + r] = __float2bfloat16_rn(silu);
        }
        ws[kOffAct + r] = __float2bfloat16_rn(mul_rn(silu, u));
        if (p.debug) debug_layer[kDbgAct + r] = ws[kOffAct + r];
      }
    }
  } else {
#pragma unroll 1
    for (int tile = gw; tile < kI / 16; tile += kGlobalWarps) {
      const int r0 = tile * 16;
      wmma_project_tile<kH>(L.gate_up, vec_s, tile, warp, wmma_a, wmma_c);
      const int b = lane / 16, j = lane & 15, r = r0 + j;
      const bf16 g = __float2bfloat16_rn(wmma_c[warp * 256 + lane]);
      ws[kOffQKV + b * kI + r] = g;
      wmma_project_tile<kH>(L.gate_up + static_cast<size_t>(kI) * kH, vec_s, tile, warp, wmma_a, wmma_c);
      const float gf = bf2f(ws[kOffQKV + b * kI + r]);
      const float uf = rnd(wmma_c[warp * 256 + lane]);
      const float silu = rnd(gf / (1.f + expf(-gf)));
      if (p.debug) {
        debug_layer[kDbgGate + b * kI + r] = g;
        debug_layer[kDbgUp + b * kI + r] = __float2bfloat16_rn(uf);
        debug_layer[kDbgSilu + b * kI + r] = __float2bfloat16_rn(silu);
      }
      ws[kOffAct + b * kI + r] = __float2bfloat16_rn(mul_rn(silu, uf));
      if (p.debug) debug_layer[kDbgAct + b * kI + r] = ws[kOffAct + b * kI + r];
    }
  }
  sync_grid();

  for (int idx = t; idx < B * kI; idx += kThreads) vec_s[idx] = bf2f(ws[kOffAct + idx]);
  __syncthreads();
  if constexpr (B == 1) {
    const int lane16 = lane & 15;
    const int row_group = lane >> 4;
#pragma unroll
    for (int r = 2 * gw + row_group; r < kH; r += 2 * kGlobalWarps) {
      const float out = dots16<kI>(L.down + static_cast<size_t>(r) * kI, vec_s, lane16);
      if (lane16 == 0) {
        if (p.debug) debug_layer[kDbgDown + r] = __float2bfloat16_rn(rnd(out));
        const float y = add_rn(bf2f(ws[kOffH1 + r]), rnd(out));
        p.out_hidden[r] = __float2bfloat16_rn(y);
        if (p.debug) debug_layer[kDbgOut + r] = p.out_hidden[r];
      }
    }
  } else {
#pragma unroll 1
    for (int tile = gw; tile < kH / 16; tile += kGlobalWarps) {
      wmma_project_tile<kI>(L.down, vec_s, tile, warp, wmma_a, wmma_c);
      const int b = lane / 16, j = lane & 15, r = tile * 16 + j;
      if (p.debug) debug_layer[kDbgDown + b * kH + r] = __float2bfloat16_rn(rnd(wmma_c[warp * 256 + lane]));
      const float y = add_rn(bf2f(ws[kOffH1 + b * kH + r]), rnd(wmma_c[warp * 256 + lane]));
      p.out_hidden[b * kH + r] = __float2bfloat16_rn(y);
      if (p.debug) debug_layer[kDbgOut + b * kH + r] = p.out_hidden[b * kH + r];
    }
  }
  sync_grid(); // layer output is complete before the next layer reads it
  if (p.layer_outputs) {
    for (int i = t; i < B * kH; i += kThreads)
      p.layer_outputs[(static_cast<size_t>(layer_index) * B * kH) + i] = p.out_hidden[i];
    sync_grid();
  }
  }
  if (blockIdx.x == 0) {
    for (int b = 0; b < B; ++b) {
      rms_norm_to_smem(p.out_hidden + b * kH, p.final_norm, vec_s + b * kH, red);
      for (int i = t; i < kH; i += kThreads) p.out_hidden[b * kH + i] = __float2bfloat16_rn(vec_s[b * kH + i]);
    }
  }
}

template <int B>
static int query_props(int* sm_count, int* coop, int* active, int* major, int* minor,
                       int* regs, int* static_smem, int* optin_smem) {
  int device = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return e;
  cudaDeviceProp prop{};
  e = cudaGetDeviceProperties(&prop, device);
  if (e != cudaSuccess) return e;
  e = cudaDeviceGetAttribute(coop, cudaDevAttrCooperativeLaunch, device);
  if (e != cudaSuccess) return e;
  e = cudaDeviceGetAttribute(optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (e != cudaSuccess) return e;

  constexpr int dynamic_bytes = static_cast<int>(dynamic_smem_bytes<B>());
  e = cudaFuncSetAttribute(mk_kernel<B>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);
  if (e != cudaSuccess) return e;
  cudaFuncAttributes fa{};
  e = cudaFuncGetAttributes(&fa, mk_kernel<B>);
  if (e != cudaSuccess) return e;
  if (fa.sharedSizeBytes + dynamic_bytes > *optin_smem) return kErrResidency;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(active, mk_kernel<B>, kThreads, dynamic_bytes);
  if (e != cudaSuccess) return e;
  *sm_count = prop.multiProcessorCount;
  *major = prop.major;
  *minor = prop.minor;
  *regs = fa.numRegs;
  *static_smem = static_cast<int>(fa.sharedSizeBytes);
  return cudaSuccess;
}

extern "C" size_t decode360_workspace_bytes() { return kWorkspaceBytes; }
extern "C" size_t decode360_debug_bytes() { return size_t(kDebugBf16) * sizeof(bf16); }

extern "C" int decode360(void* stream, int B, const void* input_ids, const void* embed, const void* final_norm,
                                const void* positions, const void* seq_lens,
                                const void* block_table, int bt_stride,
                                const void* slot_mapping, const void* layers,
                                void* const* kv, long long k_off, long long v_off,
                                long long block_stride, long long token_stride,
                                long long head_stride, int block_size,
                                const void* cos_sin, int cs_fp32, void* out_hidden, void* layer_outputs,
                                void* scratch, size_t scratch_bytes,
                                void* debug, size_t debug_bytes, int max_seq_len,
                                int debug_layer_index) {
  if (B < 1 || B > 2) return kErrB;
  if (max_seq_len < 1 || max_seq_len > kMaxSeq) return kErrMaxSeq;
  if (debug_layer_index < -1 || debug_layer_index >= kLayers) return kErrBlock;
  if (!input_ids || !embed || !final_norm || !positions || !seq_lens || !block_table || !slot_mapping || !layers ||
      !kv || !cos_sin || !out_hidden || !scratch) return kErrNull;
  if (scratch_bytes < kWorkspaceBytes) return kErrScratch;
  if (debug && debug_bytes < decode360_debug_bytes()) return kErrScratch;
  if (block_size != 16 || bt_stride < 1 ||
      static_cast<long long>(bt_stride) * block_size < max_seq_len) return kErrBlock;
  if (k_off % 8 || v_off % 8 || block_stride % 8 || token_stride % 8 || head_stride % 8 ||
      k_off < 0 || v_off < 0 || block_stride < 0 || token_stride < 0 || head_stride < 0) return kErrAlign;
  if (reinterpret_cast<uintptr_t>(embed) % 16 || reinterpret_cast<uintptr_t>(final_norm) % 16 ||
      reinterpret_cast<uintptr_t>(layers) % 16 || reinterpret_cast<uintptr_t>(kv) % 16 || reinterpret_cast<uintptr_t>(out_hidden) % 16 ||
      (layer_outputs && reinterpret_cast<uintptr_t>(layer_outputs) % 16) ||
      reinterpret_cast<uintptr_t>(scratch) % 16 || (debug && reinterpret_cast<uintptr_t>(debug) % 16))
    return kErrAlign;

  int sm = 0, coop = 0, active = 0, major = 0, minor = 0, regs = 0, static_smem = 0, optin_smem = 0;
  int status = B == 1 ? query_props<1>(&sm, &coop, &active, &major, &minor, &regs, &static_smem, &optin_smem)
                      : query_props<2>(&sm, &coop, &active, &major, &minor, &regs, &static_smem, &optin_smem);
  if (status != cudaSuccess) return status;
  if (major != 8 || minor != 9) return kErrSm89;
  if (!coop) return kErrCoop;
  if (active * sm < kCTAs) return kErrResidency;

  Params p{};
  p.layers = reinterpret_cast<const LayerW*>(layers);
  p.embed = reinterpret_cast<const bf16*>(embed);
  p.final_norm = reinterpret_cast<const bf16*>(final_norm);
  p.input_ids = reinterpret_cast<const int32_t*>(input_ids);
  p.kv = reinterpret_cast<void* const*>(kv);
  p.positions = reinterpret_cast<const int64_t*>(positions);
  p.seq_lens = reinterpret_cast<const int*>(seq_lens);
  p.block_table = reinterpret_cast<const int*>(block_table);
  p.bt_stride = bt_stride;
  p.slot_mapping = reinterpret_cast<const int64_t*>(slot_mapping);
  p.cos_sin = cos_sin;
  p.cs_fp32 = cs_fp32 ? 1 : 0;
  p.out_hidden = reinterpret_cast<bf16*>(out_hidden);
  p.layer_outputs = reinterpret_cast<bf16*>(layer_outputs);
  p.debug = reinterpret_cast<bf16*>(debug);
  p.ws = reinterpret_cast<bf16*>(scratch);
  p.k_off = k_off; p.v_off = v_off; p.block_stride = block_stride;
  p.token_stride = token_stride; p.head_stride = head_stride;
  p.block_size = block_size;
  p.max_seq_len = max_seq_len;
  p.debug_layer_index = debug_layer_index;
  void* args[] = {&p};
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const size_t dynamic_bytes = dynamic_smem_bytes<1>();
  cudaError_t e;
  if (B == 1) {
    e = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(mk_kernel<1>), dim3(kCTAs),
                                    dim3(kThreads), args, dynamic_bytes, s);
  } else {
    e = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(mk_kernel<2>), dim3(kCTAs),
                                    dim3(kThreads), args, dynamic_smem_bytes<2>(), s);
  }
  if (e != cudaSuccess) return e;
  return cudaPeekAtLastError();
}

extern "C" const char* decode360_status_string(int status) {
  switch (status) {
    case kErrB: return "batch_must_be_1_or_2";
    case kErrMaxSeq: return "max_seq_len_must_be_1_to_8192";
    case kErrNull: return "null_device_pointer";
    case kErrCoop: return "cooperative_launch_unsupported";
    case kErrResidency: return "shared_memory_or_residency_limit";
    case kErrSm89: return "kernel_requires_sm89";
    case kErrAlign: return "kv_offsets_or_pointers_not_16_byte_aligned";
    case kErrScratch: return "scratch_or_debug_buffer_too_small";
    case kErrBlock: return "block_size_or_stride_invalid";
    default: return cudaGetErrorString(static_cast<cudaError_t>(status));
  }
}
