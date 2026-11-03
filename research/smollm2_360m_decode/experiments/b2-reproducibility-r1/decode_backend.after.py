"""Opt-in fused decode path for supported SmolLM2 Llama geometries.

Pure decode batches compute final hidden states with one decode-step call instead of vLLM's
layer stack. lm_head, logits processing and sampling stay in vLLM. Unsupported steps fall
through to vLLM's stock forward. Batch-invariant mode recomputes mixed-batch decode
rows so a joining prefill does not change their arithmetic path.
"""

import importlib.util
import os
from collections import Counter
from copy import copy

import torch
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.llama import LlamaModel

logger = init_logger("vllm.prime_rl.decode_backend")

REFERENCE_ENV = "PRIME_DECODE_REFERENCE"
ENABLE_360M_ENV = "PRIME_DECODE_MEGAKERNEL_360M"
ENABLE_360M_LOSSLESS_ENV = "PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS"
KERNEL_NAME_135M = "decode_megakernel"
KERNEL_NAME_360M = "decode_megakernel_360m"
KERNEL_NAME = KERNEL_NAME_135M
MAX_BATCH = 4
MAX_BATCH_BY_KERNEL = {KERNEL_NAME_135M: MAX_BATCH, KERNEL_NAME_360M: 2}
BLOCK_SIZE_360M = 16
MAX_SEQ_360M = 8192

HIDDEN_SIZE = 576
INTERMEDIATE_SIZE = 1536
NUM_LAYERS = 30
NUM_HEADS = 9
NUM_KV_HEADS = 3
HEAD_DIM = 64
RMS_EPS = 1e-5

HIDDEN_SIZE_360M = 960
INTERMEDIATE_SIZE_360M = 2560
NUM_LAYERS_360M = 32
NUM_HEADS_360M = 15
NUM_KV_HEADS_360M = 5

stats: Counter[str] = Counter()
_lossless_module = None
_lossless_route_logged = False
_lossless_decode_logged = False


def _load_reference(path: str):
    spec = importlib.util.spec_from_file_location("prime_rl_decode_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_step


def _resolve_decode_steps():
    global _lossless_module
    if os.environ.get(REFERENCE_ENV):
        reference = _load_reference(os.environ[REFERENCE_ENV])
        return {KERNEL_NAME_135M: reference}
    import prime_kernels

    names = [KERNEL_NAME_135M]
    if os.environ.get(ENABLE_360M_LOSSLESS_ENV) == "1":
        from prime_kernels import decode_megakernel_360m_lossless

        _lossless_module = decode_megakernel_360m_lossless
        return {KERNEL_NAME_360M: _lossless_module.decode_step}
    # The 360M kernel is slower than stock vLLM in serving, so it stays a research opt-in.
    if os.environ.get(ENABLE_360M_ENV) == "1":
        names.append(KERNEL_NAME_360M)
    return {
        name: prime_kernels.load(name).decode_step
        for name in names
        if name in prime_kernels.KERNELS and prime_kernels.is_available(name)
    }


def _kernel_name_for_config(config) -> str | None:
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    geometry = (
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        head_dim,
        config.rms_norm_eps,
    )
    if geometry == (HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, RMS_EPS):
        return KERNEL_NAME_135M
    if geometry == (
        HIDDEN_SIZE_360M,
        INTERMEDIATE_SIZE_360M,
        NUM_LAYERS_360M,
        NUM_HEADS_360M,
        NUM_KV_HEADS_360M,
        HEAD_DIM,
        RMS_EPS,
    ):
        return KERNEL_NAME_360M
    return None


def _model_kernel_name(model: LlamaModel) -> str | None:
    """Checks model-only eligibility and selects an exact supported geometry."""
    if type(model) is not LlamaModel or model.aux_hidden_state_layers:
        return None
    if get_tensor_model_parallel_world_size() != 1 or get_pp_group().world_size != 1:
        return None
    if model.embed_tokens.weight.dtype != torch.bfloat16:
        return None
    if type(model.layers[0].self_attn.qkv_proj) is not QKVParallelLinear:
        return None
    return _kernel_name_for_config(model.config)


def _model_eligible(model: LlamaModel) -> bool:
    """Compatibility alias for the original 135M eligibility predicate."""
    return _model_kernel_name(model) == KERNEL_NAME_135M


def _metadata_eligible(meta, model: LlamaModel, kernel_name: str, batch: int) -> bool:
    if (
        type(meta).__name__ != "FlashAttentionMetadata"
        or meta.max_query_len != 1
        or meta.num_actual_tokens != batch
        or batch > MAX_BATCH_BY_KERNEL[kernel_name]
    ):
        return False
    if kernel_name == KERNEL_NAME_360M:
        caches = [layer.self_attn.attn.kv_cache for layer in model.layers]
        first = caches[0]
        if (
            first.dtype != torch.bfloat16
            or first.ndim != 4
            or first.shape[1] != NUM_KV_HEADS_360M
            or first.shape[2] != BLOCK_SIZE_360M
            or first.shape[3] != 2 * HEAD_DIM
            or first.stride(3) != 1
            or meta.block_table.dtype != torch.int32
            or meta.block_table.shape[0] < batch
            or meta.block_table.stride(1) != 1
            or meta.block_table.shape[1] < 1
            or meta.block_table.shape[1] * BLOCK_SIZE_360M > MAX_SEQ_360M
        ):
            return False
        return all(
            kv.dtype == first.dtype and kv.shape == first.shape and kv.stride() == first.stride() for kv in caches
        )
    return True


def _select_decode_step(decode_steps, model: LlamaModel, kernel_name: str | None, meta, batch: int):
    if (
        kernel_name is None
        or kernel_name not in decode_steps
        or not 0 < batch <= MAX_BATCH_BY_KERNEL[kernel_name]
        or meta is None
        or not _metadata_eligible(meta, model, kernel_name, batch)
    ):
        return None
    return decode_steps[kernel_name]


def _decode_metadata(model: LlamaModel):
    """FLASH_ATTN metadata when every request in the batch contributes one token, else None."""
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return None
    meta = attn_metadata.get(model.layers[0].self_attn.attn.layer_name)
    if type(meta).__name__ != "FlashAttentionMetadata" or meta.max_query_len != 1:
        return None
    return meta


def _decode_hidden_states(model: LlamaModel, decode_step, input_ids, positions, meta):
    batch = input_ids.shape[0]
    layers = model.layers
    return decode_step(
        input_ids=input_ids,
        positions=positions,
        seq_lens=meta.seq_lens[:batch],
        block_table=meta.block_table[:batch],
        slot_mapping=meta.slot_mapping[:batch],
        embed=model.embed_tokens.weight,
        layers=[
            (
                layer.input_layernorm.weight,
                layer.self_attn.qkv_proj.weight,
                layer.self_attn.o_proj.weight,
                layer.post_attention_layernorm.weight,
                layer.mlp.gate_up_proj.weight,
                layer.mlp.down_proj.weight,
            )
            for layer in layers
        ],
        final_norm=model.norm.weight,
        kv_caches=[layer.self_attn.attn.kv_cache for layer in layers],
        cos_sin_cache=layers[0].self_attn.rotary_emb.cos_sin_cache,
        eps=model.norm.variance_epsilon,
    )


def _repair_mixed_decode(model: LlamaModel, decode_steps, input_ids, positions, hidden_states):
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return hidden_states
    meta = metadata.get(model.layers[0].self_attn.attn.layer_name)
    if type(meta).__name__ != "FlashAttentionMetadata" or meta.max_query_len <= 1:
        return hidden_states
    if torch.cuda.is_current_stream_capturing():
        return hidden_states
    starts = meta.query_start_loc.cpu().tolist()
    requests = [i for i, (start, end) in enumerate(zip(starts, starts[1:])) if end - start == 1]
    if not requests:
        return hidden_states
    request_rows = torch.tensor(requests, dtype=torch.long, device=input_ids.device)
    token_rows = torch.tensor([starts[i] for i in requests], dtype=torch.long, device=input_ids.device)
    decode_meta = copy(meta)
    decode_meta.max_query_len = 1
    decode_meta.num_actual_tokens = len(requests)
    decode_meta.seq_lens = meta.seq_lens.index_select(0, request_rows)
    decode_meta.block_table = meta.block_table.index_select(0, request_rows)
    decode_meta.slot_mapping = meta.slot_mapping.index_select(0, token_rows)
    step = _select_decode_step(decode_steps, model, KERNEL_NAME_360M, decode_meta, len(requests))
    if step is None:
        return hidden_states
    # Keep decode arithmetic and current-token KV entries consistent when a prefill joins.
    decoded = _decode_hidden_states(
        model, step, input_ids.index_select(0, token_rows), positions.index_select(0, token_rows), decode_meta
    )
    result = hidden_states.clone()
    result.index_copy_(0, token_rows, decoded)
    stats["backend_mixed_decode"] += len(requests)
    return result


def refresh_lossless_weights(model: LlamaModel) -> None:
    """Refresh encoded weights after finalized layerwise reloads when the route is enabled."""
    llama_model = model if isinstance(model, LlamaModel) else getattr(model, "model", None)
    if (
        os.environ.get(ENABLE_360M_LOSSLESS_ENV) == "1"
        and isinstance(llama_model, LlamaModel)
        and _kernel_name_for_config(llama_model.config) == KERNEL_NAME_360M
    ):
        global _lossless_module
        if _lossless_module is None:
            from prime_kernels import decode_megakernel_360m_lossless

            _lossless_module = decode_megakernel_360m_lossless
        _lossless_module.refresh_weights([
            (
                layer.input_layernorm.weight,
                layer.self_attn.qkv_proj.weight,
                layer.self_attn.o_proj.weight,
                layer.post_attention_layernorm.weight,
                layer.mlp.gate_up_proj.weight,
                layer.mlp.down_proj.weight,
            )
            for layer in llama_model.layers
        ])
        cache_stats = _lossless_module.cache_statistics(llama_model.layers[0].mlp.gate_up_proj.weight.device)
        if cache_stats is not None:
            logger.info("Lossless 360M GU cache refreshed: %s", cache_stats)


def monkey_patch_decode_backend() -> None:
    """Route pure decode batches of up to MAX_BATCH requests through the decode step.

    Installed only when a decode step is available. Decode steps replayed from CUDA graphs do
    not re-enter Python, so the path each captured graph takes is fixed at capture time.
    """
    if getattr(LlamaModel.forward, "_prime_rl_decode_backend", False):
        return
    decode_steps = _resolve_decode_steps()
    if not decode_steps:
        return

    original_forward = LlamaModel.forward

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs):
        global _lossless_decode_logged, _lossless_route_logged
        kernel_name = _model_kernel_name(self) if inputs_embeds is None else None
        if kernel_name is not None and kernel_name in decode_steps:
            if kernel_name == KERNEL_NAME_360M and os.environ.get(ENABLE_360M_LOSSLESS_ENV) == "1":
                weights = [
                    (
                        layer.input_layernorm.weight,
                        layer.self_attn.qkv_proj.weight,
                        layer.self_attn.o_proj.weight,
                        layer.post_attention_layernorm.weight,
                        layer.mlp.gate_up_proj.weight,
                        layer.mlp.down_proj.weight,
                    )
                    for layer in self.layers
                ]
                cache = _lossless_module.prepare_weights(weights)
                if not _lossless_route_logged:
                    logger.info("Lossless 360M decode route prepared: %s", cache.statistics())
                    _lossless_route_logged = True
            meta = _decode_metadata(self)
            batch = input_ids.shape[0]
            if meta is None:
                stats["stock_not_pure_decode"] += 1
            elif batch > MAX_BATCH_BY_KERNEL[kernel_name] or meta.num_actual_tokens != batch:
                stats["stock_decode_over_max_batch"] += 1
            else:
                decode_step = _select_decode_step(decode_steps, self, kernel_name, meta, batch)
                if decode_step is None:
                    stats["stock_unsupported_metadata"] += 1
                else:
                    stats["backend_decode"] += 1
                    if kernel_name == KERNEL_NAME_360M and os.environ.get(ENABLE_360M_LOSSLESS_ENV) == "1" and not _lossless_decode_logged:
                        logger.info("Lossless 360M decode route active: batch=%d", batch)
                        _lossless_decode_logged = True
                    return _decode_hidden_states(self, decode_step, input_ids, positions, meta)
        elif kernel_name is None or inputs_embeds is not None:
            stats["stock_ineligible"] += 1
        else:
            stats["stock_kernel_unavailable"] += 1
        hidden_states = original_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if (
            kernel_name == KERNEL_NAME_360M
            and os.environ.get(ENABLE_360M_LOSSLESS_ENV) == "1"
            and os.environ.get("VLLM_BATCH_INVARIANT") == "1"
        ):
            hidden_states = _repair_mixed_decode(self, decode_steps, input_ids, positions, hidden_states)
        return hidden_states

    forward._prime_rl_decode_backend = True
    LlamaModel.forward = forward
    logger.info("Decode backend installed for supported SmolLM2 geometries: %s", sorted(decode_steps))
