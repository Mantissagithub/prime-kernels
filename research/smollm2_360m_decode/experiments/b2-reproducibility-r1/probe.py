"""Fixed-weight, seeded serving under different batch groupings."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
p = argparse.ArgumentParser()
p.add_argument('arm', choices=['lossless', 'stock', 'lossless-rowhead', 'lossless-fullaccum', 'lossless-trace', 'lossless-invariant', 'lossless-invariant-stagger', 'lossless-invariant-repair'])
p.add_argument('tag')
a = p.parse_args()
for path in [ROOT / 'src', ROOT / 'packages/prime-rl-configs/src', ROOT / 'deps/prime-kernels']:
    sys.path.insert(0, str(path))
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
if 'invariant' in a.arm:
    os.environ['VLLM_BATCH_INVARIANT'] = '1'
os.environ['TORCH_EXTENSIONS_DIR'] = str(ROOT / 'outputs/megakernel-20261025/lossless-b2-vadd2-r1/codec-build')
for key in ['PRIME_DECODE_MEGAKERNEL', 'PRIME_DECODE_MEGAKERNEL_360M', 'PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS']:
    os.environ.pop(key, None)
if a.arm.startswith('lossless'):
    os.environ['PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS'] = '1'
    import torch.utils.cpp_extension as cpp
    original_load = cpp.load
    def load(name, *args, **kwargs):
        if name == 'prime_decode_megakernel_360m_lossless':
            return str(ROOT / 'outputs/megakernel-20261025/lossless-b2-vadd2-r1/frozen-package/baseline.so')
        return original_load(name, *args, **kwargs)
    cpp.load = load
from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.server import setup_vllm_env
setup_vllm_env(InferenceConfig(decode_megakernel=a.arm.startswith('lossless')))
import torch
from vllm import LLM, SamplingParams
if a.arm == 'lossless-fullaccum':
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
if a.arm == 'lossless-rowhead':
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    apply_head = LogitsProcessor._apply_head
    def row_head(self, lm_head, hidden_states, embedding_bias):
        if hidden_states.shape[0] <= 1:
            return apply_head(self, lm_head, hidden_states, embedding_bias)
        return torch.cat([apply_head(self, lm_head, row.unsqueeze(0), embedding_bias)
                          for row in hidden_states], dim=0)
    LogitsProcessor._apply_head = row_head
phase = None
projection_trace = {}
if a.arm == 'lossless-trace':
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    linear_apply = UnquantizedLinearMethod.apply
    def traced_apply(self, layer, x, bias=None):
        result = linear_apply(self, layer, x, bias)
        entries = projection_trace.setdefault(phase, []) if phase is not None else []
        if phase is not None and x.shape[0] > 2 and len(entries) < 4:
            entries.append({'shape':list(x.shape),'weight_shape':list(layer.weight.shape),
                            'implementation':str(self._gemm_impl),
                            'input_row':x[79].contiguous().view(torch.int16).cpu().tolist(),
                            'output_row':result[79].contiguous().view(torch.int16).cpu().tolist()})
        return result
    UnquantizedLinearMethod.apply = traced_apply
model = '/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951'
llm = LLM(model=model, dtype='auto', hf_overrides={'head_dtype':'float32'}, logprobs_mode='processed_logprobs', max_model_len=512, max_num_seqs=2,
          max_num_batched_tokens=512, gpu_memory_utilization=0.15,
          enable_prefix_caching=False, enforce_eager=False,
          compilation_config={'mode':'NONE', 'cudagraph_mode':'FULL_DECODE_ONLY', 'cudagraph_capture_sizes':[1,2]},
          kernel_config={'enable_jit_warmup':False}, seed=42)
requests = json.loads((HERE / 'prompts.json').read_text())
plans = {
    'single': [[i] for i in range(16)],
    'paired': [list(range(i,i+2)) for i in range(0,16,2)],
    'reversed': [list(reversed(range(i,i+2))) for i in range(0,16,2)],
    'cross_pair': [[i,i+8] for i in range(8)],
    'paired_repeat': [list(range(i,i+2)) for i in range(0,16,2)],
}
results = {}
for name, groups in plans.items():
    phase = name
    records = {}
    for group in groups:
        params = [SamplingParams(temperature=1, top_p=1, top_k=-1, max_tokens=64,
                                 seed=requests[i]['seed'], logprobs=1) for i in group]
        outputs = llm.generate([{'prompt_token_ids':requests[i]['token_ids']} for i in group], params, use_tqdm=False)
        assert len(outputs) == len(group)
        for i, out in zip(group,outputs):
            choice = out.outputs[0]
            records[str(i)] = {'token_ids':list(choice.token_ids),
                              'sampled_logprobs':[float(row[token].logprob) for row,token in zip(choice.logprobs,choice.token_ids)],
                              'finish_reason':choice.finish_reason}
    results[name] = records
    (HERE / f'{a.tag}-partial.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,'completed',len(records),'requests',flush=True)
if a.arm in {'lossless-invariant-stagger','lossless-invariant-repair'}:
    from vllm.model_executor.models.llama import LlamaModel
    from vllm.forward_context import get_forward_context
    model_forward = LlamaModel.forward
    mixed_metadata = []
    def forward(self, input_ids, positions, *args, **kwargs):
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            meta = metadata.get(self.layers[0].self_attn.attn.layer_name)
            if meta is not None and meta.max_query_len > 1:
                starts = meta.query_start_loc.cpu().tolist()
                mixed_metadata.append({'input_tokens':input_ids.numel(), 'max_query_len':meta.max_query_len,
                                       'query_lengths':[y-x for x,y in zip(starts,starts[1:])]})
        result = model_forward(self, input_ids, positions, *args, **kwargs)
        if a.arm == 'lossless-invariant-repair' and isinstance(metadata,dict) and meta is not None and meta.max_query_len > 1:
            starts=meta.query_start_loc.cpu().tolist()
            decode_requests=[i for i,(x,y) in enumerate(zip(starts,starts[1:])) if y-x==1]
            if decode_requests:
                import copy
                from prime_rl.inference.vllm import decode_backend as backend
                req_ids=torch.tensor(decode_requests,device=input_ids.device,dtype=torch.long)
                token_ids=torch.tensor([starts[i] for i in decode_requests],device=input_ids.device,dtype=torch.long)
                sub=copy.copy(meta);sub.max_query_len=1;sub.num_actual_tokens=len(decode_requests)
                sub.seq_lens=meta.seq_lens.index_select(0,req_ids)
                sub.block_table=meta.block_table.index_select(0,req_ids)
                sub.slot_mapping=meta.slot_mapping.index_select(0,token_ids)
                step=backend._select_decode_step({backend.KERNEL_NAME_360M:backend._lossless_module.decode_step},self,backend.KERNEL_NAME_360M,sub,len(decode_requests))
                assert step is not None
                repaired=backend._decode_hidden_states(self,step,input_ids.index_select(0,token_ids),positions.index_select(0,token_ids),sub)
                result=result.clone();result.index_copy_(0,token_ids,repaired)
        return result
    LlamaModel.forward = forward
    records = {}
    engine = llm.llm_engine
    for first in range(0,8,2):
        for offset in range(2):
            i=first+offset
            params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=64,seed=requests[i]['seed'],logprobs=1)
            engine.add_request(str(i),{'prompt_token_ids':requests[i]['token_ids']},params)
            if offset == 0:
                for _ in range(3):
                    assert not any(out.finished for out in engine.step()), 'request finished before stagger'
        while engine.has_unfinished_requests():
            for out in engine.step():
                if out.finished:
                    choice=out.outputs[0]
                    records[out.request_id]={'token_ids':list(choice.token_ids),
                        'sampled_logprobs':[float(row[token].logprob) for row,token in zip(choice.logprobs,choice.token_ids)],
                        'finish_reason':choice.finish_reason}
    from prime_rl.inference.vllm import decode_backend
    (HERE/f'{a.tag}-backend-stats.json').write_text(json.dumps(dict(decode_backend.stats),indent=2)+'\n')
    (HERE/f'{a.tag}-stagger.json').write_text(json.dumps({'metadata':mixed_metadata,'results':records,
        'differences':{k:{'tokens':v['token_ids']!=results['single'][k]['token_ids'],
                          'logprobs':v['sampled_logprobs']!=results['single'][k]['sampled_logprobs']} for k,v in records.items()}},indent=2)+'\n')

comparisons = {}
for name, records in results.items():
    comparisons[name] = {
        'tokens_differ_from_single':sum(records[str(i)]['token_ids']!=results['single'][str(i)]['token_ids'] for i in range(16)),
        'logprobs_differ_from_single':sum(records[str(i)]['sampled_logprobs']!=results['single'][str(i)]['sampled_logprobs'] for i in range(16)),
        'tokens_differ_from_paired':sum(records[str(i)]['token_ids']!=results['paired'][str(i)]['token_ids'] for i in range(16)),
    }
(HERE / f'{a.tag}.json').write_text(json.dumps({'arm':a.arm,'head_dtype':'float32','prefix_caching':False,'projection_trace':projection_trace,'plans':plans,'comparisons':comparisons,'results':results},indent=2)+'\n')
print(json.dumps(comparisons),flush=True)
