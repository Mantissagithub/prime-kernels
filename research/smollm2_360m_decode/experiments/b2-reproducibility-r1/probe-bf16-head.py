"""Fixed-weight, seeded serving under different batch groupings."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
p = argparse.ArgumentParser()
p.add_argument('arm', choices=['lossless', 'stock'])
p.add_argument('tag')
a = p.parse_args()
for path in [ROOT / 'src', ROOT / 'packages/prime-rl-configs/src', ROOT / 'deps/prime-kernels']:
    sys.path.insert(0, str(path))
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['TORCH_EXTENSIONS_DIR'] = str(ROOT / 'outputs/megakernel-20261025/lossless-b2-vadd2-r1/codec-build')
for key in ['PRIME_DECODE_MEGAKERNEL', 'PRIME_DECODE_MEGAKERNEL_360M', 'PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS']:
    os.environ.pop(key, None)
if a.arm == 'lossless':
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
setup_vllm_env(InferenceConfig(decode_megakernel=a.arm == 'lossless'))
from vllm import LLM, SamplingParams
model = '/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951'
llm = LLM(model=model, dtype='auto', max_model_len=512, max_num_seqs=2,
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
comparisons = {}
for name, records in results.items():
    comparisons[name] = {
        'tokens_differ_from_single':sum(records[str(i)]['token_ids']!=results['single'][str(i)]['token_ids'] for i in range(16)),
        'logprobs_differ_from_single':sum(records[str(i)]['sampled_logprobs']!=results['single'][str(i)]['sampled_logprobs'] for i in range(16)),
        'tokens_differ_from_paired':sum(records[str(i)]['token_ids']!=results['paired'][str(i)]['token_ids'] for i in range(16)),
    }
(HERE / f'{a.tag}.json').write_text(json.dumps({'arm':a.arm,'plans':plans,'comparisons':comparisons,'results':results},indent=2)+'\n')
print(json.dumps(comparisons),flush=True)
