"""Full-RL comparison; exact request multiset and training work gates."""
import argparse,collections,json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent

def read(name):
 p=HERE/'runs'/name
 result=json.loads((p/'completion.json').read_text())
 guard=json.loads((HERE/f'guard-{name}/result.json').read_text())
 assert result['returncode']==guard['returncode']==0 and guard['stop_reason'] is None and guard['peak_swap_bytes']==0
 requests=[json.loads(l) for f in (p/'trace').glob('request-*.jsonl') for l in f.read_text().splitlines()]
 assert requests and all(r.get('status')==200 and 'error' not in r for r in requests)
 result['requests']=len(requests);result['input_tokens']=sum(r['input_tokens'] for r in requests);result['output_tokens']=sum(r['output_tokens'] for r in requests)
 result['request_multiset']=collections.Counter(json.dumps(r,sort_keys=True) for r in requests)
 metrics=[json.loads(l) for l in (p/'rl/monitors/file/metrics.jsonl').read_text().splitlines()]
 last=next(r for r in reversed(metrics) if 'progress/total_rollouts' in r)
 result['training_work']={k:last[k] for k in ('progress/total_rollouts','progress/total_tokens','progress/total_tasks','rollout/no_signal','rollout/discarded')}
 result['training_work_by_step']=[{k:r[k] for k in ('step','progress/total_rollouts','progress/total_tokens','progress/total_tasks','rollout/no_signal','rollout/discarded') if k in r} for r in metrics if 'progress/total_rollouts' in r]
 result['phase_metrics']=[r for r in metrics if r.get('producer')=='trainer' and 'time/forward_backward' in r]
 assert len(result['phase_metrics'])==10
 result['trainer_metrics']=[{k:r[k] for k in ('step','optim/grad_norm','loss/mean','mismatch_kl/all/mean') if k in r} for r in metrics if r.get('producer')=='trainer' and any(k in r for k in ('optim/grad_norm','loss/mean'))]
 libraries=[json.loads(l) for f in (p/'trace').glob('kernel-*.jsonl') for l in f.read_text().splitlines() if json.loads(l).get('event')=='library']
 assert len(libraries)==1 and libraries[0]['sha256']==result['library_sha256']
 inference=(p/'rl/logs/attempt_1/inference.log').read_text();assert 'Lossless 360M decode route active' in inference and 'batch=2' in inference
 trainer=(p/'rl/logs/attempt_1/trainer.log').read_text();assert 'RL trainer finished at step 10' in trainer
 checkpoint=json.loads((p/'checkpoint-manifest.json').read_text())
 assert checkpoint and sum(f['bytes'] for f in checkpoint)>4_000_000_000
 assert any('.metadata' in f['path'] for f in checkpoint)
 result['checkpoint_bytes']=sum(f['bytes'] for f in checkpoint)
 result['checkpoint_files']=len(checkpoint)
 result['guard']=guard
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 rows=[read(n) for n in args.runs];ref=rows[0];request_reference=ref["request_multiset"]
 for r in rows:
  r['exact_request_match']=r['request_multiset']==request_reference;r['training_work_match']=r['training_work']==ref['training_work'];r.pop('request_multiset')
  print(r['run_id'],r['arm'],round(r['elapsed_seconds'],3),'requests',r['requests'],'outtokens',r['output_tokens'],'match',r['exact_request_match'],r['training_work_match'])
 report={'runs':rows,'scope':'Ten-update named 360M preset; max_num_seqs=2, default max_inflight=4, distinct deterministic per-prompt rollout seeds; final checkpoint enabled. Rewards, FP32 optimization/reduction and no-signal filtering unchanged.','trainer_metrics_match':all(r['trainer_metrics']==ref['trainer_metrics'] for r in rows),'all_work_matches':all(r['exact_request_match'] and r['training_work_match'] for r in rows),'notes':['Unused V1 GPU-runner counter hook records zero under the V2 runner; not used as evidence. Complete successful request hashes/counts provide the work comparison.','No timed run had request errors or cancellations. Times include Python startup, training, weight refresh and shutdown.']}
 args.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
