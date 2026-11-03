"""Check request divergence and resolved normal-preset settings."""
import collections,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'summary.json').read_text())
requests={}
for row in summary['runs']:
 p=HERE/'runs'/row['run_id']
 requests[row['run_id']]={(r['prompt_sha256'],r['seed']):r for f in (p/'trace').glob('request-*') for line in f.read_text().splitlines() for r in [json.loads(line)]}
 config=json.loads((p/'rl/configs/attempt_1/resolved/trainer.json').read_text())
 assert config['model']['optimization_dtype']=='float32' and config['model']['reduce_dtype']=='float32'
 assert config['ckpt'] is not None and config['ckpt']['single_file_per_rank'] is False
 row['startup_log_seconds']=None
 log=(p/'rl.log').read_text(); trainer=(p/'rl/logs/attempt_1/trainer.log').read_text()
 def stamp(text,phrase):
  line=next(l for l in text.splitlines() if phrase in l)
  h,m,s=map(int,re.search(r'(\d{2}):(\d{2}):(\d{2})',line).groups())
  return h*3600+m*60+s
 row['startup_log_seconds']=stamp(trainer,'Broadcasting startup policy weights')-stamp(log,'Starting inference on GPU')
comparisons=[]
for name in ['e1','c2']:
 a,b=requests['c1'],requests[name];common=a.keys()&b.keys()
 comparisons.append({'reference':'c1','run':name,'common_prompt_seeds':len(common),'changed_outputs':sum(a[k]!=b[k] for k in common),'reference_only':len(a.keys()-b.keys()),'run_only':len(b.keys()-a.keys())})
summary['request_divergence']=comparisons
summary['verdict']='Inconclusive for whole-RL performance: successful updates/checkpoints, but rollout work and training trajectories differ.'
summary['repeat_decision']='Stop after control/early/control diagnostic; do not spend on five-per-arm timing without comparable work.'
(HERE/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
for row in summary['runs']:
 print(row['run_id'],row['elapsed_seconds'],'requests',row['requests'],'output tokens',row['output_tokens'],'startup log interval',row['startup_log_seconds'])
print(json.dumps(comparisons))
