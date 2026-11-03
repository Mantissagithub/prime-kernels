"""Read native wall timers without changing the timed workload."""
import argparse
import datetime
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');a=p.parse_args()
report=[]
for name in a.runs:
 run=HERE/'runs'/name
 c=json.loads((run/'completion.json').read_text())
 metrics=[json.loads(line) for line in (run/'rl/monitors/file/metrics.jsonl').read_text().splitlines()]
 times=[r for r in metrics if r.get('producer')=='trainer' and 'time/forward_backward' in r]
 assert len(times)==10 and sorted(r['step'] for r in times)==list(range(1,11))
 trainer=(run/'rl/logs/attempt_1/trainer.log').read_text();launcher=(run/'rl.log').read_text()
 def stamp(text,phrase):
  line=next(l for l in text.splitlines() if phrase in l)
  h,m,s=map(int,re.search(r'(\d{2}):(\d{2}):(\d{2})',line).groups())
  local=datetime.datetime.fromtimestamp(c['started_unix']).replace(hour=h,minute=m,second=s,microsecond=0)
  if local.timestamp()<c['started_unix']-3600:local+=datetime.timedelta(days=1)
  return local.timestamp()
 buckets={k:sum(r[k] for r in times) for k in ['time/step','time/wait_for_batch','time/load_data','time/broadcast_weights','time/save_ckpt','time/forward_backward']}
 accounted=sum(v for k,v in buckets.items() if k!='time/step')
 startup=stamp(trainer,'Starting training loop')-c['started_unix']
 checkpoint=stamp(trainer,'RL trainer finished at step 10')-stamp(trainer,'Saving final checkpoint')
 launch_done=stamp(launcher,'Training finished!')
 report.append({'run':name,'arm':c['arm'],'whole_rl_seconds':c['elapsed_seconds'],
                'native_trainer_totals_seconds':buckets,'native_other_step_work_seconds':buckets['time/step']-accounted,
                'startup_to_training_loop_log_seconds':startup,'final_checkpoint_log_seconds':checkpoint,
                'launcher_finished_to_process_exit_log_seconds':c['elapsed_seconds']-(launch_done-c['started_unix']),
                'outside_steps_startup_checkpoint_log_seconds':c['elapsed_seconds']-buckets['time/step']-startup-checkpoint,
                'orchestrator_policy_wait_seconds':sum(r.get('time/wait_for_policy',0) for r in metrics if r.get('producer')=='orch'),
                'trainer_steps':times})
 notes=['Native forward_backward includes gradient handling,optimizer and annotations; asynchronous GPU tail can be charged to the following broadcast synchronize. These are process wall buckets, not CUDA-kernel durations.',
        'Broadcast is trainer publication/handshake, not complete engine reload. Some reload and rollout work appears as trainer wait. Orchestrator wait overlaps the trainer lane and must not be added to it.',
        'Startup,final-checkpoint and launcher-exit intervals use one-second log timestamps; residual includes that rounding,per-step metric logging,final sync and teardown.',
        'First-step residual includes the startup weight publication; later per-step publication is measured in time/broadcast_weights.']
 (HERE/'phase-profile.json').write_text(json.dumps({'runs':report,'notes':notes},indent=2)+'\n')
 print(name,{k:round(v,3) for k,v in buckets.items()},'checkpoint log seconds',checkpoint)
