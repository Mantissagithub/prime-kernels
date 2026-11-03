"""Serial guarded confirmation with a frozen comparable-work gate."""
import collections
import importlib.util
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('matched_work', HERE / 'compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
order = ['c1','e1','e2','c2','c3','e3','e4','c4','c5','e5']

def gate(names):
    rows = [compare.read(n) for n in names]
    ref = rows[0]
    def inputs(row):
        return collections.Counter({json.dumps({k:json.loads(key)[k] for k in ('prompt_sha256','seed','input_tokens')},sort_keys=True): count
                                    for key,count in row['request_multiset'].items()})
    expected = inputs(ref)
    for row in rows:
        assert inputs(row) == expected, (row['run_id'], 'prompt/seed request mismatch')
        assert row['library_sha256'] == ref['library_sha256']
        assert abs(row['output_tokens']/ref['output_tokens']-1) <= .02, (row['run_id'],'generated work outside 2%')
        for key,value in ref['training_work'].items():
            if key == 'progress/total_tokens':
                assert abs(row['training_work'][key]/value-1) <= .01, (row['run_id'],'finalized tokens outside 1%')
            else:
                assert row['training_work'][key] == value, (row['run_id'],key)
        assert all(math.isfinite(value) for m in row['trainer_metrics'] for key,value in m.items() if key != 'step')
        row.pop('request_multiset')
    (HERE/'comparable-work.json').write_text(json.dumps({'runs':rows,'order':names,
        'scope':'Two-update B2-capable checkpointed RL, same accepted kernel and VLLM_BATCH_INVARIANT=1 in both arms.',
        'gate':'Exact prompt/seed/input request multiset and finalized rollouts/tasks/discards; generated tokens within 2% and finalized tokens within 1% of c1; zero errors and clean guards. Outputs/gradients need not be identical.',
        'gate_frozen':'After diagnostic c1/e1, before the additional four runs per arm.'},indent=2)+'\n')
    print('WORK GATE PASS', names, flush=True)

completed=['c1','e1']
gate(completed)
for name in order[2:]:
    command=['bash',str(ROOT/'outputs/megakernel-20261025/runtime/run_gpu.sh'),
             f'tiny-prl-invariant-confirm-{name}',str(HERE/f'guard-{name}'),str(HERE/'run_case.py'),
             'control' if name.startswith('c') else 'early',name]
    if name == order[-1]: command.append('--retain-checkpoint')
    subprocess.run(command,cwd=ROOT,check=True)
    completed.append(name)
    gate(completed)
print('CONFIRMATION COMPLETE',flush=True)
