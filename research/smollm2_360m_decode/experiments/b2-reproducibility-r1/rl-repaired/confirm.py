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
    def requests(row):
        result={}
        for raw,count in row['request_multiset'].items():
            request=json.loads(raw);key=(request['prompt_sha256'],request['seed'])
            assert count==1 and key not in result
            result[key]=request
        return result
    expected = requests(ref)
    for row in rows:
        actual=requests(row)
        common=actual.keys() & expected.keys()
        assert len(actual.keys() ^ expected.keys()) <= 4, (row['run_id'],'shutdown request variation exceeds four')
        assert all(actual[k]==expected[k] for k in common), (row['run_id'],'common request output mismatch')
        assert row['library_sha256'] == ref['library_sha256']
        assert abs(row['output_tokens']/ref['output_tokens']-1) <= .05, (row['run_id'],'generated work outside 5%')
        assert row['training_work'] == ref['training_work'], (row['run_id'],'finalized work mismatch')
        assert row['trainer_metrics'] == ref['trainer_metrics'], (row['run_id'],'training trajectory mismatch')
        assert all(math.isfinite(value) for m in row['trainer_metrics'] for key,value in m.items() if key != 'step')
        row.pop('request_multiset')
    (HERE/'comparable-work.json').write_text(json.dumps({'runs':rows,'order':names,
        'scope':'Two-update B2-capable checkpointed RL, same accepted kernel and VLLM_BATCH_INVARIANT=1 in both arms.',
        'gate':'Exact finalized work and trainer loss/gradient/KL at both steps; exact outputs for common prompt/seed requests; at most four differing shutdown requests and 5% generated-token variation; clean guards and no request failures.',
        'gate_frozen':'Repaired integration only: frozen after diagnostic c1/e1 and before four additional runs per arm. Earlier invariant-only gate failed and was not relaxed or reused.'},indent=2)+'\n')
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
