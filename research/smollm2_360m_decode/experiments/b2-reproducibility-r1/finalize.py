"""Summarize the complete guarded confirmation without dropping failed pilots."""
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
root=HERE/'rl-repaired'
r=json.loads((root/'comparable-work.json').read_text())
assert len(r['runs'])==10
rows=r['runs'];by={x['run_id']:x for x in rows}
a=[x['elapsed_seconds'] for x in rows if x['arm']=='control']
b=[x['elapsed_seconds'] for x in rows if x['arm']=='early']
assert len(a)==len(b)==5
blocks=[]
for controls,earlies in [(['c1','c2'],['e1','e2']),(['c3','c4'],['e3','e4']),(['c5'],['e5'])]:
 av=[by[x]['elapsed_seconds'] for x in controls];bv=[by[x]['elapsed_seconds'] for x in earlies]
 blocks.append({'controls':controls,'earlies':earlies,'both_beat_both':max(bv)<min(av),'mean_change_pct':(statistics.mean(bv)/statistics.mean(av)-1)*100})
r['timing']={'control_median_seconds':statistics.median(a),'early_median_seconds':statistics.median(b),
             'change_pct':(statistics.median(b)/statistics.median(a)-1)*100,'control_range_seconds':[min(a),max(a)],
             'early_range_seconds':[min(b),max(b)],'ranges_overlap':max(b)>=min(a),'strict_blocks':blocks}
r['verified']=all(x['both_beat_both'] for x in blocks) and max(b)<min(a)
r['request_count_range']=[min(x['requests'] for x in rows),max(x['requests'] for x in rows)]
r['generated_token_range']=[min(x['output_tokens'] for x in rows),max(x['output_tokens'] for x in rows)]
r['peak_gpu_mib']=max(x['guard']['peak_gpu_mib'] for x in rows)
r['minimum_host_available_bytes']=min(x['guard']['minimum_available_bytes'] for x in rows)
r['limits']=['Only startup overlap differs between arms; both use invariant serving and repaired mixed decode. No kernel-only or default-preset improvement is inferred.',
             'Identical finalized work and exact trainer metrics; bounded shutdown request variation is included in elapsed time, with exact outputs for common requests.',
             'Two initial diagnostic runs are included in five-per-arm statistics; final gate was frozen before four additional runs per arm.',
             'Earlier BF16-head exploration and failed invariant-only confirmation remain separate and are not pooled.']
(HERE/'final-report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r['timing'],indent=2));print('VERIFIED:',r['verified'])
