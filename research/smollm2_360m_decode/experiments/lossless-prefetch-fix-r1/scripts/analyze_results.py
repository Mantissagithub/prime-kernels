import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRIALS = {
    'candidate-r1': 'all-row phase-ahead hints',
    'early-staged-r1': 'metadata and first row staged early',
    'row-local-r1': 'adjacent rows per CTA',
    'register-epilogue-r1': 'SiLU intermediate stays in registers',
    'early-register-r1': 'early staging plus register epilogue',
    'decode-schedule-r1': 'remove compiler memory clobbers',
    'depth-two-r1': 'two-row lookahead',
    'bulk-cooperative-r1': 'cooperative whole-buffer hints',
    'l1-prefetch-r1': 'L1 hint target',
    'paired-loads-r1': 'pair gate and up loads before reconstruction',
    'sign-prmt-r1': 'sign-extending PRMT reconstruction',
    'shared-gemv-r1': 'shared B2 projection body with dense fallback',
    'parallel-plane-r1': 'overlap plane and header loads',
    'parallel-plane-r2': 'fresh-process repeat',
    'packaged-r1': 'actual packaged extension and poisoned fallback',
    'old-new-r1': 'frozen old lossless versus fixed package',
}
records = []
for tag, hypothesis in TRIALS.items():
    path = HERE / (tag + '.json')
    data = json.loads(path.read_text())
    names = sorted(data['arms'])
    cases = []
    for case in data['cases']:
        placements = case['blocks']
        values = {a: [p['median_ms_per_step'] for p in placements if p['arm'] == a] for a in names}
        assert all(len(v) == 6 for v in values.values())
        assert all(sum(len(p['events_ms']) for p in placements if p['arm'] == a) == 120 for a in names)
        medians = {a: statistics.median(v) for a, v in values.items()}
        exact = case['parity_outside_timing']['A_vs_B']['exact_bitwise_all_layers']
        assert exact
        gates = [max(p['median_ms_per_step'] for p in placements if p['arm'] == names[1] and p['block'] == b) < min(p['median_ms_per_step'] for p in placements if p['arm'] == names[0] and p['block'] == b) for b in range(3)]
        cases.append({'case': case['case_id'], 'batch': case['batch'], 'context': case['context'],
                      'median_ms': medians, 'change_percent': 100 * (medians[names[1]] / medians[names[0]] - 1),
                      'placement_ranges_ms': {a: [min(v), max(v)] for a, v in values.items()},
                      'strict_blocks': gates, 'exact': exact,
                      'graph_refresh': case['parity_outside_timing']['A_vs_B'].get('graph_refresh')})
    guards = {level: json.loads((HERE / ('guard-' + level + '-' + tag) / 'result.json').read_text()) for level in ('inner', 'outer')}
    assert all(g['returncode'] == 0 and g['stop_reason'] is None and g['peak_swap_bytes'] == 0 for g in guards.values())
    records.append({'tag': tag, 'hypothesis': hypothesis,
                    'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'cases': cases,
                    'strict_b2_all_cases': all(all(c['strict_blocks']) for c in cases if c['batch'] == 2),
                    'peak_gpu_mib': max(g['peak_gpu_mib'] for g in guards.values())})
(HERE / 'analysis.json').write_text(json.dumps(records, indent=2) + '\n')
lines = ['attempt\thypothesis\tb2_change_percent_ctx32_128_1024\tstrict_b2_gate\texact\tpeak_gpu_mib']
for record in records:
    delta = ','.join(f'{c["change_percent"]:.4f}' for c in record['cases'] if c['batch'] == 2)
    lines.append(f'{record["tag"]}\t{record["hypothesis"]}\t{delta}\t{record["strict_b2_all_cases"]}\tTrue\t{record["peak_gpu_mib"]}')
(HERE / 'decision.tsv').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
