import json
import statistics
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
rows = []
for case in report['cases']:
    samples = {a: [p['median_ms_per_step'] for p in case['blocks'] if p['arm'] == a] for a in ('A-R30', 'B-Lossless')}
    medians = {a: statistics.median(values) for a, values in samples.items()}
    gates = [max(p['median_ms_per_step'] for p in case['blocks'] if p['arm'] == 'B-Lossless' and p['block'] == i) < min(p['median_ms_per_step'] for p in case['blocks'] if p['arm'] == 'A-R30' and p['block'] == i) for i in range(3)]
    rows.append({'case': case['case_id'], 'median_ms': medians,
                 'placement_ranges_ms': {a: [min(v), max(v)] for a, v in samples.items()},
                 'change_percent': 100 * (medians['B-Lossless'] / medians['A-R30'] - 1),
                 'strict_blocks': gates,
                 'exact': case['parity_outside_timing']['A_vs_B']['exact_bitwise_all_layers']})
print(json.dumps(rows, indent=2))
