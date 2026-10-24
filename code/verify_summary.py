"""Validate the public summary without retraining; standard library only."""
from pathlib import Path
import json, math
root = Path(__file__).resolve().parents[1]
rows = json.loads((root / 'data/results_1100000.json').read_text())
assert len(rows) == 216
assert {(r['instance'],r['seed'],r['method']) for r in rows} == {(i,s,m) for i in range(24) for s in range(3) for m in ['mean_loss','exponential_loss','GDSH']}
for r in rows:
    for ref in ['LIR','WNH']:
        actual = 100*(r['mean_cost']/r[ref+'_mean']-1)
        assert math.isclose(actual,r['gap_'+ref+'_percent'],abs_tol=1e-8)
for method in ['mean_loss','exponential_loss','GDSH']:
    part = [r for r in rows if r['method'] == method]
    print(method, 'average gap vs LIR:', sum(r['gap_LIR_percent'] for r in part)/len(part))
checks = json.loads((root/'evidence/reevaluation_check_1100000.json').read_text())
assert len(checks['checks']) == 216 and not checks['failures']
assert all(r['passed'] for r in checks['checks'])
print('Public summary and recorded validation: PASS')
