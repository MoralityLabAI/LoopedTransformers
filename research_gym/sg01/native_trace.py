"""Development-only SG-01 paired trace reader; no model execution."""
import argparse
import json
import math
from collections import defaultdict


def score(rows, relative_dose_tolerance):
    if not 0 <= relative_dose_tolerance < 1:
        raise ValueError("invalid dose tolerance")
    roots = defaultdict(list)
    seen = set()
    for row in rows:
        if row['phase'] != 'development':
            raise ValueError('development only; confirmation remains sealed')
        root, case = row['root_id'], row['case_id']
        for key in ('root_id', 'case_id', 'checkpoint', 'source_id', 'basis_id', 'placement_sha256', 'policy_sha256'):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f'missing provenance: {key}')
        if (root, case) in seen:
            raise ValueError('duplicate case')
        seen.add((root, case))
        if row['head_parity'] is not True:
            raise ValueError('failed head parity; retain failed receipt, block scoring')
        arms = row['arms']
        names = ('geometry', 'baseline', 'sham')
        doses = [arms[a]['displacement_l2'] for a in names]
        if not all(math.isfinite(d) and d > 0 for d in doses) or max(doses)-min(doses) > relative_dose_tolerance*max(doses):
            raise ValueError('dose mismatch; retain failed receipt, block scoring')
        for a in ('clean', 'corrupt', *names):
            z = arms[a]
            if type(z['correct']) is not bool or not math.isfinite(z['nll']) or z['nll'] < 0 or not math.isfinite(z['state_radius']) or z['state_radius'] < 0:
                raise ValueError('invalid endpoint')
        c, b, g, s = (arms[a] for a in ('corrupt', 'baseline', 'geometry', 'sham'))
        roots[root].append({'paired_effect': int(g['correct'])-int(b['correct']),
                            'sham': int(s['correct'])-int(c['correct']),
                            'nll_gain': b['nll']-g['nll'],
                            'geometry_recovery': int(g['correct'] and not c['correct']),
                            'geometry_corruption': int(c['correct'] and not g['correct'])})
    if not roots:
        raise ValueError('no native development rows')
    out = []
    for root, cases in sorted(roots.items()):
        out.append({'unit_id': root, 'n_cases': len(cases),
                    'paired_effect': str(sum(c['paired_effect'] for c in cases))+'/'+str(len(cases)),
                    'sham_effects': [str(sum(c['sham'] for c in cases))+'/'+str(len(cases))],
                    'mean_nll_gain': sum(c['nll_gain'] for c in cases)/len(cases),
                    'recovery': sum(c['geometry_recovery'] for c in cases),
                    'corruption': sum(c['geometry_corruption'] for c in cases)})
    return {'evidence': 'development trace summaries only; independence and sham family require audit',
            'n_units': len(out), 'n_cases': len(seen), 'pilot': out}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('jsonl')
    p.add_argument('--dose-tolerance', type=float, required=True)
    a = p.parse_args()
    with open(a.jsonl, encoding='utf-8') as f:
        print(json.dumps(score((json.loads(line) for line in f if line.strip()), a.dose_tolerance), indent=2))
