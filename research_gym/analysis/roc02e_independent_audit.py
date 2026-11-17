"""ROC-02E independent standard-library audit of per-case causal receipts.

Checks original raw row census, SHA hashes, paired identities, radius parity,
source-family restrictions, sign-flip inference and best-control upper bound.
Neither source-family-6/7 data nor neural checkpoints are read.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

SEEDS = (2026111601, 2026111603, 2026111607,
         2026111609, 2026111613, 2026111619)
DEPTHS = (8, 16)
ARMS = ('full_legal_unmatched', 'soft_input_matched',
        'input_induced_hidden', 'observed_input_projection',
        'dark_input_projection', 'random_observed', 'isotropic')
RADII = (.05, .1, .2)
RANKS = (7, 8)
OBS = ('task_only', 'joint')
EDITS = ('rule', 'cue')
PER_MODEL = 256 * len(ARMS) * len(RADII) * len(RANKS) * len(OBS) * len(EDITS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def p_sign_flip(vals):
    assert len(vals) == 6
    observed = sum(vals) / 6
    flips = [sum(a * b for a, b in zip(s, vals)) / 6
             for s in itertools.product((-1., 1.), repeat=6)]
    return sum(x >= observed - 1e-14 for x in flips) / len(flips)


def audit(folder: Path):
    per_seed, per_model = [], []
    for depth in DEPTHS:
        for seed in SEEDS:
            receipt = json.loads((folder / f'roc02e_R{depth}_S{seed}.json').read_text())
            gz = folder / receipt['per_case_file']
            assert gz.name == f'roc02e_R{depth}_S{seed}.jsonl.gz'
            assert sha(gz) == receipt['per_case_sha256']
            assert receipt['rows'] == PER_MODEL == 43008
            assert (receipt['seed'], receipt['depth']) == (seed, depth)
            assert not receipt['sealed_test_evaluated']
            assert set(receipt['observers']) == {
                '7:task_only', '7:joint', '8:task_only', '8:joint'
            }
            keys, original_ids = set(), {}
            select = {}
            invalid, n = 0, 0
            with gzip.open(gz, 'rt', encoding='utf-8') as f:
                for line in f:
                    a = json.loads(line)
                    assert a['seed'] == seed and a['depth'] == depth
                    assert a['source_is_calibration'] and not a['sealed_test_evaluated']
                    assert a['family'] in (4, 5)
                    assert a['counterfactual_label'] == (a['original_label'] + 4) % 8
                    assert 0 <= a['case_index'] < 256
                    assert a['rank'] in RANKS and a['edit'] in EDITS
                    assert a['observer'] in OBS and a['arm'] in ARMS
                    assert a['radius'] in RADII
                    assert a['gamma'] == receipt['observers'][
                        f"{a['rank']}:{a['observer']}"]['gamma']
                    key = (a['rank'], a['observer'], a['edit'], a['radius'],
                           a['arm'], a['case_index'])
                    assert key not in keys
                    keys.add(key)
                    n += 1
                    prev = original_ids.setdefault(a['case_index'], a['original_case_id'])
                    assert prev == a['original_case_id']
                    if a['valid']:
                        assert a['logit_directional_gain'] is not None
                        assert a['observer_directional_gain'] is not None
                        assert math.isfinite(a['logit_directional_gain'])
                        assert math.isfinite(a['observer_normalized_gain'])
                        if a['arm'] != 'full_legal_unmatched':
                            assert abs(a['actual_midpoint_l2'] - a['radius']) <= 5.01e-5
                        if a['arm'] == 'soft_input_matched':
                            assert a['interpolation_t'] is not None
                            assert abs(a['radius_error']) <= 2.01e-5
                    else:
                        invalid += 1
                        assert a['logit_directional_gain'] is None
                    if depth == 16 and a['rank'] == 7 and a['radius'] == .2:
                        select[(a['observer'], a['edit'], a['arm'], a['case_index'])] = a
            assert n == len(keys) == PER_MODEL and len(original_ids) == 256
            per_model.append({'seed': seed, 'depth': depth, 'rows': n,
                              'invalid_matched_rows': invalid, 'sha256': sha(gz)})
            if depth == 16:
                diffs, best_diffs = [], []
                family_diffs = defaultdict(list)
                rule_legal, cue_legal = [], []
                eligibility = {'rule': 0, 'cue': 0}
                for i in range(256):
                    def val(ob, edit, arm):
                        row = select[(ob, edit, arm, i)]
                        return float(row['logit_directional_gain']) if row['valid'] else None
                    joint = val('joint', 'rule', 'observed_input_projection')
                    task = val('task_only', 'rule', 'observed_input_projection')
                    assert joint is not None and task is not None
                    family = select[('joint', 'rule',
                                     'observed_input_projection', i)]['family']
                    delta = joint - task
                    diffs.append(delta)
                    family_diffs[family].append(delta)
                    controls = {
                        'rule_soft_input': val('joint', 'rule', 'soft_input_matched'),
                        'cue_soft_input': val('joint', 'cue', 'soft_input_matched'),
                        'task_observed_rule': task,
                        'joint_isotropic_rule': val('joint', 'rule', 'isotropic'),
                        'input_induced_hidden_rule':
                            val('joint', 'rule', 'input_induced_hidden'),
                    }
                    eligibility['rule'] += controls['rule_soft_input'] is not None
                    eligibility['cue'] += controls['cue_soft_input'] is not None
                    available = [v for v in controls.values() if v is not None]
                    assert available
                    best_diffs.append(joint - max(available))
                    rule_legal.append(val('joint', 'rule', 'full_legal_unmatched'))
                    cue_legal.append(val('joint', 'cue', 'full_legal_unmatched'))
                assert {k: len(v) for k,v in family_diffs.items()} == {4:117,5:139}
                per_seed.append({
                    'seed': seed, 'depth': depth, 'rank': 7, 'radius': .2,
                    'mean_joint_minus_task': sum(diffs) / 256,
                    'mean_joint_minus_best_control': sum(best_diffs) / 256,
                    'mean_full_legal_rule_gain': sum(rule_legal) / 256,
                    'mean_full_legal_cue_gain': sum(cue_legal) / 256,
                    'family4_joint_minus_task': sum(family_diffs[4]) / 117,
                    'family5_joint_minus_task': sum(family_diffs[5]) / 139,
                    'n_rule_soft_eligible': eligibility['rule'],
                    'n_cue_soft_eligible': eligibility['cue'],
                })
    assert len(per_model) == 12 and len(per_seed) == 6
    assert sum(r['rows'] for r in per_model) == 516096
    d = [x['mean_joint_minus_task'] for x in per_seed]
    best = [x['mean_joint_minus_best_control'] for x in per_seed]
    result = {
        'study': 'ROC-02E-independent-public-receipt-audit',
        'source': 'Exposed family4/5 calibration; no sealed source read',
        'per_model': per_model, 'primary_by_seed': per_seed,
        'primary_exact_six_sign_flip_p': p_sign_flip(d),
        'primary_mean_joint_minus_task': sum(d) / 6,
        'primary_positive_seeds': sum(x > 0 for x in d),
        'best_input_and_task_control_mean_diff': sum(best) / 6,
        'best_control_positive_seeds': sum(x > 0 for x in best),
        'total_per_case_rows': 516096, 'sealed_test_evaluated': False,
    }
    summary = json.loads((folder / 'roc02e_summary.json').read_text())
    primary = summary['primary']
    assert abs(result['primary_mean_joint_minus_task'] -
               primary['mean_joint_minus_task']) < 1e-10
    assert result['primary_positive_seeds'] == primary['count_positive']
    assert abs(result['primary_exact_six_sign_flip_p'] -
               primary['one_sided_sign_flip_p_64']) < 1e-12
    for row in per_seed:
        assert abs(row['mean_joint_minus_task'] -
                   float(primary['seed_differences'][str(row['seed'])])) < 1e-10
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--receipt-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.receipt_dir)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ('per_model', 'primary_by_seed')}, indent=2))


if __name__ == '__main__':
    main()
