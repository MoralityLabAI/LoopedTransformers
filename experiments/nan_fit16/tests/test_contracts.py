import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from experiments.nan_fit16.common import ROOT, eligible, read, rows
from experiments.nan_fit16.derive_recipe import derive, REPLACEMENTS
from experiments.nan_fit16.audit import check_steps


class Contracts(unittest.TestCase):
    def test_all_CAL_worlds_retained_in_fixed_order(self):
        public = rows(ROOT / 'fixtures/public.jsonl')
        original = rows('experiments/nan_comp14/fixtures/public.jsonl')
        assert [r['id'] for r in public] == [r['id'] for r in original if r['arm'] in ('focused', 'copy', 'focused_erased')]
        cal = [r for r in public if r['split'] == 'calibration' and not r['copy_control']]
        self.assertEqual(len(cal), 16); self.assertEqual(len({r['root'] for r in cal}), 8)
        for root in {r['root'] for r in cal}:
            self.assertEqual({r['world'] for r in cal if r['root'] == root}, {'A', 'B'})
        self.assertEqual(len(public), 144)

    def test_recipe_changes_only_registered_counts_and_hash_guard(self):
        old = Path('experiments/nan_recomp09/fit.py').read_text(encoding='utf-8')
        generated = (ROOT / 'fit.py').read_text(encoding='utf-8')
        self.assertEqual(generated, derive(old))
        for before, after in REPLACEMENTS.items():
            generated = generated.replace(after, before)
        self.assertEqual(ast.dump(ast.parse(generated)), ast.dump(ast.parse(old)))

    def test_no_duplicate_world_or_DEV_step_substitution(self):
        cfg = read(ROOT / 'config.json'); public = rows(ROOT / 'fixtures/public.jsonl')
        cal = [r for r in public if r['split'] == 'calibration' and not r['copy_control']]
        steps = [dict(step=i, source_id=r['id'], mode='closed_loop', total_loss=1., faith_loss=0., min_loss=1., stoch_loss=0., adv_loss=0.) for i, r in enumerate(cal)]
        self.assertEqual(check_steps(steps, public, cfg)['CAL_roots'], 8)
        broken = copy.deepcopy(steps); broken[1]['source_id'] = broken[0]['source_id']
        with self.assertRaises(AssertionError):
            check_steps(broken, public, cfg)
        broken = copy.deepcopy(steps); broken[-1]['source_id'] = next(r['id'] for r in public if r['split'] == 'development')
        with self.assertRaises(AssertionError):
            check_steps(broken, public, cfg)

    def test_every_gradient_case_and_both_screens_required(self):
        seal = read(ROOT / 'config.json')['upstream_grad15_seal']
        a = dict(status='passed', worker_status='completed', source_manifest_sha256=seal, CAL_prefixes_verified=4, optimizer_steps=0,
            conditions=[dict(passes=t, condition_status='passed', successful_gradient_cases=4, archived_cases=4, missing_registered_cases=0) for t in (1, 2)])
        audits = {'nf4': copy.deepcopy(a), 'bf16': copy.deepcopy(a)}
        with patch('experiments.nan_grad15.common.eligible', return_value={'passed': True}):
            self.assertTrue(eligible({}, audits)['passed'])
            audits['bf16']['conditions'][1]['successful_gradient_cases'] = 3
            self.assertFalse(eligible({}, audits)['passed'])
        with patch('experiments.nan_grad15.common.eligible', return_value={'passed': False}):
            self.assertFalse(eligible({}, {'nf4': a, 'bf16': a})['passed'])


if __name__ == '__main__':
    unittest.main()
