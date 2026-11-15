"""RAM-01 deadline and proof-release tests; mocks never count as proofs."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import regression as r

class RegressionTests(unittest.TestCase):
    def test_axiom_audit_accepts_only_standard_axioms(self):
        wrap = lambda x: 'RAM01_AXIOMS_BEGIN\n'+x+'\nRAM01_AXIOMS_END\n'
        self.assertEqual(r.axiom_audit(wrap("'toy' does not depend on any axioms")), (True, []))
        self.assertTrue(r.axiom_audit(wrap("'toy' depends on axioms: [propext, Classical.choice, Quot.sound]"))[0])
        for bad in ('sorryAx', 'customAxiom', 'Lean.trustCompiler'):
            self.assertFalse(r.axiom_audit(wrap("'toy' depends on axioms: ["+bad+"]"))[0])
        self.assertFalse(r.axiom_audit('plain success')[0])

    def test_scratch_closure_needs_separate_final_call(self):
        phases = []
        def fake(_src, _name, _acts, probe, _timeout, _path):
            phases.append(probe)
            return {'accepted': probe, 'goal_count': 0, 'remaining_goals': [], 'timed_out': False}
        with tempfile.TemporaryDirectory() as td:
            row = r.solve('', 'toy', '⊢ True', None, Path(td), compiler=fake,
                          generator=lambda _g: {'omega': ['omega']})
        self.assertEqual(phases, [True, False])
        self.assertFalse(row['accepted'])
        self.assertEqual(row['reason'], 'final_rejected')

    def test_late_success_gets_no_credit(self):
        ticks = [0.]
        def fake(_src, _name, _acts, probe, _timeout, _path):
            ticks[0] += .2
            return {'accepted': True, 'goal_count': 0, 'remaining_goals': [], 'timed_out': False}
        with tempfile.TemporaryDirectory() as td:
            row = r.solve('', 'toy', '⊢ True', None, Path(td), compiler=fake,
                          generator=lambda _g: {'omega': ['omega']}, clock=lambda: ticks[0], deadline=.1)
        self.assertFalse(row['accepted'])
        self.assertEqual(row['reason'], 'hard_deadline')
        self.assertEqual(row['lean_subprocesses'], 1)

    def test_budget_counts_each_process(self):
        def fake(*args):
            return {'accepted': False, 'goal_count': 0, 'remaining_goals': [], 'timed_out': False}
        with tempfile.TemporaryDirectory() as td:
            row = r.solve('', 'toy', '⊢ True', None, Path(td), compiler=fake,
                          generator=lambda _g: {'exact_local': ['exact h'+str(i) for i in range(14)]})
        self.assertFalse(row['accepted'])
        self.assertEqual(row['lean_subprocesses'], 12)
        self.assertEqual(row['reason'], 'subprocess_budget')

    def test_multigoal_abstention(self):
        def fake(*args):
            return {'accepted': True, 'goal_count': 2, 'remaining_goals': ['⊢ P', '⊢ Q'], 'timed_out': False}
        with tempfile.TemporaryDirectory() as td:
            row = r.solve('', 'toy', '⊢ True', None, Path(td), compiler=fake,
                          generator=lambda _g: {'constructor': ['constructor']})
        self.assertFalse(row['accepted'])
        self.assertEqual(row['reason'], 'multigoal_unsupported')

    def test_policy_cannot_inject_unregistered_script(self):
        class Bad:
            def rank(self, _goal, _pool): return [('exact_local', 'sorry')]
        with tempfile.TemporaryDirectory() as td, self.assertRaises(AssertionError):
            r.solve('', 'toy', '⊢ True', Bad(), Path(td),
                    generator=lambda _g: {'omega': ['omega']})

if __name__ == '__main__':
    unittest.main()
