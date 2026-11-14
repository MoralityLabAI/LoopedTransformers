"""Checks for RAM-01 training leakage, source hashes and portable policies."""
from __future__ import annotations
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

class TrainingTests(unittest.TestCase):
    def test_fixed_parameters_and_cpu_forward(self):
        expected = {'flat64': 173252, 'loop32': 82692, 'loop64': 198148, 'loop128': 527364}
        rows = [{'goal_text': 'P : Prop\nh : P\n⊢ P', 'family': 'exact_local', 'script': 'exact h'}]
        for c in train.CONFIGS:
            model = train.CandidatePolicy(c)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected[c.name])
            out = model(train.encode(rows))
            self.assertEqual(tuple(out.shape), (1, 4))
            self.assertEqual(out.device.type, 'cpu')
            out.sum().backward()
            self.assertTrue(model.embedding.weight.grad is not None)

    def test_metadata_and_outcomes_never_enter_features(self):
        row = {'goal_text': 'h : True\n⊢ True', 'family': 'exact_local', 'script': 'exact h'}
        poisoned = {**row, 'theorem': 'SECRET', 'source_file': 'SECRET', 'source_command': 'sorry',
                    'label': 3, 'accepted': True, 'progress': True, 'compiler_returncode': 0}
        a, b = train.encode([row]), train.encode([poisoned])
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]))

    def test_concrete_arguments_affect_features(self):
        common = {'goal_text': 'P Q : Prop\nh : P\ng : Q\n⊢ P', 'family': 'exact_local'}
        a = train.encode([{**common, 'script': 'exact h'}])
        b = train.encode([{**common, 'script': 'exact g'}])
        self.assertFalse(torch.equal(a['action'], b['action']))

    def test_outcomes_distinguish_unsupported_branches(self):
        p = {'accepted': True, 'progress': True, 'closed_in_scratch': False, 'remaining_goal_count': 2}
        self.assertEqual(train.label(p), 2)
        self.assertEqual(train.label({**p, 'remaining_goal_count': 1}), 1)
        self.assertEqual(train.label({**p, 'closed_in_scratch': True, 'remaining_goal_count': 0}), 3)
        self.assertEqual(train.label({**p, 'accepted': False}), 0)

    def test_rejects_unstructured_or_multigoal_text(self):
        for goal in ['hello', '⊢ P\n⊢ Q']:
            with self.assertRaises(ValueError):
                train.encode([{'goal_text': goal, 'family': 'rfl', 'script': 'rfl'}])

    def test_no_inference_candidate_fabrication(self):
        torch.manual_seed(9)
        model = train.CandidatePolicy(train.CONFIGS[1]).eval()
        pool = [('rfl', 'rfl'), ('exact_local', 'exact h')]
        result = model.rank('P : Prop\nh : P\n⊢ P', pool)
        self.assertEqual(sorted(result), sorted(pool))
        self.assertEqual(model.rank('⊢ True', []), [])

    def test_checkpoint_trust_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'bad.pt'
            torch.save({'state_sha256': 'wrong', 'effect_sha256': train.EFFECT_SHA}, p)
            with self.assertRaises(ValueError):
                train.load_checkpoint(p)

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
