"""FC-10 original TypedLoop, Lean state input and train-only model contracts."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc10_tactics as fc10


class FC10ModelTests(unittest.TestCase):
    def test_registry_is_frozen_and_kernel_proof_candidates_safe(self):
        self.assertEqual(len(fc10.REGISTRY),12)
        self.assertNotIn("sorry",fc10.REGISTRY)
        self.assertNotIn("admit",fc10.REGISTRY)
        self.assertEqual(fc10.PREREG_SHA,"32330e86366f3ea6378ceb1bf6a4b189610b4483")

    def test_hashed_goal_feature_is_deterministic(self):
        text="a b : Nat\nhab : a ≤ b\n⊢ a + 2 ≤ b + 2"
        a=fc10.features([text])
        b=fc10.features([text])
        self.assertEqual(a["tokens"].shape,(1,fc10.MAX_TOKENS))
        self.assertEqual(a["prem"].shape,(1,12,fc10.MAX_TOKENS))
        for key in a:
            self.assertTrue(torch.equal(a[key],b[key]))

    def test_real_goal_turnstile_is_required(self):
        with self.assertRaises(AssertionError):
            fc10.features(["just a theorem label without Lean goal"])

    def test_original_lr01_typed_loop_is_actually_used(self):
        for loops in (1,4):
            model=fc10.TacticLoop(loops=loops)
            self.assertIsInstance(model.cell,fc10.TypedLoop)
            self.assertEqual(model.loops,loops)
            inp=fc10.features(["P : Prop\nh : P\n⊢ P",
                               "a : Nat\n⊢ a = a"])
            logits=model(inp)
            self.assertEqual(tuple(logits.shape),(2,12))
            self.assertTrue(torch.isfinite(logits).all())
            loss=logits.square().mean()
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in model.cell.parameters()))

    def test_training_only_lexical_reproducibility(self):
        training=[
            {"goal_state":"P : Prop\nh : P\n⊢ P","reference_tactic":"assumption"},
            {"goal_state":"x : Nat\n⊢ x = x","reference_tactic":"rfl"},
        ]
        a=fc10.lexical_rank(training,"Q : Prop\nh : Q\n⊢ Q")
        b=fc10.lexical_rank(training,"Q : Prop\nh : Q\n⊢ Q")
        self.assertEqual(a,b)
        self.assertEqual(sorted(a),list(range(12)))

    def test_model_training_does_not_need_test_generator(self):
        root=os.environ.get("FC10_TRAIN_ROOT")
        if not root:
            raise RuntimeError("FC10_TRAIN_ROOT must point to pinned pre-test source")
        p=Path(root)
        self.assertFalse((p/"research/refutation_bench/fc10/test_data.py").exists())
        self.assertTrue((p/"research/refutation_bench/fc10/train_data.py").exists())
        self.assertTrue((p/"research/refutation_bench/fc10/core.py").exists())

    def test_short_train_uses_reference_labels_without_test(self):
        rows=[{"goal_state":"P : Prop\nh : P\n⊢ P","reference_tactic":"assumption"},
              {"goal_state":"a : Nat\n⊢ a = a","reference_tactic":"rfl"}]*8
        old=fc10.TRAIN_STEPS
        try:
            fc10.TRAIN_STEPS=2
            torch.set_num_threads(2)
            model,report=fc10.train_one(rows,rows,17,1)
            self.assertEqual(report["steps"],2)
            self.assertGreater(report["model_parameters"],0)
        finally:
            fc10.TRAIN_STEPS=old


if __name__=="__main__":
    unittest.main()
