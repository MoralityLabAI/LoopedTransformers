"""FC-11 imported actual Lean tactic-state model and certificate safety."""
from __future__ import annotations
from pathlib import Path
import os
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc11_policy as model
import fc11_scratch as scratch


class FC11PolicyTests(unittest.TestCase):
    def test_source_pins_and_candidate_coverage(self):
        self.assertEqual(model.PREREG,"b57e2500e829b12fdfc257077b8e410171c93b98")
        self.assertEqual(model.TRAIN_REV,"2e7e749d4c7f8610a4083f4f8551d084102ea55a")
        self.assertEqual(len(model.VOCAB),8)
        self.assertEqual(model.MAX_SEARCH_ACTIONS,3)
        self.assertTrue(set(model.VOCAB)==set(model.core.TACTICS))
        root=os.environ.get("FC11_TRAIN_ROOT")
        if root:
            self.assertFalse(
              (Path(root)/"research/refutation_bench/fc11/test_data.py").exists())

    def test_encoder_carries_real_turnstile_through_long_context(self):
        s=" ".join(f"hyp{i} : Nat" for i in range(250))+" ⊢ 1 ≤ 2"
        out=model.graph_batch([s])
        self.assertEqual(out["tokens"].shape,(1,model.MAX_TOKENS))
        self.assertEqual(out["head"].shape,(1,8,model.MAX_TOKENS))
        self.assertEqual(float(out["head"].sum()),8.)
        with self.assertRaises(AssertionError):
            model.tokenize("missing a Lean goal turnstile")

    def test_orig_lr01_cell_shared_over_four_loops(self):
        torch.set_num_threads(2)
        policy=model.ImportedLoop(loops=4)
        self.assertEqual(sum(p.numel() for p in policy.parameters()),62401)
        self.assertIsInstance(policy.cell,model.TypedLoop)
        source=["P : Prop\nh : P\n⊢ P",
                "a : Nat\n⊢ a = a"]
        logits=policy(model.graph_batch(source))
        self.assertEqual(tuple(logits.shape),(2,8))
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in policy.cell.parameters()))

    def test_all_eight_candidate_outcomes_are_train_labels(self):
        rows=[{"stage_id":"a","actual_lean_goal_state":"⊢ True"}]
        probes=[{"stage_id":"a","action":a,
                 "accepted":True,"progress":bool(i%2),
                 "closed":i==0} for i,a in enumerate(model.VOCAB)]
        labels,n=model._labels(rows,probes)
        self.assertEqual(tuple(labels.shape),(1,8))
        self.assertEqual(n,1)
        self.assertEqual(float(labels[0,0]),1.)
        self.assertEqual(float(labels[0,1]),0.5)
        self.assertEqual(float(labels[0,2]),0.)

    def test_original_source_proof_is_not_a_model_inference_fallback(self):
        text=("import Init\nnamespace Scratch\n"
              "theorem test (P : Prop) : P → P := by\n"
              "  intro h\n"
              "  exact h\n"
              "end Scratch\n")
        native=model.native_first_script(text,"test")
        self.assertIn("  first | omega",native)
        self.assertIn("  all_goals (first |",native)
        self.assertIn(" | fc11_ctor | intro",native)
        self.assertIn("import Init\n\nmacro",native)
        self.assertNotIn("  exact h\n",native)
        self.assertNotIn("sorry",native)
        with self.assertRaises(ValueError):
            scratch.make_source(text,"test",["sorry"],probe=False)


if __name__=="__main__":
    unittest.main()
