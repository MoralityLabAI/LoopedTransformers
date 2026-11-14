"""FC-12 registered original TypedLoop and exact typed-tactic inference guards."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc12_policy as experiment
from research.refutation_bench.fc12 import core,scratch


class FC12PolicyTests(unittest.TestCase):
    def test_action_family_and_lean_source_pins(self):
        self.assertEqual(experiment.REGISTERED_PREREG,
                         "d6fae0ddb9ce60036cff09e5ec93820ed3879af8")
        self.assertEqual(len(core.FAMILIES),12)
        self.assertEqual(core.MAX_ARGUMENTS,4)
        self.assertEqual(core.MAX_TOTAL_LEAN_PROBES,12)
        root=os.getenv("FC12_TRAIN_ROOT")
        if root:
            self.assertFalse((Path(root)/"research/refutation_bench/fc12/test_data.py").exists())

    def test_exact_original_typedloop_parameter_count_and_shared_gradients(self):
        torch.set_num_threads(2)
        x=experiment.FC12Loop()
        self.assertIsInstance(x.cell,experiment.TypedLoop)
        self.assertEqual(sum(p.numel() for p in x.parameters()),62401)
        inputs=experiment.graph(["P : Prop\nh : P\n⊢ P","x : Nat\n⊢ x = x"])
        scores=x(inputs)
        self.assertEqual(tuple(scores.shape),(2,12))
        scores.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in x.cell.parameters()))

    def test_long_real_lean_state_retains_turnstile_and_parameter_symbols(self):
        goal=" ".join(f"h{i} : Nat" for i in range(120)) + "\nxs : List Nat\n⊢ xs = xs"
        x=experiment.graph([goal])
        self.assertEqual(tuple(x["tokens"].shape),(1,112))
        self.assertEqual(float(x["head"].sum()),12.)
        self.assertIn("induction xs",core.generate(goal)["induction_list"])

    def test_dynamic_candidates_are_all_context_derived_and_safe(self):
        state=("xs : List Nat\nh : xs = xs\n"
               "witness : ∃ (n : Nat), n > 0\n⊢ xs = xs")
        choices=list(experiment.policy_candidates(state,list(core.FAMILIES)))
        self.assertTrue(choices)
        self.assertTrue(all(scratch.valid_script(s) for _,s in choices))
        self.assertIn(("rw_local","rw [h]"),choices)
        self.assertIn(("induction_list","induction xs"),choices)
        self.assertIn(("rcases_local",
                "rcases witness with ⟨fc12_left, fc12_right⟩"),choices)

    def test_source_only_label_matrix_never_uses_reference_tactic(self):
        state=[{"stage_id":"a","actual_lean_goal_state":"h : True\n⊢ True"}]
        records=[
            {"stage_id":"a","family":core.FAMILIES[i],
             "script":core.FAMILIES[i],"progress":i==0}
            for i in range(6)]
        y,mask=experiment.build_train_matrix(state,records,core.FAMILIES)
        self.assertEqual(tuple(y.shape),(1,12))
        self.assertEqual(float(y.sum()),1.)
        self.assertEqual(float(mask.sum()),6.)

    def test_sorry_free_final_and_opaque_tactic_rejected(self):
        src=("import Init\nnamespace Test\n"
             "theorem trivial (P : Prop) (h : P) : P := by\n"
             "  exact h\nend Test\n")
        final=scratch.make_source(src,"trivial",["exact h"],probe=False)
        self.assertNotIn("sorry",final)
        with self.assertRaises(ValueError):
            scratch.make_source(src,"trivial",["exact h\nsorry"],probe=False)

    def test_native_parameter_search_proves_original_theorem_under_lean(self):
        import shutil
        if not shutil.which("lean"):
            self.skipTest("lean4.35 not available")
        goal="P : Prop\nh : P\n⊢ P"
        src=("import Init\nnamespace Test\n"
             "theorem trivial (P : Prop) (h : P) : P := by\n"
             "  exact h\nend Test\n")
        r=experiment.run_closed_loop(src,"trivial",goal,model=None,allowed_ms=15000.)
        self.assertTrue(r["accepted"],r)
        self.assertTrue(r["sorry_free_release"])
        self.assertGreaterEqual(r["candidate_probe_count"],1)
        self.assertLessEqual(r["candidate_probe_count"],12)


if __name__=="__main__":
    unittest.main()
