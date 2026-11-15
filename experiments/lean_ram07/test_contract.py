"""RAM-07 natural exact-source mining and immutable fresh-family contract."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import ram07_core as c
from collect import EXTRA_TRAIN_SOURCE_FILES
from evaluate import EXACT_CASES
from shape_frequency import Frequency,ShapeFrequency

class Protocol(unittest.TestCase):
    def test_preregistered_two_immutable_source_commits(self):
        config=json.loads((HERE/"config.json").read_text(encoding="utf8"))
        self.assertEqual(config["registration"],"e2a717aca927221f21fd3df9b47520b6f74f1f7c")
        self.assertEqual(config["source_train_commit"],"c84e2025334e9a967c1297425a490f98d5b3d189")
        self.assertEqual(config["source_test_commit"],"fca46bfb3430ac85ffca16f6a6a83bb0b0231e41")
        self.assertEqual(config["schema"],"RAM07-v1")
        self.assertEqual({k:tuple(v) for k,v in config["prospective"].items()},EXACT_CASES)
        self.assertEqual(len(EXACT_CASES),4)
        self.assertEqual(sum(map(len,EXACT_CASES.values())),12)
        self.assertTrue(set(EXACT_CASES).isdisjoint(EXTRA_TRAIN_SOURCE_FILES))
        self.assertEqual(len(EXTRA_TRAIN_SOURCE_FILES),4)
        self.assertEqual(config["additional_train_sources"],list(EXTRA_TRAIN_SOURCE_FILES))
        self.assertEqual(config["collector"]["depth"],5)
        self.assertEqual(config["collector"]["states_per_theorem"],12)
        self.assertEqual(config["primary"],dict(seconds=30,evaluations=48,
             transitions=8,request_seconds=3,max_goals=8))
        self.assertEqual(config["secondary"],dict(seconds=60,evaluations=128,
             transitions=16,request_seconds=3,max_goals=8))
        self.assertEqual(config["shape_frequency_prior_weight"],8)
        self.assertEqual(config["training"]["exact_positive_fraction"],.1)
        self.assertEqual(config["seeds"],[17,29,43])
        self.assertNotIn("curriculum",config)

    def test_named_intro_is_context_deterministic(self):
        x=dict(accepted=True,goals=["State : Type u\nh : True\n⊢ True"],definitions=[])
        y=dict(accepted=True,goals=[
            "State : Type u\nram07_arg0 : Nat\nh : True\n⊢ True"],definitions=[])
        scripts=c.candidates(x)
        self.assertEqual(len(c.FAMILIES),19)
        self.assertIn(("intro_named","intro ram07_arg0"),scripts)
        self.assertIn(("intro_named","intro ram07_arg1"),c.candidates(y))
        self.assertFalse(any(f=="intro_named" and ";" in a for f,a in scripts))

    def test_natural_hypothesis_application_generated_only_from_visible_local_context(self):
        goal="""State : Type u
Action : Type v
Obs : Type w
allowed : Action → Prop
step : Action → State → State
readout : State → Obs
s t r : State
hst : obsEqOn allowed step readout s t
htr : obsEqOn allowed step readout t r
ram07_arg0 : List Action
ram07_arg1 : ∀ (a : Action), a ∈ ram07_arg0 → allowed a
⊢ readout (runActions step ram07_arg0 s) = readout (runActions step ram07_arg0 r)"""
        scripts=c.candidates(dict(accepted=True,goals=[goal],definitions=[]))
        match="exact (hst ram07_arg0 ram07_arg1).trans (htr ram07_arg0 ram07_arg1)"
        self.assertIn(("exact_typed",match),scripts)
        self.assertTrue(c.valid_script(match))
        self.assertLessEqual(sum(f=="exact_typed" for f,a in scripts),24)
        self.assertTrue(all(c.valid_script(a) for f,a in scripts))

    def test_symmetric_natural_application_has_no_teacher_prefix(self):
        goal="""Action : Type u
State : Type v
allowed : Action → Prop
step : Action → State → State
readout : State → Nat
s t : State
h : obsEqOn allowed step readout s t
ram07_arg0 : List Action
ram07_arg1 : ∀ a, a ∈ ram07_arg0 → allowed a
⊢ readout (runActions step ram07_arg0 t) = readout (runActions step ram07_arg0 s)"""
        scripts=c.candidates(dict(accepted=True,goals=[goal],definitions=[]))
        self.assertIn(("exact_typed","exact (h ram07_arg0 ram07_arg1).symm"),scripts)
        self.assertNotIn(("exact_typed","exact (h word valid).symm"),scripts)

    def test_typed_exact_and_intro_strict_source_whitelist(self):
        for name in ["intro ram07_arg0","intro ram07_arg4","exact (h x y).symm",
                     "exact (a x y).trans (b x y)"]:
            self.assertTrue(c.valid_script(name))
        for bad in ["intro ram07_arg5","intro ram07_arg0; sorry",
            "exact (by sorry)","exact (h x y).trans (by sorry)",
            "exact h; sorry","exact MathCan.Cert.unknown","intro _"]:
            self.assertFalse(c.valid_script(bad),bad)

    def test_source_renderer_preserves_term_proof_statement_without_reference_term(self):
        original="""import Init

namespace MathCan
theorem plain_term (a b : Int) : a * b = b * a :=
  Int.mul_comm a b

theorem second (n : Nat) : n = n := by
  rfl
end MathCan
"""
        start,begin,end=c.original_theorem_bounds(original.splitlines(),"plain_term")
        self.assertEqual(original.splitlines()[start].split()[1],"plain_term")
        self.assertIn(":=",original.splitlines()[begin])
        self.assertGreater(end,begin)
        proof=c.make_source(original,"plain_term",["omega"],server=False)
        self.assertIn("theorem plain_term (a b : Int) : a * b = b * a := by",proof)
        self.assertIn("  omega",proof)
        self.assertNotIn("Int.mul_comm a b",proof)
        self.assertNotIn("theorem second",proof)
        self.assertIn("end MathCan",proof)
        original_tactic="""import Init

namespace MathCan
theorem plain_tactic (n : Nat) : n = n := by
  rfl
end MathCan
"""
        assert "theorem plain_tactic" in c.make_source(original_tactic,"plain_tactic",["rfl"])

    def test_strong_baselines_do_not_use_development_labels(self):
        root={"goals":["h : P\n⊢ P"],"definitions":[]}
        rows=[dict(before=root,development=False,family="exact_typed",label=3,delta=1),
              dict(before=root,development=False,family="omega",label=0,delta=0)]
        self.assertEqual(Frequency(rows).rank(root,[("omega","omega"),("exact_typed","exact h")])[0][0],"exact_typed")
        self.assertEqual(ShapeFrequency(rows,prior_weight=8).rank(root,[("omega","omega"),("exact_typed","exact h")])[0][0],"exact_typed")
        with self.assertRaises(AssertionError):
            Frequency(rows+[dict(rows[0],development=True)])

if __name__=="__main__":unittest.main()
