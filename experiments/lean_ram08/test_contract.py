"""RAM-08 preregistered source split, original tactic grammar and baseline tests."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import ram08_core as core
from collect import EXTRA_TRAIN_SOURCE_FILES
from evaluate import EXACT_CASES
from shape_frequency import Frequency,ShapeFrequency

class RAM08Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((HERE/"config.json").read_text(encoding="utf8"))

    def test_exact_immutable_mathcan_revision_pair(self):
        c=self.cfg
        self.assertEqual(c["schema"],"RAM08-v1")
        self.assertEqual(c["registration"],"b8001e50e89401b44c8acb276f4c15e02c4b5b05")
        self.assertEqual(c["source_train_commit"],"c84e2025334e9a967c1297425a490f98d5b3d189")
        self.assertEqual(c["source_test_commit"],"83f61997948818b68da94429c2fe736d93ea3093")
        self.assertEqual(c["candidate_families"],20)
        self.assertEqual({f:tuple(v) for f,v in c["prospective"].items()},EXACT_CASES)
        self.assertEqual(sum(map(len,EXACT_CASES.values())),12)
        self.assertEqual(len(EXACT_CASES),4)
        self.assertTrue(all(f not in EXTRA_TRAIN_SOURCE_FILES for f in EXACT_CASES))

    def test_training_original_source_roots_and_hard_diversity_gate(self):
        c=self.cfg
        self.assertEqual(len(c["additional_explicit_train_roots"]),18)
        self.assertEqual(len(set(c["additional_explicit_train_roots"])),18)
        self.assertEqual(c["natural_exact_minimum_distinct_train_roots"],2)
        self.assertEqual(c["natural_exact_minimum_distinct_positive_actions"],3)
        self.assertEqual(c["collector"]["states_per_theorem"],18)
        self.assertEqual(c["collector"]["depth"],6)
        self.assertEqual(c["collector"]["max_goals"],8)
        self.assertEqual(c["training"]["exact_positive_fraction"],0.1)
        self.assertEqual(c["seeds"],[17,29,43])
        self.assertTrue(all(item.startswith("MathCan/") and ".lean:" in item
                        for item in c["additional_explicit_train_roots"]))
        self.assertTrue(all(item not in c["development"]
                        for item in c["additional_explicit_train_roots"]))

    def test_budgets_and_frequency_prior_frozen(self):
        c=self.cfg
        self.assertEqual(c["primary"],{"seconds":30,"evaluations":48,
            "transitions":8,"request_seconds":3,"max_goals":8})
        self.assertEqual(c["secondary"],{"seconds":60,"evaluations":128,
            "transitions":16,"request_seconds":3,"max_goals":8})
        self.assertEqual(c["shape_frequency_prior_weight"],8)

    def test_core_decide_shared_with_all_original_families(self):
        self.assertEqual(len(core.FAMILIES),20)
        self.assertEqual(core.FAMILIES.count("decide"),1)
        self.assertLess(core.FAMILIES.index("rfl"),core.FAMILIES.index("decide"))
        self.assertLess(core.FAMILIES.index("decide"),core.FAMILIES.index("simp_all"))
        self.assertTrue(core.valid_script("decide"))
        state=dict(accepted=True,goals=["x : Nat\nh : x = x\n⊢ x = x"],definitions=[])
        all_actions=core.candidates(state)
        self.assertIn(("decide","decide"),all_actions)
        self.assertIn(("intro_named","intro ram07_arg0"),all_actions)
        self.assertIn(("exact_typed","exact h"),all_actions)
        self.assertTrue(all(core.valid_script(action) for fam,action in all_actions))
        self.assertLessEqual(sum(f=="exact_typed" for f,a in all_actions),24)

    def test_registered_symbolic_controls_require_training_only(self):
        rows=[
          {"development":False,"family":"decide","label":3,"delta":1,
           "before":{"goals":["⊢ True"]}},
          {"development":False,"family":"exact_typed","label":0,"delta":0,
           "before":{"goals":["⊢ True"]}},
        ]
        native=[("exact_typed","exact h"),("decide","decide")]
        ranked=Frequency(rows).rank({"goals":["⊢ True"]},native)
        self.assertEqual(ranked[0],("decide","decide"))
        ranked=ShapeFrequency(rows,prior_weight=8).rank({"goals":["⊢ True"]},native)
        self.assertEqual(ranked[0],("decide","decide"))
        with self.assertRaises(AssertionError):
            Frequency(rows+[dict(rows[0],development=True)])
        with self.assertRaises(AssertionError):
            ShapeFrequency(rows,prior_weight=4)

    def test_strict_source_based_actions_preclude_shortcuts(self):
        for action in ("sorry","exact (by sorry)","admit","intro foo","decide; sorry",
                       "exact False.elim (by contradiction)","exact h; rfl"):
            self.assertFalse(core.valid_script(action),action)

if __name__=="__main__":
    unittest.main()
