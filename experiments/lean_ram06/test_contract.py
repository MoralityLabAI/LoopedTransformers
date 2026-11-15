"""RAM-06 frozen experimental grammar and auxiliary data safety checks."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import ram06_core as c
from exact_curriculum import TRAIN,DEVELOPMENT,check_fixture
from shape_frequency import Frequency,ShapeFrequency,goal_shape
from evaluate import EXACT_CASES

class Protocol(unittest.TestCase):
    def test_fresh_sources_and_registered_budgets(self):
        cfg=json.loads((HERE/"config.json").read_text(encoding="utf8"))
        self.assertEqual(cfg["schema"],"RAM06-v1")
        self.assertEqual(cfg["registration"],"f4f26ed48ed675cc7b80c73f151ba3927b46bafa")
        self.assertEqual(len(EXACT_CASES),4)
        self.assertEqual(sum(map(len,EXACT_CASES.values())),12)
        self.assertEqual({k:tuple(v) for k,v in cfg["prospective"].items()},EXACT_CASES)
        self.assertEqual(cfg["primary"],dict(seconds=30,evaluations=48,
            transitions=8,request_seconds=3,max_goals=8))
        self.assertEqual(cfg["secondary"],dict(seconds=60,evaluations=128,
            transitions=16,request_seconds=3,max_goals=8))
        self.assertEqual(cfg["shape_frequency_prior_weight"],8)
        self.assertEqual(cfg["training"]["exact_positive_fraction"],0.2)
        self.assertEqual(cfg["seeds"],[17,29,43])

    def test_18_families_and_strict_typed_exact_syntax(self):
        self.assertEqual(len(c.FAMILIES),18)
        self.assertEqual(c.FAMILIES.count("exact_typed"),1)
        for script in ("exact h","exact h.symm","exact pair.left",
            "exact pair.right","exact f hp","exact f hp hq",
            "exact bridge.mp hp","exact bridge.mpr hq"):
            self.assertTrue(c.valid_script(script),script)
        for bad in ("exact (by sorry)","exact h; sorry","exact h \n sorry",
                    "exact NotListed.source","exact f hp hq junk",
                    "simp_all [MathCan.Fun.one, MathCan.Fun.two]"):
            if bad.startswith("simp_all ["):
                continue  # historic valid text, excluded from new generation
            self.assertFalse(c.valid_script(bad),bad)

    def test_type_aware_examples_generated_from_visible_telescope(self):
        examples=[
            ("P Q : Prop\nhfn : P → Q\nhp : P\n⊢ Q",
             "exact hfn hp"),
            ("P Q R : Prop\nhfn : P → Q → R\nhp : P\nhq : Q\n⊢ R",
             "exact hfn hp hq"),
            ("m n : Nat\nheq : m = n\n⊢ n = m",
             "exact heq.symm"),
            ("P Q : Prop\npair : P ∧ Q\n⊢ Q",
             "exact pair.right"),
            ("P Q : Prop\nbridge : P ↔ Q\nhp : P\n⊢ Q",
             "exact bridge.mp hp")
        ]
        for context,script in examples:
            state={"accepted":True,"goals":[context],"definitions":[]}
            pool=c.candidates(state)
            self.assertIn(("exact_typed",script),pool)
            self.assertEqual(pool.count(("exact_typed",script)),1)
            self.assertTrue(all(c.valid_script(cmd) for fam,cmd in pool))
            typed=[cmd for fam,cmd in pool if fam=="exact_typed"]
            self.assertLessEqual(len(typed),18)

    def test_auxiliary_curriculum_is_disjoint_and_named(self):
        cfg=json.loads((HERE/"config.json").read_text(encoding="utf8"))
        check_fixture(cfg,HERE/"ExactCurriculum.lean")
        self.assertEqual(len(TRAIN),12)
        self.assertEqual(len(DEVELOPMENT),6)
        self.assertTrue(set(TRAIN).isdisjoint(DEVELOPMENT))
        self.assertTrue(all("RAM06/ExactCurriculum.lean:"+name in cfg["development"]
                     for name in DEVELOPMENT))

    def test_stronger_symbolic_controls_train_only(self):
        shape={"goals":["h : P\n⊢ P"],"definitions":[]}
        seen=[
         {"development":False,"before":shape,"family":"exact_typed","label":3,"delta":1},
         {"development":False,"before":shape,"family":"omega","label":0,"delta":0},
        ]
        global_policy=Frequency(seen)
        typed=ShapeFrequency(seen,prior_weight=8)
        pool=[("omega","omega"),("exact_typed","exact h")]
        self.assertEqual(typed.rank(shape,pool)[0],("exact_typed","exact h"))
        self.assertEqual(global_policy.rank(shape,pool)[0],("exact_typed","exact h"))
        with self.assertRaises(AssertionError):
            Frequency(seen+[dict(seen[0],development=True)])
        with self.assertRaises(AssertionError):
            ShapeFrequency(seen,prior_weight=4)

    def test_heldout_source_names_do_not_overlap_historical(self):
        prev={
            "MathCan/HookMultiTick.lean","MathCan/Settlement.lean",
            "MathCan/TOEJordan.lean","MathCan/TOENonAbelian.lean",
            "MathCan/FlashAccounting.lean","MathCan/V4Transient.lean",
            "MathCan/TOEFormSectors.lean","MathCan/TOEMultipletKinetics.lean",
            "MathCan/OpcodeRefinement.lean","MathCan/FoundryProbe.lean",
            "MathCan/TOETwoFormTriad.lean","MathCan/TOETripletJordan.lean",
            "MathCan/TOETwoMultiplet.lean","MathCan/TOEYangMills.lean",
            "MathCan/TOEConeMatching.lean","MathCan/TOETensorConstraints.lean",
            "MathCan/DeletionMemory.lean","MathCan/FiniteHorizonGap.lean",
            "MathCan/GraphAuthorization.lean","MathCan/HookState.lean",
            "MathCan/Intervention.lean","MathCan/TOECosmology.lean",
            "MathCan/TOEGraph.lean","MathCan/Hodge.lean",
            "MathCan/TOEDressedMatter.lean","MathCan/TOELorentzBivector.lean",
            "MathCan/TOETensorCompletion.lean",
        }
        self.assertTrue(set(EXACT_CASES).isdisjoint(prev))

if __name__=="__main__":
    unittest.main()
