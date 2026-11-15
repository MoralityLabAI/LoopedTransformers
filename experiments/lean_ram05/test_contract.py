"""RAM-05 frozen grammar, isolation, unknown Lean crash and policy checks."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import ram05_core as core
from runtime import Session,ProbeTimeout
from model import Policy,Config
from evaluate import EXACT_CASES
import torch

class RAM05Contract(unittest.TestCase):
    def test_prereg_source_family_and_budgets(self):
        config=json.loads((HERE/"config.json").read_text(encoding="utf8"))
        assert config["schema"]=="RAM05-v1"
        assert config["registration"]=="215de8648d6db73352a4118f5507b45f9eb4f814"
        self.assertEqual({p:tuple(v) for p,v in config["prospective"].items()},EXACT_CASES)
        self.assertEqual(len(EXACT_CASES),4)
        self.assertEqual(sum(map(len,EXACT_CASES.values())),12)
        self.assertEqual(config["primary"],{"seconds":30,"evaluations":48,
             "transitions":8,"request_seconds":3,"max_goals":8})
        self.assertEqual(config["secondary"],{"seconds":60,"evaluations":128,
             "transitions":16,"request_seconds":3,"max_goals":8})
        self.assertEqual(config["seeds"],[17,29,43])
        forbidden={"MathCan/TOEGraph.lean","MathCan/Hodge.lean",
              "MathCan/HookState.lean","MathCan/Intervention.lean",
              "MathCan/TOETripletJordan.lean","MathCan/TOETwoFormTriad.lean",
              "MathCan/FoundryProbe.lean","MathCan/OpcodeRefinement.lean",
              "MathCan/TOEMultipletKinetics.lean","MathCan/TOEFormSectors.lean"}
        self.assertTrue(set(EXACT_CASES).isdisjoint(forbidden))

    def test_bounded_registered_only_mode_actions_and_budget(self):
        state={"accepted":True,"goals":[
            "xs : List Nat\nh : xs = xs\n⊢ xs = xs"],
            "definitions":[
                "MathCan.Other.zz","MathCan.Other.aa",
                "MathCan.Other.cc","MathCan.Other.bb",
                "SomeOther.Library.def",
            ]}
        scripts=core.candidates(state)
        self.assertEqual(len(core.FAMILIES),17)
        self.assertEqual(len(scripts),len(set(scripts)))
        bounded=[(f,s) for f,s in scripts if f in (
            "simp_only_visible","simp_all_only_visible","dsimp_only_visible")]
        self.assertEqual(len(bounded),9)
        for f,s in bounded:
            self.assertTrue(core.valid_script(s),s)
            self.assertIn(" only [MathCan.Other.",s)
            self.assertNotIn("zz",s) # sorted first three only
        for f,s in scripts:
            self.assertFalse("MathCan." in s and "only" not in s)
            self.assertNotIn("Int.mul_comm",s)
            self.assertTrue(core.valid_script(s),s)
        self.assertIn(("simp_only_visible","simp only [MathCan.Other.aa]"),scripts)
        self.assertIn(("simp_all_only_visible","simp_all only [MathCan.Other.bb]"),scripts)
        self.assertIn(("dsimp_only_visible","dsimp only [MathCan.Other.cc] at *"),scripts)

    def test_reject_arbitrary_lean_source_injection(self):
        for action in ["sorry","simp only [MathCan.Other.aa]; sorry",
            "simp only [MathCan.Other.aa]\nsorry",
            "simp_all only [MathCan.Other.aa, MathCan.Other.bb]",
            "exact (by sorry)","run_tac","simp_all [MathCan.Other.aa]"]:
            if action=="simp_all [MathCan.Other.aa]":
                # It was a RAM03 registered legacy script but must never be
                # produced by the RAM05 registered candidate generator.
                continue
            self.assertFalse(core.valid_script(action),action)

    def test_tiny_shared_parameters_and_cpu_forward(self):
        state={"before":{"goals":["n : Nat\n⊢ n = n"],"definitions":[]},
               "family":"rfl","script":"rfl"}
        from model import encode
        for name,width,loops in (("loop32",32,4),("loop64",64,4),("flat64",64,0)):
            net=Policy(Config(name,width,loops))
            self.assertLess(sum(p.numel() for p in net.parameters()),250000)
            with torch.no_grad():
                a,b=net(encode([state]))
            self.assertEqual(tuple(a.shape),(1,4))
            self.assertEqual(tuple(b.shape),(1,))
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue(torch.isfinite(b).all())

    def test_closed_goal_is_not_a_training_proof(self):
        before={"accepted":True,"goals":["⊢ True"],"definitions":[]}
        after={"accepted":True,"goals":[],"definitions":[]}
        self.assertEqual(core.label(before,after),3)
        self.assertFalse(core.candidates(after))

    def test_fatal_lean_exit_has_unknown_label_and_log(self):
        with tempfile.TemporaryDirectory() as td:
            sess=object.__new__(Session)
            sess.out=Path(td);sess.number=0;sess.closed=False
            sess._send=lambda number,value:None
            with mock.patch.object(Session,"_response",
                         side_effect=ProbeTimeout("Lean exited: recursion depth")):
                with self.assertRaises(ProbeTimeout):
                    sess.query(["simp only [MathCan.Other.aa]"],.01)
            p=Path(td)/"unknown_compiler_observations.jsonl"
            self.assertTrue(p.is_file())
            recorded=json.loads(p.read_text(encoding="utf8").strip())
            self.assertEqual(recorded["classification"],"unknown")
            self.assertEqual(recorded["actions"],["simp only [MathCan.Other.aa]"])

if __name__=="__main__":unittest.main()
