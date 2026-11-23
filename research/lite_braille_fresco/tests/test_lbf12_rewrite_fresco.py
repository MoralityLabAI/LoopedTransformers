"""No proof labels or future candidate menus enter a recurrent decision fresco."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("optional CPU Torch is absent from the standard-library codec job")

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lbf12_rewrite_fresco import capture,observe
from lbf_gym import FrescoEnv,encode,SCHEMA_LABEL
from ppt_gym.core import Config,PlasticProofTransformer


class DecisionFrescoTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11);torch.set_num_threads(1)
        self.model=PlasticProofTransformer(Config(plan="b",width=16,heads=4,max_bytes=384)).eval()
        self.case=SimpleNamespace(root="engineering_fixture",lineage="fixture_lineage")
        self.packet="E|S:aabc|T:abcc|B:16|0:ab=bc"
        self.menu=["r0:0:1:abcc","r1:1:0:bbbc"]

    def test_actual_boundaries_and_hook_removal(self):
        for mode in ("normal","zero","roll"):
            out,bank=capture(self.model,self.packet,self.menu,8,mode)
            self.assertEqual(len(out),2)
            self.assertEqual(float(bank["gate_a"][0]["bank_in"].abs().max()),0)
            self.assertFalse(self.model.gate_a._forward_hooks)
            self.assertFalse(self.model.gate_b.gate._forward_hooks)

    def test_current_menu_only_and_prefix_clock(self):
        ob,measurements,_=observe(self.model,self.case,self.packet,self.menu,8,"0"*64,1)
        self.assertEqual(ob["candidates"],self.menu)
        self.assertEqual(len(measurements),2)
        self.assertEqual([v["pass"] for v in ob["visits"]],[1,2])
        self.assertNotIn("labels",ob)
        encoded=encode(ob)
        private={"schema":SCHEMA_LABEL,"root_id":ob["root_id"],"lineage":ob["lineage"],
                 "evidence_class":"externally_attested_native",
                 "outcomes":dict.fromkeys(self.menu,"censored")}
        env=FrescoEnv(encoded,private)
        self.assertEqual(len(env.observation()["visits"]),1)
        self.assertEqual(env.observation()["candidates"],self.menu)
        self.assertNotIn("outcomes",env.observation())
        env.step("continue")
        self.assertEqual(len(env.observation()["visits"]),2)
        self.assertEqual(env.step("select",self.menu[0])["status"],"censored")

    def test_deterministic_public_input(self):
        a,_,_=observe(self.model,self.case,self.packet,self.menu,8,"0"*64,1)
        b,_,_=observe(self.model,self.case,self.packet,self.menu,8,"0"*64,1)
        self.assertEqual(a,b)
        c,_,_=observe(self.model,self.case,self.packet,[self.menu[0]],8,"0"*64,2)
        self.assertEqual(c["candidates"],[self.menu[0]])
        self.assertNotEqual(a["root_id"],c["root_id"])


if __name__=="__main__":unittest.main()
