"""Independent pre-unseal tests. No ARG03 test.jsonl is opened."""
from __future__ import annotations
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import unittest

from mechinterp.dark_fiber_gyms.arg03_preunseal.audit import (
    DEV, GRAM, audit, paired)
from mechinterp.dark_fiber_gyms.arg03_preunseal.calibration import (
    DevRankRow, fit_position_operator_bias, program_scores,
    rank_of_gold, rank_summary, PROGRAMS)
from mechinterp.dark_fiber_gyms.arg03_preunseal.prepare_final_r import make_fresh


class TestDevelopmentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev=json.loads(DEV.read_text())
        cls.gram=json.loads(GRAM.read_text())
        cls.r=audit(cls.dev,cls.gram,source_path=DEV)

    def test_source_provenance_and_strict_split(self):
        self.assertFalse(self.r["heldout_test_loaded"])
        self.assertEqual(self.r["source_dev_blob_sha"],"ae0db0293b5d85593eb1f61f5baa746223f4eed5")
        self.assertEqual(len(self.dev["rows"]),96)
        with self.assertRaises(ValueError):
            audit({**self.dev,"split":"test"},self.gram,source_path=DEV)
        with self.assertRaises(ValueError):
            audit(self.dev,self.gram,source_path=Path("runs/dark-fibers/arg03/public/test.jsonl"))
        with self.assertRaises(ValueError):
            audit({**self.dev,"ranking_list_available":True},self.gram,source_path=DEV)

    def test_depth_non_nestedness(self):
        q=self.r["arg03"]["depth"]
        self.assertEqual({d:x["future_correct"] for d,x in q["by_depth"].items()},
                         {"1":26,"2":42,"4":43,"8":42})
        self.assertEqual({d:x["top1_future_correct"] for d,x in q["by_depth"].items()},
                         {"1":7,"2":5,"4":6,"8":5})
        self.assertEqual(q["correct_all_2_4_8"],13)
        self.assertEqual(q["correct_any_2_4_8"],72)
        self.assertEqual(q["pairs"]["1_vs_2"]["both"],1)
        self.assertEqual(q["pairs"]["1_vs_2"]["flips"],66)
        self.assertAlmostEqual(q["pairs"]["1_vs_2"]["phi"],-.4902325224,places=7)
        self.assertEqual(q["pairs"]["2_vs_8"]["flips"],58)
        self.assertEqual(q["by_depth"]["2"]["distinct_baseline_top1_programs"],2)
        self.assertEqual(q["by_depth"]["4"]["distinct_baseline_top1_programs"],10)
        self.assertEqual(q["by_depth"]["8"]["distinct_baseline_top1_programs"],5)
        self.assertIsNone(q["full_top12_distinct_programs"])
        self.assertIsNone(q["target_final_operator_by_case"])

    def test_h2a_state_magnitude_confound(self):
        r=self.r["arg03"]["displacement"]["2"]
        self.assertAlmostEqual(r["mean_state_displacement_ratio_swap_over_random"],1.8491907478,places=7)
        self.assertAlmostEqual(r["mean_readout_l2_per_state_l2_swap"],4.4564573726,places=6)
        self.assertAlmostEqual(r["mean_readout_l2_per_state_l2_random"],6.7534199910,places=6)
        self.assertEqual(r["swap_larger_per_unit_cases"],4)
        self.assertTrue(all(self.r["arg03"]["displacement"][d]["mean_readout_l2_per_state_l2_random"] >
                            self.r["arg03"]["displacement"][d]["mean_readout_l2_per_state_l2_swap"]
                            for d in ("2","4","8")))

    def test_gramian_displays_robust_strata(self):
        g=self.r["loopscope04"]["gramian"]["gaoxin492"]
        self.assertEqual({s:x["n"] for s,x in g.items()},
                         {"WB-Prop":40,"WB-Stall_partial":40,"S4_stalled":40})
        self.assertAlmostEqual(g["WB-Prop"]["median_lambda1"],32.18,delta=.02)
        self.assertAlmostEqual(g["WB-Stall_partial"]["median_lambda1"],49.34,delta=.02)
        self.assertAlmostEqual(g["S4_stalled"]["median_lambda1"],5.136e6,delta=2000)
        self.assertGreater(g["WB-Prop"]["top3_fraction_of_lambda1_sum"],.88)
        self.assertGreater(g["WB-Stall_partial"]["top3_fraction_of_lambda1_sum"],.87)
        self.assertEqual(self.gram["sampled_tangent_dimension"],8)
        self.assertEqual(self.gram["ambient_kernel_dimension"],501)

    def test_boolean_phi_ties(self):
        self.assertEqual(paired([True,False],[True,False])["flips"],0)
        with self.assertRaises(ValueError):
            paired([True],[1])
        self.assertIsNone(paired([True,True],[True,True])["phi"])


class TestFreshFinalR(unittest.TestCase):
    def test_synthetic_generator_divides_program_roots_without_sealed_test(self):
        class DummyCase:
            def __init__(self, identifier, graph, family):
                self.identifier=identifier;self.source_graph=graph
                self.family=family
                self.signature=sha256(identifier.encode()).hexdigest()
        class MockOracle:
            CAL_FAMILIES=("signed","product","sparse")
            PROGRAMS=tuple(product("INSR",repeat=3))
            TOPOLOGIES=("path","diamond")
            def roster(self): return {"calibration":[]}
            def make_case(self,rng,*,identifier,graph,family,**kw):
                return DummyCase(identifier,graph,family)
        class MockARG02:
            def dev_cases(self):return []
            def shot_cases(self):return []
        groups,programs=make_fresh(MockOracle(),MockARG02(),set())
        self.assertEqual({k:len(v) for k,v in groups.items()},
                         {"development_finalR_fit":48,"development_finalR_valid":48})
        self.assertTrue(set(programs["development_finalR_fit"]).isdisjoint(
                             programs["development_finalR_valid"]))
        self.assertTrue(all(c.source_graph[2]=="R" for arr in groups.values() for c in arr))
        self.assertTrue(set(x.signature for x in groups["development_finalR_fit"]).isdisjoint(
                             x.signature for x in groups["development_finalR_valid"]))

    def test_biased_source_program_rank_and_training_split_gate(self):
        raw=tuple(tuple((0.,0.,0.,0.) for _ in range(3)) for _ in range(16))
        row=DevRankRow("case1","development_finalR_fit",raw,3)
        default=[[0.]*4 for _ in range(3)]
        self.assertEqual(rank_of_gold(program_scores(row,default),3),4)
        fit=fit_position_operator_bias([row],steps=90,lr=.15,ridge=.01)
        self.assertEqual(fit["stored_parameters"],12)
        self.assertEqual(fit["effective_degrees_of_freedom"],9)
        self.assertEqual(rank_of_gold(program_scores(row,fit["parameters_3x4"]),3),1)
        with self.assertRaises(ValueError):
            fit_position_operator_bias([DevRankRow("v","development_finalR_valid",raw,3)])

class TestNewDevRGrader(unittest.TestCase):
    def test_unseen_program_split_rank_and_source_hash(self):
        from mechinterp.dark_fiber_gyms.arg03_preunseal.grade_final_r import run
        logits=[[[0.,0.,0.,0.] for _ in range(3)] for _ in range(16)]
        base={str(t):[0.]*64 for t in (1,2,4,8)}
        pub_fit=[{"case_id":"f","public_sha256":"sha_fit"}]
        pub_valid=[{"case_id":"v","public_sha256":"sha_valid"}]
        host=[{"case_id":"f","source_sha256":"sha_fit",
               "split":"development_finalR_fit","gold_program_index":3},
              {"case_id":"v","source_sha256":"sha_valid",
               "split":"development_finalR_valid","gold_program_index":7}]
        tape_fit=[{"case_id":"f","public_sha256":"sha_fit",
                   "loop1_operator_site_logits":logits,"baseline":base}]
        tape_valid=[{"case_id":"v","public_sha256":"sha_valid",
                     "loop1_operator_site_logits":logits,"baseline":base}]
        report=run(pub_fit,pub_valid,host,tape_fit,tape_valid)
        self.assertFalse(report["sealed_test_loaded"])
        self.assertEqual(report["fit"]["n"],1)
        self.assertEqual(report["valid"]["n"],1)
        self.assertEqual(report["fit"]["baseline"]["1"]["gold_rank"][0],4)
        self.assertEqual(report["valid"]["baseline"]["1"]["gold_rank"][0],8)
        self.assertIn("gold_rank",report["valid"]["bias12"])
        with self.assertRaises(ValueError):
            run(pub_fit,[{"case_id":"v","public_sha256":"bad"}],
                host,tape_fit,tape_valid)


class TestNewH2aArm(unittest.TestCase):
    def test_per_depth_random_displacement_and_current_readout_parity(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch required only for prospective H2a arm")
        from mechinterp.dark_fiber_gyms.arg03_preunseal.controls import match_random_direction
        clean=torch.zeros(3,16,5)
        swap=clean.clone()
        swap[:,2:10]=2.
        direction=torch.ones_like(clean)
        out,cert=match_random_direction(clean,swap,direction,prefix_width=8,scored_tail=3)
        self.assertTrue(torch.equal(out[:,-3:],clean[:,-3:]))
        self.assertLess(cert["max_relative_norm_error"],1e-6)
        self.assertEqual(cert["intervention"],"time_local_depth_t_new_arm")
        with self.assertRaises(ValueError):
            match_random_direction(clean,swap,torch.zeros_like(clean))
        with self.assertRaises(ValueError):
            match_random_direction(clean,swap,direction,prefix_width=14,scored_tail=3)

if __name__=="__main__":unittest.main()
