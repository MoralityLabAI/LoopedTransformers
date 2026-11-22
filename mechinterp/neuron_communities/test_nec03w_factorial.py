"""NEC-03W synthetic-only functional redundancy and physical-sham tests.

The toy OR-model is deliberately known to have two redundant useful pairs:
each individual pair can be lost while joint loss breaks the function.
Every test is generated locally, using no original source logits or labels.
"""
from __future__ import annotations

import copy
import unittest

from mechinterp.neuron_communities.nec03w_factorial import (
    discover_candidates,quartet_partitions,matched_pair_geometry,
    factorial_metrics,select_from_calibration,
)
from mechinterp.neuron_communities.nec03w_independent import verify_pair_geometry


def toy_decoder():
    cols=[[0.0]*32 for _ in range(128)]
    for i in range(16):
        cols[i][i//2]=1.0
    return cols


def toy_pairs():
    return matched_pair_geometry(
        [[i,1.0] for i in range(16)],toy_decoder(),
        (0,1,2,3),[[0,1],[2,3]],
        root=0,arm="composed",group=24,mask=0,visit=0,trials=512)


def toy_case(redundant=True,*,admissible=True,delayed=True):
    base=0
    modes={
        "baseline":base,
        "remove_A":0 if redundant else 1,
        "remove_B":0 if redundant else 1,
        "remove_AB":1,
        "sham_C":0,"sham_D":0,"sham_CD":0,
        "late_target_rescue":0 if delayed else 1,
        "late_matched_sham_rescue":1,
        "same_visit_exact_addback":base,
    }
    if not admissible:
        modes={name:base for name in modes}
    return {"eligible":admissible,"delayed_rescue_eligible":delayed,"errors":modes}


class FunctionalComplementarityTests(unittest.TestCase):
    def test_training_only_candidate_discovery_does_not_rank_from_query(self):
        rows=[list(range(16)) for _ in range(1280)]
        chosen=discover_candidates(rows,budget=12)
        self.assertEqual(len(chosen),12)
        self.assertEqual(chosen[0]["source_discovery_count"],512)
        self.assertEqual(chosen[0]["source_query_ID_support_only"],256)
        edited=copy.deepcopy(rows)
        edited[1024:]=[list(range(16,32)) for _ in range(256)]
        other=discover_candidates(edited,budget=12)
        self.assertEqual(
            [(x["quartet"],x["source_discovery_count"]) for x in chosen],
            [(x["quartet"],x["source_discovery_count"]) for x in other],
        )
        self.assertEqual(other[0]["source_query_ID_support_only"],0)

    def test_three_partitions_exhaust_four_code_2x2_pairs(self):
        self.assertEqual(quartet_partitions((0,1,2,3)),[
            [[0,1],[2,3]],[[0,2],[1,3]],[[0,3],[1,2]]
        ])

    def test_physically_matched_disjoint_pair_sham_is_reconstructed(self):
        r=toy_pairs()
        self.assertTrue(r["admissible"],r)
        self.assertEqual(len(set(
            r["pair_A"]+r["pair_B"]+
            r["matched_control_C"]+r["matched_control_D"]
        )),8)
        self.assertLessEqual(r["pair_internal_cosine_gap"],.4)
        self.assertLessEqual(r["max_pair_subgroup_norm_ratio_gap"],.35)
        audit=verify_pair_geometry([[i,1.0] for i in range(16)],toy_decoder(),r)
        self.assertEqual(audit["physical_coordinates_recomputed"],192)
        self.assertEqual(audit["new_model_forwards"],0)

    def test_inactive_feature_or_protected_world_cannot_enter_sham_search(self):
        r=matched_pair_geometry([[i,1.0] for i in range(3,19)],
            toy_decoder(),(0,1,2,3),[[0,1],[2,3]],
            root=0,arm="composed",group=24,mask=0,visit=0)
        self.assertFalse(r["admissible"])
        with self.assertRaises(PermissionError):
            matched_pair_geometry([[i,1.0] for i in range(16)],
                toy_decoder(),(0,1,2,3),[[0,1],[2,3]],
                root=9,arm="composed",group=24,mask=0,visit=0)

    def test_independent_vector_auditor_rejects_tampered_controls(self):
        row=toy_pairs()
        row["projected32"]["CD"][0]+=0.05
        with self.assertRaisesRegex(AssertionError,"32D source pair differs"):
            verify_pair_geometry([[i,1.0] for i in range(16)],toy_decoder(),row)

    def test_redundant_two_pair_function_is_detected_against_matched_sham(self):
        rows=[toy_case() for _ in range(256)]
        result=factorial_metrics(rows)
        self.assertEqual(result["group_joint_error_excess"],1.0)
        self.assertEqual(result["matched_sham_joint_error_excess"],0.0)
        self.assertEqual(result["factorial_group_minus_sham_interaction"],1.0)
        self.assertEqual(result["delayed_target_minus_sham_rescue_benefit"],1.0)
        self.assertEqual(result["denominator"],256)
        self.assertFalse(result["p_value_or_population_claim"])

    def test_saturated_and_function_fails_redundant_joint_necessity_test(self):
        result=factorial_metrics([toy_case(redundant=False) for _ in range(256)])
        self.assertEqual(result["group_joint_error_excess"],-1.0)
        self.assertEqual(result["factorial_group_minus_sham_interaction"],-1.0)

    def test_ineligible_and_last_pass_cases_preserve_denominator(self):
        rows=[toy_case() for _ in range(128)]+[
            toy_case(admissible=False) for _ in range(128)]
        result=factorial_metrics(rows)
        self.assertEqual(result["denominator"],256)
        self.assertEqual(result["admissible_cases"],128)
        self.assertEqual(result["factorial_group_minus_sham_interaction"],0.5)
        self.assertEqual(result["delayed_target_minus_sham_rescue_benefit"],0.5)
        bad=copy.deepcopy(rows)
        bad[-1]["errors"]["remove_AB"]=1
        with self.assertRaisesRegex(ValueError,"identical no-op"):
            factorial_metrics(bad)
        later=[toy_case(delayed=False) for _ in range(256)]
        self.assertEqual(factorial_metrics(later)["delayed_target_minus_sham_rescue_benefit"],0.0)

    def test_exact_addback_is_identity_control_only(self):
        bad=[toy_case() for _ in range(256)]
        bad[5]["errors"]["same_visit_exact_addback"]=1
        with self.assertRaisesRegex(ValueError,"same-visit feature restoration"):
            factorial_metrics(bad)

    def test_functional_calibration_does_not_open_query(self):
        selected=select_from_calibration({
            "source-quartet-redundant":[toy_case() for _ in range(256)],
            "source-quartet-saturated":[toy_case(redundant=False) for _ in range(256)],
        })
        self.assertEqual(selected["selected"],"source-quartet-redundant")
        self.assertEqual(selected["source_calibration_cases"],256)
        self.assertFalse(selected["source_query_32_to_39_read"])
        self.assertIn("selection_conditioning",selected)


if __name__=="__main__":
    unittest.main()
