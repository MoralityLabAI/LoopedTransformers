"""NEC-03Y immutable archived functional damage method tests, stdlib only."""
from __future__ import annotations

import unittest

from mechinterp.neuron_communities.nec03y_damage_archive import (
    ARMS, MODES, ROOTS, PANEL, ZERO, QUALIFIED, INTERVENTIONS,
    COSINE_BINS, SUPPORT_BINS, ANGLE_BINS, DOSE_BINS,
    classify_correct, transition, transitions_summary,
    cosine_bin, support_bin, angle_bin, dose_bin,
    validate_source_receipt_shape, case_geometry, fixed_quality_strata,
    verify_lock,
)


def good_site_case():
    return {
        "active_ids":list(range(16)),
        "full_topk_support":True,
        "source_ids":[0,1,2,3],
        "continuous":{
            "source_ids":[0,1,2,3],
            "selected_sham_ids":[4,5,6,7],
            "selected_candidate_index":0,
            "candidate_count":495,
            "solver_iterations":32,
            "solver_restarts":1,
            "solver_max_relative_coefficient_multiplier":3.,
            "qualified_candidate_count":120,
            "optimized_coefficients":[.2,.4,.6,.8],
            "original_coefficients":[.3,.3,.4,.4],
            "coefficient_upper_bounds":[.9,.9,1.2,1.2],
            "angle_gap":.12,
            "cohesion_gap":.21,
            "cross_group_absolute_gram_overlap":.3,
            "physical_source_control_output_cosine":.7,
            "joint_score":.12+.25*.21+.25*(1-.7),
            "solver_optimal_objective":.12+.25*.21+.25*(1-.7),
            "source_physical_output_norm":2.,
            "control_physical_output_norm":3.,
            "source_physical_cohesion":.5,
            "sham_physical_cohesion":.29,
            "selected_candidate_quality_valid":True,
        },
        "discrete":{
            "sham_ids":[8,9,10,11],
            "original_candidates":495,
            "angle_gap":.2,
            "cohesion_gap":.4,
            "joint_score":.4,
            "valid_physical_match":True,
        },
    }


def good_dose():
    return {
        "raw_source_output_L2":[2.]*32,
        "raw_sham_output_L2":[4.]*32,
        "shared_output_L2":[2.]*32,
        "source_gain":[1.]*32,
        "sham_gain":[.5]*32,
        "dose_match_within_1pct":[True]*32,
        "positive_matched_dose":[True]*32,
    }


def one_case(base,src,sham, *, valid):
    return {
        "original_control_available":{"continuous":valid,"discrete":valid},
        "correct_by_mode":{
            "full_affine":base,"continuous4_source":src,"continuous4_sham":sham,
            "discrete4_source":src,"discrete4_sham":sham,
        },
        "transition_vs_full_affine":{
            "continuous4_source":transition(base,src),
            "continuous4_sham":transition(base,sham),
            "discrete4_source":transition(base,src),
            "discrete4_sham":transition(base,sham),
        },
        "physical_null_geometry":{
            "worst_source_control_cosine":.8,
            "qualified_candidate_min":120,
            "worst_angle_gap":.12,
            "smallest_shared_output_L2":2.,
        },
    }


class TestNEC03Y(unittest.TestCase):
    def test_exact_source_ancestry_and_lock(self):
        cfg,digest=verify_lock()
        self.assertEqual(len(digest),64)
        self.assertEqual(cfg["original_nec03x_run"],44975316374)
        self.assertEqual(ROOTS,tuple(range(8)))
        self.assertEqual(PANEL,tuple(range(40,48)))
        self.assertEqual(ARMS,("composed","additive"))
        self.assertEqual(QUALIFIED,(0,1,2,4,6,7))
        self.assertEqual(ZERO,(3,5))
        self.assertEqual(len(MODES),9)
        self.assertEqual(len(INTERVENTIONS),4)
        self.assertEqual(cfg["original_primary_ITT_effect"],-5/2048)
        self.assertTrue(cfg["same_old_source_sham_family_only"])
        self.assertFalse(cfg["posthoc_tests_computed"])
        for key in (
            "new_native_or_SAE_model_forwards","new_native_or_SAE_optimizer_updates",
            "new_sham_candidates_optimized","new_development_eval_groups_scored",
            "new_reserved_groups_scored","new_recruited_worlds_scored",
            "new_five_input_embargoed_worlds_scored",
            "new_four_input_confirmation_worlds_scored",
        ):
            self.assertEqual(cfg[key],0)

    def test_original_binary_prediction_and_tie_policy(self):
        self.assertEqual(classify_correct([1.,2.],1),(True,1))
        self.assertEqual(classify_correct([1.,2.],0),(False,1))
        self.assertEqual(classify_correct([1.,1.],0),(True,0))
        self.assertEqual(classify_correct([1.,1.],1),(False,0))
        for bad in ([float("nan"),0.],[0.,float("inf")],[1.],[]):
            with self.assertRaises(AssertionError):
                classify_correct(bad,0)
        with self.assertRaises(AssertionError):
            classify_correct([0.,1.],2)

    def test_valid_conditional_and_full_denominators(self):
        cases=[
            one_case(True,False,True,valid=True),
            one_case(True,False,False,valid=True),
            one_case(False,False,True,valid=True),
            one_case(True,True,True,valid=False),
        ]
        sham=transitions_summary(cases,"continuous4_sham")
        self.assertEqual(sham["case_count"],4)
        self.assertEqual(sham["full_affine_correct_baseline"],3)
        self.assertEqual(sham["correct_to_wrong"],1)
        self.assertEqual(sham["wrong_to_correct"],1)
        valid=transitions_summary(cases,"continuous4_sham",True)
        self.assertEqual(valid["case_count"],3)
        self.assertEqual(valid["correct_to_wrong"],1)
        self.assertEqual(valid["full_affine_correct_baseline"],2)
        self.assertEqual(valid["lost_correct_rate_given_correct_baseline"],.5)
        grouped=fixed_quality_strata(cases)
        self.assertEqual(grouped["output_cosine"]["075_and_up"]["cases"],3)
        self.assertEqual(grouped["output_cosine"]["075_and_up"]["sham_correct_to_wrong"],1)

    def test_predefined_physical_quality_bins(self):
        self.assertEqual(tuple(map(cosine_bin,[-.8,.25,.5,.75,1.])),
                         ("below_025","025_to_050","050_to_075","075_and_up","075_and_up"))
        self.assertEqual(tuple(map(support_bin,[0,1,49,50,199,200,495])),
                         ("zero","one_to_49","one_to_49","50_to_199",
                          "50_to_199","200_to_495","200_to_495"))
        self.assertEqual(tuple(map(angle_bin,[0.,.05,.15,.25,.251])),
                         ("below_005","005_to_015","015_to_025",
                          "015_to_025","above_025"))
        self.assertEqual(tuple(map(dose_bin,[0.,.001,.1,1.])),
                         ("zero","up_to_01","01_to_1","one_and_up"))
        self.assertEqual(len(COSINE_BINS),4)
        self.assertEqual(len(SUPPORT_BINS),4)
        self.assertEqual(len(ANGLE_BINS),4)
        self.assertEqual(len(DOSE_BINS),4)

    def test_original_continuous_null_quality_and_bounded_coefficients(self):
        item=good_site_case()
        validate_source_receipt_shape(item,good_dose(),0,positive_topk=True)
        item["continuous"]["optimized_coefficients"][0]=1.8
        with self.assertRaises(AssertionError):
            validate_source_receipt_shape(item,good_dose(),0,positive_topk=True)

    def test_invalid_physical_32d_dose_rejected(self):
        item=good_site_case()
        data=good_dose()
        data["source_gain"][0]=1.2
        with self.assertRaises(AssertionError):
            validate_source_receipt_shape(item,data,0,positive_topk=True)

    def test_transitions_distinguish_generic_damage_and_improvements(self):
        self.assertEqual(transition(True,False),"correct_to_wrong")
        self.assertEqual(transition(False,True),"wrong_to_correct")
        self.assertEqual(transition(True,True),"remained_correct")
        self.assertEqual(transition(False,False),"remained_wrong")


if __name__=="__main__":
    unittest.main()
