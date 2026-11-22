"""NEC-03W pure-stdlib tests of immutable archived sham-coverage anatomy."""
from __future__ import annotations

import math
import unittest

from mechinterp.neuron_communities.nec03w_coverage import (
    ROOTS,QUALIFIED,ORIGINAL_ZEROS,PANEL,BUDGETS,ARMS,
    ANGLE_GRID,COHESION_GRID,PRIORITY,OLD_MODES,
    admissible,raw_site_flags,summary_stats,verify_lock,
)


def fixture(*,angle=.2,cohesion=.3,raw0=2.,raw1=4.):
    return {
        "k":4,
        "active_ids":[list(range(16))],
        "exactly_16_positive_active_codes":[True],
        "source_ids":[list(range(4))],
        "sham_ids":[list(range(4,8))],
        "hash_sham_ids":[list(range(8,12))],
        "exhaustive_control_count":[495],
        "source_physical_cohesion":[.51],
        "sham_physical_cohesion":[.51+cohesion],
        "angular_gap":[angle],
        "cohesion_gap":[cohesion],
        "sham_joint_score":[angle+.5*cohesion],
        "angular_physical_dose":{
            "raw_source_32d_dose":[raw0],
            "raw_sham_32d_dose":[raw1],
            "shared_output_32d_dose":[min(raw0,raw1)],
            "source_gain":[min(raw0,raw1)/raw0],
            "sham_gain":[min(raw0,raw1)/raw1],
            "matched_1pct":[True],
            "nonzero_dose":[True],
        },
    }


class TestNEC03W(unittest.TestCase):
    def test_original_archive_and_retrospective_grid_lock(self):
        cfg,sha=verify_lock()
        self.assertEqual(len(sha),64)
        self.assertEqual(cfg["old_nec03v_actions_run"],44968927511)
        self.assertEqual(ROOTS,tuple(range(8)))
        self.assertEqual(QUALIFIED,(0,1,2,4,6,7))
        self.assertEqual(ORIGINAL_ZEROS,(3,5))
        self.assertEqual(PANEL,tuple(range(32,40)))
        self.assertEqual(BUDGETS,{"four":4,"six":6})
        self.assertEqual(ARMS,("composed","additive"))
        self.assertEqual((ANGLE_GRID,COHESION_GRID),((.25,.35,.5),(.5,.75,1.)))
        self.assertEqual(len(OLD_MODES),11)
        self.assertEqual(cfg["original_primary_accuracy_effect"],25/2048)
        self.assertEqual(cfg["original_source_effect_p_exploratory"],.5)
        self.assertEqual(cfg["first_failure_priority"],list(PRIORITY))
        self.assertTrue(cfg["posthoc_grid_has_no_counterfactual_predictions"])
        self.assertTrue(cfg["no_new_sham_candidates"])
        for k in ("new_model_forwards","model_or_SAE_training_updates",
                  "new_development_eval_scored","new_reserved_scored",
                  "new_recruited_worlds_scored","new_embargoed_five_input_scored",
                  "new_four_input_confirmation_scored"):
            self.assertEqual(cfg[k],0)

    def test_exact_strong_sham_angular_and_cohesion_flags(self):
        v=fixture()
        flags,physical=raw_site_flags(v,0)
        self.assertEqual(flags,set())
        self.assertTrue(admissible([physical],angle=.25,cohesion=.50))
        flags,physical=raw_site_flags(fixture(angle=.30,cohesion=.61),0)
        self.assertEqual(flags,{"angle","cohesion"})
        self.assertFalse(admissible([physical],angle=.25,cohesion=.50))
        self.assertTrue(admissible([physical],angle=.35,cohesion=.75))
        self.assertFalse(admissible([],angle=.5,cohesion=1.))

    def test_control_outside_original_topk_and_invalid_dose_rejected(self):
        x=fixture()
        x["sham_ids"]=[[3,4,5,6]]
        with self.assertRaises(AssertionError):
            raw_site_flags(x,0)
        x=fixture()
        x["angular_physical_dose"]["source_gain"]=[1.5]
        with self.assertRaises(AssertionError):
            raw_site_flags(x,0)
        x=fixture()
        x["exhaustive_control_count"]=[494]
        with self.assertRaises(AssertionError):
            raw_site_flags(x,0)

    def test_original_missing_topk_is_explicit(self):
        x=fixture()
        x["angular_physical_dose"]["raw_source_32d_dose"]=[0.]
        x["angular_physical_dose"]["raw_sham_32d_dose"]=[0.]
        x["angular_physical_dose"]["shared_output_32d_dose"]=[0.]
        x["angular_physical_dose"]["source_gain"]=[0.]
        x["angular_physical_dose"]["sham_gain"]=[0.]
        x["angular_physical_dose"]["nonzero_dose"]=[False]
        x["active_ids"]=[[0,1]]
        x["exactly_16_positive_active_codes"]=[False]
        x["source_ids"]=[[]]
        x["sham_ids"]=[[]]
        x["hash_sham_ids"]=[[]]
        x["exhaustive_control_count"]=[0]
        for k in ("source_physical_cohesion","sham_physical_cohesion",
                  "angular_gap","cohesion_gap","sham_joint_score"):
            x[k]=[None]
        flags,_=raw_site_flags(x,0)
        self.assertEqual(flags,{"insufficient_topk","dose"})

    def test_summary_distribution_retains_original_data(self):
        x=summary_stats([1.,2.,3.,4.,5.])
        self.assertEqual(x["n"],5)
        self.assertEqual(x["median"],3.)
        self.assertEqual(x["min"],1.)
        self.assertEqual(x["max"],5.)
        self.assertIsNone(summary_stats([])["median"])

    def test_failure_priority_does_not_depend_on_json_order(self):
        self.assertEqual(PRIORITY,(
            "no_source_site","insufficient_topk","angle",
            "cohesion","dose","mismatch"))
        both={"cohesion","angle"}
        self.assertEqual(next(k for k in PRIORITY if k in both),"angle")


if __name__=="__main__":
    unittest.main()
