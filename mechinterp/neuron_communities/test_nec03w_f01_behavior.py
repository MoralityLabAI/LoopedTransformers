"""NEC03W/F01 root0 source calibration behavior pre-execution synthetic tests.

No real original-source native models, task labels, calibration outcomes,
recruited or confirmation roots are accessed by this test module.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mechinterp.neuron_communities.nec03w_f01_behavior_independent import (
    independent_metrics,
)
from mechinterp.neuron_communities.nec03w_f01_behavior import (
    execute, input_gate,
)
from mechinterp.neuron_communities.nec03w_f01_behavior_registry import (
    registered_behavior,
)


def toy_case(*,redundant=True,valid=True,delayed=True):
    e={
        "baseline":0,
        "remove_A":0 if redundant else 1,
        "remove_B":0 if redundant else 1,
        "remove_AB":1,
        "sham_C":0,"sham_D":0,"sham_CD":0,
        "late_target_rescue":0 if delayed else 1,
        "late_matched_sham_rescue":1,
        "same_visit_exact_addback":0,
    }
    if not valid:
        e={key:0 for key in e}
    if not delayed:
        e["late_target_rescue"]=e["remove_AB"]
        e["late_matched_sham_rescue"]=e["remove_AB"]
    return {"eligible":valid,"delayed":delayed and valid,"errors":e}


class SourceBehaviorRegistrationTests(unittest.TestCase):
    def test_initial_only_root0_allowed_and_160_fixed_forward_batches(self):
        entry=registered_behavior()
        self.assertEqual(entry["allowed_initial_source_calibration_behavior_roots"],[0])
        self.assertEqual(entry["source_cases_per_arm"],256)
        self.assertEqual(len(entry["factorial_modes"]),10)
        self.assertEqual(entry["native_recurrent_visits_per_inference"],12)
        self.assertEqual(entry["original_source_input_roots"]["0"]["admitted_cases_by_arm"],
                         {"composed":256,"additive":152})
        self.assertEqual(entry["meaningful_positive_root_macro_floor_for_later_8root_study"],.02)
        self.assertFalse(entry["source_query_groups32to39_authorized"])
        self.assertFalse(entry["protected_recruited_reserved_embargoed_confirmation_access"])

    def test_original_recruited_and_other_unregistered_roots_blocked_before_files(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            for root in (1,3,5,7,8,95):
                with self.assertRaises(PermissionError):
                    input_gate(root,p,p,None,p,p,p,p,p,p,p)

    def test_zero_forward_original_root0_dry_run_keeps_outcome_file_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            original=({"allowed_initial_source_calibration_behavior_roots":[0]},
                      {"world_id":"dev5-000"},None,None,None,
                      {"physical_roster_sha256":"frozen-source"},
                      {"admissible_original_cases_by_arm":{"composed":256,"additive":152}})
            with patch(
                "mechinterp.neuron_communities.nec03w_f01_behavior.input_gate",
                return_value=original,
            ) as gate:
                result=execute(0,p,p,None,p,p,p,p,p,p,p,p/"must_not_exist",
                               allow_registered_source_behavior=False)
                self.assertTrue(result["dry_run"])
                self.assertEqual(result["new_native_forwards"],0)
                self.assertEqual(result["task_labels_read"],0)
                self.assertFalse((p/"must_not_exist").exists())
                self.assertEqual(gate.call_count,1)

    def test_paired_redundant_errors_pass_independent_factorial_calculation(self):
        metrics=independent_metrics([toy_case() for _ in range(256)])
        self.assertEqual(metrics["admissible_cases"],256)
        self.assertEqual(metrics["factorial_group_minus_sham_interaction"],1.)
        self.assertEqual(metrics["delayed_target_minus_sham_rescue_benefit"],1.)
        self.assertEqual(metrics["observed_binary_error_counts"]["remove_AB"],256)

    def test_saturated_AND_interaction_negative_and_last_visit_rescue_zero(self):
        metrics=independent_metrics([
            toy_case(redundant=False,delayed=False) for _ in range(256)])
        self.assertEqual(metrics["factorial_group_minus_sham_interaction"],-1.)
        self.assertEqual(metrics["delayed_target_minus_sham_rescue_benefit"],0.)

    def test_missing_source_case_and_bad_same_site_identity_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"denominator"):
            independent_metrics([toy_case() for _ in range(255)])
        cases=[toy_case() for _ in range(256)]
        cases[0]["errors"]["same_visit_exact_addback"]=1
        with self.assertRaisesRegex(ValueError,"same-site"):
            independent_metrics(cases)
        cases=[toy_case() for _ in range(256)]
        cases[7]["eligible"]=False
        with self.assertRaisesRegex(ValueError,"no-op"):
            independent_metrics(cases)

    def test_original_invalid_case_kept_as_zero_in_full_denominator(self):
        cases=[toy_case() for _ in range(152)]+[
            toy_case(valid=False) for _ in range(104)]
        metrics=independent_metrics(cases)
        self.assertEqual(metrics["admissible_cases"],152)
        self.assertEqual(metrics["factorial_group_minus_sham_interaction"],152/256)
        self.assertEqual(metrics["delayed_target_minus_sham_rescue_benefit"],152/256)


if __name__=="__main__":
    unittest.main()
