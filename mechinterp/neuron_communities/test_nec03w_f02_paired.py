"""NEC-03W/F02 paired source-only physical coverage synthetic tests.

These fixtures contain no original task labels, model outputs, logits or
protected-world identities. The observed original root0 result is unchanged.
"""
from __future__ import annotations

import copy
import unittest

from mechinterp.neuron_communities.nec03w_f02_paired import derive_one
from mechinterp.neuron_communities.nec03w_f02_registry import registered_paired


def fixture(root=0,paired=152):
    frozen=root in (3,5)
    source={}
    for arm in ("composed","additive"):
        keep=256 if arm=="composed" else paired
        records=[
            {"group":16+i//32,"mask":i%32,
             "geometry":{"admissible":i<keep}}
            for i in range(256)]
        source[arm]={
            "denominator":256,"cases":records,
            "admissible_case_count":keep,
            "original_fixed_ITT_root":frozen,
        }
    return {
        "schema":"NEC-03W/F01/unscored-original-calibration-factorial-physical-roster/v1",
        "world_index":root,"world_id":f"dev5-{root:03d}",
        "source_groups":list(range(16,24)),
        "source_calibration_cases_per_arm":256,
        "new_model_forwards":0,"task_labels_or_logits_read":0,
        "factorial_behavior_executed":False,
        "source_query_groups32_to39_not_read":True,
        "recruited_reserved_embargoed_confirmation_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "physical_roster_sha256":"synthetic-only-does-not-represent-original-SHA",
        "arms":source,
    }


class SourcePairedMaskTests(unittest.TestCase):
    def test_zero_correct_original_source_case_intersection_is_label_blind(self):
        row=derive_one(0,fixture(0,152))
        self.assertEqual(row["new_paired_case_admitted"],152)
        self.assertEqual(row["composed_cases_zeroed_by_pairing"],104)
        self.assertEqual(row["additive_cases_zeroed_by_pairing"],0)
        self.assertEqual(len(row["paired_case_admissible"]),256)
        self.assertEqual(sum(row["paired_case_admissible"]),152)
        self.assertFalse(row["meets_new_paired_95_percent_gate"])
        self.assertTrue(row["original_source_eligible_root"])
        self.assertFalse(row["registered_original_ITT_fixed_zero_root"])

    def test_root6_genuine_64_case_minimum_fails_new_95percent_gate(self):
        row=derive_one(6,fixture(6,64))
        self.assertEqual(row["new_paired_case_admitted"],64)
        self.assertEqual(row["composed_cases_zeroed_by_pairing"],192)
        self.assertFalse(row["meets_new_paired_95_percent_gate"])

    def test_fully_paired_original_source_root_passes_gate(self):
        row=derive_one(2,fixture(2,256))
        self.assertEqual(row["new_paired_case_admitted"],256)
        self.assertTrue(row["meets_new_paired_95_percent_gate"])
        self.assertEqual(row["paired_case_fraction"],1.)

    def test_original_source_ineligible_root3_never_becomes_primary_evidence(self):
        row=derive_one(3,fixture(3,0))
        self.assertEqual(row["new_paired_case_admitted"],0)
        self.assertTrue(row["registered_original_ITT_fixed_zero_root"])
        self.assertFalse(row["original_source_eligible_root"])

    def test_changed_original_case_order_or_root_blocks_pairing(self):
        changed=fixture(1,256)
        changed["arms"]["composed"]["cases"][0]["mask"]=4
        with self.assertRaisesRegex(AssertionError,"chronology"):
            derive_one(1,changed)
        with self.assertRaises(PermissionError):
            derive_one(8,fixture(0,256))

    def test_registered_original_physical_method_is_source_only_and_posthoc(self):
        gate=registered_paired()
        self.assertTrue(gate["original_root0_behavior_was_seen_before_mask_registration"])
        self.assertFalse(gate["allow_reinterpret_prior_source_behavior"])
        self.assertEqual(gate["new_source_paired_coverage_threshold"],.95)
        self.assertFalse(gate["allow_task_labels_logits_or_gradients"])
        self.assertFalse(gate["allow_new_source_query_or_protected_worlds"])


if __name__=="__main__":
    unittest.main()
