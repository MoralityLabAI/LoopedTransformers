"""NEC-03V-B source-only behavioral pilot: no-forward preregistration tests.

These artificial records verify scientific custody and physical gates only.
They do not supply a behavioral outcome or pretend to represent original SAE.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from mechinterp.neuron_communities.nec03v_behavior_source import (
    ROOTS,ELIGIBLE,ITT_ZERO,physical_null_gate,inputs,
)


GATES={
    "min_eligible_case_fraction_each_original_paired_source_arm":.95,
    "min_shared_dose_32D":.1,
    "min_fraction_selected_visits_source_sham_cosine_lte_0_95":.8,
    "max_physical_matched_relative_L2_error":.01,
}


def case(arm,group,mask,*,enabled=True, parallel=False,small=False):
    a=[0.0]*32
    b=[0.0]*32
    a[0]=.01 if small else 1.0
    b[0 if parallel else 1]=a[0]
    return {"arm":arm,"group":group,"mask":mask,
            "eligible_all_visits":enabled,
            "visits":{"3":{
                "effective_source32":a if enabled else [0.0]*32,
                "effective_sham32":b if enabled else [0.0]*32,
                "effective_zero_pair":not enabled}}}


def fixture(**kwargs):
    return [case(arm,group,mask,**kwargs)
            for arm in ("composed","additive") for group in range(24,32)
            for mask in range(32)]


class FrozenBehaviorGateTests(unittest.TestCase):
    def test_512_original_case_arm_physical_controls_are_label_blind(self):
        gate=physical_null_gate(fixture(),root=0,floor=GATES)
        self.assertTrue(gate["gate_passed"])
        self.assertEqual(gate["eligible_by_arm"],{"composed":256,"additive":256})
        self.assertEqual(gate["selected_site_doses"],512)
        self.assertEqual(gate["fraction_source_sham_direction_cosine_le_095"],1.0)
        self.assertFalse(gate["behavioral_labels_used"])

    def test_dose_or_direction_violation_fails_before_native_forward(self):
        with self.assertRaisesRegex(ValueError,"physical null gate failed"):
            physical_null_gate(fixture(parallel=True),root=2,floor=GATES)
        with self.assertRaisesRegex(ValueError,"physical null gate failed"):
            physical_null_gate(fixture(small=True),root=2,floor=GATES)

    def test_fixed_zero_source_ITT_and_full_denominator(self):
        rows=fixture()
        for record in rows:
            if record["arm"]=="additive":
                record["eligible_all_visits"]=False
        gate=physical_null_gate(rows,root=3,floor=GATES)
        self.assertTrue(gate["gate_passed"])
        self.assertEqual(gate["eligible_by_arm"]["additive"],0)
        self.assertEqual(ITT_ZERO,(3,5))
        self.assertEqual(ELIGIBLE,(0,1,2,4,6,7))
        self.assertEqual(ROOTS,tuple(range(8)))

    def test_changed_eligible_case_coverage_fails(self):
        rows=fixture()
        for row in rows[:18]:
            row["eligible_all_visits"]=False
        with self.assertRaisesRegex(ValueError,"physical null gate failed"):
            physical_null_gate(rows,root=0,floor=GATES)

    def test_dry_run_actually_invokes_original_source_custody_before_model(self):
        from unittest.mock import patch
        from mechinterp.neuron_communities.nec03v_behavior_source import run
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with patch(
                "mechinterp.neuron_communities.nec03v_behavior_source.inputs",
                return_value=(None,None,None,{"gate_passed":True})
            ) as validator:
                receipt=run(0,p,p,None,p,p,p,p,p/"not_created",
                            execute_registered_source_scoring=False)
                self.assertTrue(receipt["dry_run"])
                self.assertEqual(receipt["model_forwards"],0)
                self.assertEqual(validator.call_count,1)
                self.assertFalse((p/"not_created").exists())

    def test_missing_recruited_avoids_even_loading_historical_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaises(PermissionError):
                inputs(8,p,p,None,p,p,p,p)
            with self.assertRaises(PermissionError):
                inputs(95,p,p,None,p,p,p,p)

    def test_wrong_source_case_arm_count_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,"full two-arm"):
            physical_null_gate(fixture()[:-1],root=0,floor=GATES)


if __name__=="__main__":
    unittest.main()
