"""NEC-03W/F01 purely synthetic original-source physical roster tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mechinterp.neuron_communities.nec03w_f01_physical import build
from mechinterp.neuron_communities.nec03w_f01_registry import physical_method_lock


class OriginalSourceFactorialPhysicalTests(unittest.TestCase):
    def fake_source(self,root=0,*,supported=256):
        cols=[[0.0]*32 for _ in range(128)]
        for f in range(16):
            cols[f][f//2]=1.
        selected={
            "visit":0,"quartet":[0,1,2,3],
            "partition":[[0,1],[2,3]],
            "candidate_index":0,"partition_index":0,
            "full_source_calibration_support_count":supported,
            "physical_sample_cases":32,"matched_physical_sample_count":32,
        }
        codes=[
            [[f,1.] for f in range(16)] if i<supported
            else [[f,1.] for f in range(4,20)]
            for i in range(256)]
        orig={"world_index":root,"world_id":f"dev5-{root:03d}"}
        shards={
            arm:{"visits":{"0":{"decoder32":cols}}} for arm in ("composed","additive")
        }
        positive={
            "arms":{
                arm:{"positive_codes_by_visit":{"0":codes}} for arm in ("composed","additive")
            }
        }
        shortlist={
            "selection_roster_sha256":"synthetic-original-shortlist",
            "selected_physical_source_candidate_per_arm":{
                "composed":selected,"additive":selected,
            },
        }
        return orig,shards,positive,shortlist,{}

    def test_exact_physical_256_cases_per_teacher_and_source_only(self):
        root=Path("/tmp/synthetic-f01-never-read")
        with patch(
            "mechinterp.neuron_communities.nec03w_f01_physical.validate_original_shortlist",
            return_value=self.fake_source(),
        ), patch(
            "mechinterp.neuron_communities.nec03w_f01_physical.file_hash",
            return_value="0"*64,
        ):
            result=build(root,root,root,root,root)
        self.assertEqual(result["schema"],
                         "NEC-03W/F01/unscored-original-calibration-factorial-physical-roster/v1")
        self.assertEqual(result["source_groups"],list(range(16,24)))
        self.assertEqual(result["source_calibration_cases_per_arm"],256)
        self.assertEqual(result["new_model_forwards"],0)
        self.assertEqual(result["task_labels_or_logits_read"],0)
        self.assertFalse(result["factorial_behavior_executed"])
        self.assertTrue(result["source_query_groups32_to39_not_read"])
        for arm in ("composed","additive"):
            row=result["arms"][arm]
            self.assertEqual(row["denominator"],256)
            self.assertEqual(row["admissible_case_count"],256)
            self.assertEqual(len(row["cases"]),256)
            self.assertEqual(row["cases"][0]["geometry"]["visit"],0)
            self.assertEqual(row["cases"][-1]["group"],23)
            self.assertEqual(row["cases"][-1]["mask"],31)

    def test_incomplete_source_quartet_support_retains_256_case_denominator(self):
        root=Path("/tmp/synthetic-f01-never-read")
        with patch(
            "mechinterp.neuron_communities.nec03w_f01_physical.validate_original_shortlist",
            return_value=self.fake_source(supported=160),
        ), patch(
            "mechinterp.neuron_communities.nec03w_f01_physical.file_hash",
            return_value="0"*64,
        ):
            result=build(root,root,root,root,root)
        for arm in ("composed","additive"):
            self.assertEqual(result["arms"][arm]["admissible_case_count"],160)
            self.assertEqual(result["arms"][arm]["invalid_paired_zero_case_count"],96)
            self.assertFalse(result["arms"][arm]["cases"][-1]["geometry"]["admissible"])

    def test_original_paired_ineligible_root3_additive_is_zero_and_counted(self):
        orig,shards,positive,shortlist,verified=self.fake_source(root=3)
        shortlist["selected_physical_source_candidate_per_arm"]["additive"]=None
        shards["additive"]["visits"]={}
        positive["arms"]["additive"]["positive_codes_by_visit"]={}
        with patch(
            "mechinterp.neuron_communities.nec03w_f01_physical.validate_original_shortlist",
            return_value=(orig,shards,positive,shortlist,verified),
        ), patch(
            "mechinterp.neuron_communities.nec03w_f01_physical.file_hash",
            return_value="0"*64,
        ):
            root=Path("/tmp/synthetic-f01-never-read")
            result=build(root,root,root,root,root)
        self.assertEqual(result["arms"]["additive"]["admissible_case_count"],0)
        self.assertEqual(result["arms"]["additive"]["denominator"],256)
        self.assertTrue(result["arms"]["additive"]["original_fixed_ITT_root"])
        self.assertEqual(result["arms"]["additive"]["cases"][0]["geometry"]["reason"],
                         "original_source_ineligible_no_SAE_site")

    def test_registry_rejects_mutated_original_source_method(self):
        frozen=physical_method_lock()
        self.assertEqual(frozen["source_calibration_groups"],list(range(16,24)))
        self.assertEqual(frozen["new_native_model_forwards"],0)
        self.assertFalse(frozen["recruited_embargoed_or_confirmation_access"])


if __name__=="__main__":
    unittest.main()
