"""NEC03W source calibration CPU-only registration tests; zero model inference."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mechinterp.neuron_communities.nec03w_calibration_codes import (
    extract, preflight, CALIBRATION_GROUPS, SOURCE_CALIBRATION_ROWS,
)
from mechinterp.neuron_communities.nec03w_calibration_registry import calibration_lock


class RegisteredCalibrationExtractionTests(unittest.TestCase):
    def test_original_eight_and_exact_256_source_rows_frozen(self):
        frozen=calibration_lock()
        self.assertEqual(frozen["allowed_original_roots"],list(range(8)))
        self.assertEqual(CALIBRATION_GROUPS,tuple(range(16,24)))
        self.assertEqual(SOURCE_CALIBRATION_ROWS,256)
        self.assertEqual(frozen["maximum_new_native_forward_batches_per_eligible_arm"],8)
        self.assertFalse(frozen["allow_task_labels_or_model_logits_saved"])
        self.assertFalse(frozen["recruited_embargoed_reserved_or_confirmation_access"])

    def test_protected_original_world_rejected_before_model_or_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            for root in (8,88,96,-1):
                with self.assertRaises(PermissionError):
                    preflight(root,p,p,None,p)

    def test_default_calibration_dry_run_remains_zero_forward_and_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            with patch(
                "mechinterp.neuron_communities.nec03w_calibration_codes.preflight",
                return_value=(None,None,{},None),
            ) as original_check:
                reply=extract(0,p,p,None,p,p/"should_not_exist")
                self.assertTrue(reply["dry_run"])
                self.assertEqual(reply["native_model_forwards"],0)
                self.assertFalse(reply["source_query_groups_32_to_39_read"])
                self.assertEqual(original_check.call_count,1)
                self.assertFalse((p/"should_not_exist").exists())

    def test_calibration_output_never_overwrites_past_original_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            target=p/"existing"
            target.mkdir()
            with patch(
                "mechinterp.neuron_communities.nec03w_calibration_codes.preflight",
                return_value=(None,None,{},None),
            ):
                with self.assertRaises(FileExistsError):
                    extract(0,p,p,None,p,target,allow_new_source_calibration_forwards=True)

    def test_query_positive_code_extraction_is_not_part_of_calibration_contract(self):
        frozen=calibration_lock()
        self.assertEqual(frozen["query_source_group_ids_excluded"],list(range(32,40)))
        self.assertEqual(frozen["previously_scored_source_group_ids_excluded"],list(range(24,32)))
        self.assertEqual(frozen["original_source_rank_groups_excluded"],list(range(40,80)))


if __name__=="__main__":
    unittest.main()
