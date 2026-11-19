"""Source-only regression tests for tied target module cache semantics."""
import tempfile
import unittest
from pathlib import Path

import torch
from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.deuterium.native_trm.adapter_audit import cache_ledger as a


class TestTiedSourceOnlyAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_four_invocations_reduced_to_one_module_cache(self):
        torch.manual_seed(a.SMOKE_SEED)
        model=TinyRecursiveModel(48,4)
        x,_=fixed_data(2,a.SMOKE_SEED)
        r=a.record_one(model,x,a.SMOKE_SEED,"untrained_source_only")
        self.assertEqual(r["n_original_tied_module_calls"],4)
        self.assertEqual(r["stock_cache_values_per_module"],1)
        self.assertEqual(r["stock_cache_keys"],[a.TARGET])
        self.assertEqual(r["last_cached_native_input_error"],0)
        self.assertEqual(r["last_cached_CI_mask_error"],0)
        self.assertTrue(r["component_mask_used_at_all_tied_depths"])
        self.assertTrue(r["all_four_mask_uses_are_same_tensor_object"])
        self.assertEqual(len(r["mean_absolute_per_depth_mask_mismatch_against_last_mask"]),4)
        self.assertGreater(max(r["physical_input_rms_distance_from_last_pass"][:3]),1e-8)
        self.assertEqual(r["cached_single_pass_f32_bytes_per_batch"]*4,
                         r["all_four_inputs_f32_bytes_per_batch"])
        self.assertEqual(r["original_native_final_logits_shape"],[2,81,10])

    def test_reserved_sources_never_generated_by_source_modes(self):
        self.assertEqual(a.SMOKE_SEED,61076)
        self.assertEqual(a.TRAIN_SEED,31041)
        self.assertEqual(a.N_SOURCES,8)
        self.assertEqual(a.TRAINED_SEEDS,(103,211,307))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"pinned LGA-02"):
                a.run(Path(temp)/"blocked.json",mode="trained")
            self.assertFalse((Path(temp)/"blocked.json").exists())


if __name__=="__main__":
    unittest.main()
