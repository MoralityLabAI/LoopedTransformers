"""CPU smoke against *the actual* LoopedChase implementation in this repo."""
import math
import unittest

from experiments.spectral_state_cost_v0.native_phase_smoke import run_native_smoke


class NativePhaseSmokeTests(unittest.TestCase):
    def test_untrained_native_loop_state_contract(self):
        record = run_native_smoke(seed=13012902, batch=2, loops=42)
        self.assertEqual(record["model"],
                         "research_gym.scripts.loopscope_toy.LoopedChase")
        self.assertIsNone(record["trained_checkpoint"])
        self.assertFalse(record["training_performed"])
        self.assertEqual(record["answer_position_state_shape"], [42, 2, 64])
        self.assertTrue(record["all_finite"])
        self.assertEqual(len(record["per_item"]), 2)
        for item in record["per_item"]:
            self.assertEqual(item["depth_samples"], 42)
            self.assertEqual(item["feature_channels"], 64)
            self.assertEqual(item["alias_classes"], 3)
            self.assertTrue(math.isfinite(item["periodic_proxy_relative_residual"]))
            self.assertGreaterEqual(item["periodic_proxy_relative_residual"], 0)
            self.assertLessEqual(item["periodic_proxy_relative_residual"], 1.000001)
            self.assertGreaterEqual(item["spectral_tail_energy_fraction_after_k_terms"], 0)
            self.assertLessEqual(item["spectral_tail_energy_fraction_after_k_terms"], 1.000001)


if __name__ == "__main__":
    unittest.main()
