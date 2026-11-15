"""NRF-03 tests of the checkpoint schema and analytic/differential separation."""
import hashlib
import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None
from research_gym.analysis.nrf03_checkpoint_probe import (
    exact_matrix_norm_bound_ceil_6, one_step_certified_upper_ceil_6,
    probe_state_dict, load_pinned_weights,
)

@unittest.skipIf(torch is None, "PyTorch neural extra is needed for this test")
class CheckpointProbeTests(unittest.TestCase):
    def test_integer_norm_certificate(self):
        a = torch.tensor([[0., 2.], [0., 0.]], dtype=torch.float32)
        b = torch.eye(2, dtype=torch.float32)
        self.assertEqual(exact_matrix_norm_bound_ceil_6(a), 2_000_000)
        self.assertEqual(one_step_certified_upper_ceil_6(a,b), 1_500_000)
        self.assertEqual(exact_matrix_norm_bound_ceil_6(torch.zeros_like(a)), 0)

    def test_nonnormal_nilpotent_local_obstruction(self):
        a = torch.tensor([[0., 2.], [0., 0.]], dtype=torch.float32)
        b = torch.eye(2, dtype=torch.float32)
        bias = torch.zeros(2, dtype=torch.float32)
        result = probe_state_dict(a, bias, b, bias,
            rounds=2, observed_coords=1)
        self.assertGreater(result["jacobian_power_lower_witness"], 1)
        self.assertGreater(result["dark_to_observed_jvp_gain"], 0)
        self.assertEqual(result["global_one_step_upper"], 1.5)

    def test_zero_weights(self):
        a = torch.zeros(2,2, dtype=torch.float32)
        bias = torch.zeros(2, dtype=torch.float32)
        receipt = probe_state_dict(a, bias, a, bias, rounds=3, observed_coords=1)
        self.assertAlmostEqual(receipt["jacobian_power_lower_witness"],1.0,places=12)
        self.assertEqual(receipt["dark_to_observed_jvp_gain"],0)

    def test_pinned_checkpoint_hash_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/"surface_tied_R16_S101_E2048_step_16.pt"
            state={
                "blocks.0.in_proj.weight":torch.zeros(2,2),
                "blocks.0.in_proj.bias":torch.zeros(2),
                "blocks.0.out_proj.weight":torch.zeros(2,2),
                "blocks.0.out_proj.bias":torch.zeros(2),
            }
            torch.save({"state_dict":state,"optimizer":{},
                "cell_id":"surface_tied_R16_S101","exposure":2048,"step":16},path)
            entry={"file":path,"name":path.name,"rounds":16,"seed":101,
                   "exposure":2048,"step":16,
                   "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(load_pinned_weights(entry)[0].shape, (2,2))
            entry["sha256"]="bad"
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_pinned_weights(entry)


if __name__ == "__main__":
    unittest.main()
