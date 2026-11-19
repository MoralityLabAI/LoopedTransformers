import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.deuterium import steering


class SteeringTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(steering.DEFAULT_PROTOCOL.read_text())
        self.cfg = self.protocol["new_toy_lane"]

    def test_rank_and_dose_controls_and_weight_restoration(self):
        from param_decomp.experiments.metta_amalgam.model import RecursiveGateTRM
        torch.manual_seed(91)
        model = RecursiveGateTRM(4, 8, 3, 3).eval()
        x = torch.randn(13, 4)
        original = model.head.weight.detach().clone()
        clean = model(x).detach()
        delta = steering.rank_one_normalize(torch.outer(torch.randn(3), torch.randn(8)), 0.02)
        self.assertEqual(int(torch.linalg.matrix_rank(delta)), 1)
        self.assertAlmostEqual(float(delta.norm()), 0.02, places=7)
        with steering.edited_weight(model, delta):
            self.assertTrue(torch.allclose(model(x), torch.nn.functional.linear(model.hidden(x), original + delta, model.head.bias)))
        self.assertTrue(torch.equal(model.head.weight, original))
        self.assertTrue(torch.equal(model(x), clean))
        with self.assertRaises(RuntimeError):
            with steering.edited_weight(model, delta):
                raise RuntimeError("interrupted scoring")
        self.assertTrue(torch.equal(model.head.weight, original))

    def test_corrupted_checkpoint_refused_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "powered_int3_logic/model"
            dec = root / "powered_int3_spd_20/model_20.pth"
            base.parent.mkdir()
            dec.parent.mkdir()
            base.write_bytes(b"tampered checkpoint")
            dec.write_bytes(b"tampered decomposition")
            with mock.patch.object(torch, "load") as load:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    steering.recovered_preflight(root, self.protocol, root / "out")
                load.assert_not_called()

    def test_failed_seed_is_retained_and_cannot_be_a_complete_comparison(self):
        results = [
            {"seed": 103, "comparison_completed": True, "heldout_evaluated": True},
            {"seed": 211, "comparison_completed": False, "heldout_evaluated": False},
            {"seed": 307, "comparison_completed": True, "heldout_evaluated": True},
        ]
        result = steering.aggregate(self.cfg, results, "score")
        self.assertFalse(result["comparison_completed"])
        self.assertEqual(len(result["seed_results"]), 3)
        self.assertEqual(result["status"], "BLOCKED_PREREQUISITE")

    def test_scoring_requires_a_registration_commit(self):
        with self.assertRaisesRegex(ValueError, "registration-commit"):
            steering.registration_check(None, steering.DEFAULT_PROTOCOL)

    def test_failure_after_exposure_cannot_claim_an_untouched_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result"
            def fail_after_exposure(cfg, seed, phase, seed_dir):
                steering.write_json(seed_dir / "heldout_exposure.json", {"seed": seed})
                raise TimeoutError("interrupted after exposure")
            args = ["steering", "toy", "--phase", "score", "--out", str(out),
                    "--registration-commit", "0" * 40]
            with mock.patch("sys.argv", args), mock.patch.object(steering, "registration_check", return_value={}), \
                    mock.patch.object(steering, "run_seed", side_effect=fail_after_exposure):
                with self.assertRaises(SystemExit) as raised:
                    steering.main()
                self.assertEqual(raised.exception.code, 1)
            result = json.loads((out / "steering_summary.json").read_text())
            self.assertTrue(result["heldout_exposed"])
            self.assertIsNone(result["heldout_evaluated"])
            self.assertFalse(result["comparison_completed"])
            self.assertTrue(all(r["heldout_exposed"] for r in result["seed_results"]))


if __name__ == "__main__":
    unittest.main()
