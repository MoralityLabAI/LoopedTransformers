"""Feedback invariants, existing-checkpoint compatibility, and differentiable hybrid integration."""
import math
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lcgrt.feedback import advance, weight
from lcgrt.tactic_lm import PRESETS, TLMConfig, TacticLM, cfg_identity, parser, train
from lcgrt.errors import GymError


class RLTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_constant_endpoint_follows_straight_line_at_every_budget(self):
        for budget in (1, 2, 8, 16, 32):
            x = torch.tensor([2.0, -3.0], dtype=torch.float64)
            initial, endpoint = x.clone(), torch.tensor([10.0, 5.0], dtype=torch.float64)
            for step in range(budget):
                x = advance(x, endpoint, "rlt", step, budget)
                torch.testing.assert_close(x, initial + (step + 1) / budget * (endpoint - initial))
            self.assertIs(x, endpoint)
            self.assertAlmostEqual(sum(weight("rlt", t, budget) for t in range(budget)),
                                   sum(weight("matched_damping", t, budget) for t in range(budget)))

    def test_invalid_schedule_rejected(self):
        for args in (("typo", 0, 8), ("rlt", 8, 8), ("rlt", -1, 8), ("rlt", 0, 0)):
            with self.assertRaises(ValueError):
                weight(*args)
        with self.assertRaises(ValueError):
            TacticLM(replace(PRESETS["tlm-flat-XS"], feedback="rlt"))
        for option in ("--val-items", "--val-batch"):
            with self.assertRaises(GymError):
                train(parser().parse_args(["--out", ".", option, "0"]))

    def test_old_config_and_identity_are_preserved(self):
        cfg = PRESETS["tlm-tiny"]
        old = asdict(cfg); old.pop("feedback")
        self.assertEqual(TLMConfig(**old).feedback, "direct")
        identity = cfg_identity(cfg)
        self.assertNotIn("feedback", identity)
        self.assertEqual(cfg_identity(replace(cfg, feedback="rlt"))["feedback"], "rlt")

    def test_hybrid_has_identical_parameters_and_one_step_output(self):
        enc = torch.tensor([[12, 19, 1, 256], [4, 7, 256, 256]])
        mask = enc != 256
        for arch in ("loop", "hrm"):
            direct = TacticLM(replace(PRESETS["tlm-tiny"], arch=arch))
            for mode in ("rlt", "matched_damping"):
                hybrid = TacticLM(replace(direct.cfg, feedback=mode))
                hybrid.load_state_dict(direct.state_dict(), strict=True)
                torch.testing.assert_close(direct.encode(enc, mask, 1)[0], hybrid.encode(enc, mask, 1)[0], rtol=0, atol=0)
                # Probe instrumentation must leave the unperturbed forward computation unchanged.
                plain, _ = hybrid.encode(enc, mask, 4)
                trace = []
                mem, _ = hybrid.encode(enc, mask, 4, trace=trace)
                torch.testing.assert_close(plain, mem, rtol=0, atol=0)
                self.assertEqual(len(trace), 4)
                self.assertFalse(trace[-1]["state"].requires_grad)
                torch.testing.assert_close(trace[-1]["state"], trace[-1]["endpoint"], rtol=0, atol=0)
                mem.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in hybrid.fast.parameters() if p.grad is not None))
                hybrid.grad_ckpt = True
                hybrid.zero_grad(set_to_none=True)
                hybrid.encode(enc, mask, 4)[0].square().mean().backward()
                self.assertTrue(math.isfinite(float(hybrid.fast.att.q.weight.grad.norm())))
                hybrid.eval()
                with torch.no_grad():
                    zero = torch.zeros_like(trace[0]["input"])
                    same, _ = hybrid.encode(enc, mask, 4, initial_delta=zero)
                    torch.testing.assert_close(same, hybrid.encode(enc, mask, 4)[0], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                direct.encode(enc, mask, initial_delta=torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
