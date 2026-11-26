import unittest
import torch

from research.cog03_ledger_router.model import LedgerRouter, encode_observation
from research.cog09_carry_causality.run import (
    SPEC, forward_at_site, get_injection, patch, sham_directions, fit_probe,
)


class CarryTests(unittest.TestCase):
    def test_projection_erasure_and_matching(self):
        torch.manual_seed(41)
        direction = torch.randn(24)
        direction /= direction.norm()
        center = torch.randn(24)
        inj = torch.randn(1, 13, 24)
        shams = sham_directions(direction)
        erased, norm, _ = patch(inj, "erase", center, direction, shams)
        torch.testing.assert_close((erased-center) @ direction,
                                   torch.zeros(1, 13), atol=2e-6, rtol=0)
        for i, v in enumerate(shams):
            self.assertLess(abs(float(v @ direction)), 1e-6)
            patched, other_norm, _ = patch(inj, f"sham{i}", center, direction, shams)
            torch.testing.assert_close((patched-inj).norm(dim=-1),
                                       (erased-inj).norm(dim=-1))
            self.assertAlmostEqual(norm, other_norm, places=5)

    def test_explicit_site_matches_original(self):
        torch.manual_seed(7)
        model = LedgerRouter().eval()
        obs = {"world": {"atoms": [0, 1, 2], "facts": [0], "goal": 2,
                         "rules": [{"premises": [0], "conclusion": 1},
                                   {"premises": [1], "conclusion": 2}]},
               "known": [0], "history": []}
        with torch.no_grad():
            f, mask = encode_observation(obs, 16, 16)
            previous = model(f, mask, depth=SPEC["depth"])["state"]
            for prev in (None, previous):
                a = model(f, mask, prev, depth=SPEC["depth"])
                b = forward_at_site(model, f, mask, get_injection(model, prev))
                for k in a:
                    torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)

    def test_no_probe_fit_on_evaluation_roots(self):
        with self.assertRaises(ValueError):
            fit_probe(LedgerRouter(), [{"split": "ood_test"}])

    def test_first_step_and_reset(self):
        d = torch.ones(24) / 24**0.5
        c = torch.zeros(24)
        shams = sham_directions(d)
        for arm in SPEC["arms"][:7]:
            self.assertEqual(patch(None, arm, c, d, shams), (None, 0.0, 0.0))
        self.assertIsNone(patch(torch.ones(1, 2, 24), "reset_history", c, d, shams)[0])


if __name__ == "__main__":
    unittest.main()
