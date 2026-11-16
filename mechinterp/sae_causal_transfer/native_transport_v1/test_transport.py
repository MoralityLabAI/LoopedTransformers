"""LS-SAE09 tests: fixed physical transport bins, readout and boundary geometry."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from mechinterp.sae_causal_transfer.native_transport_v1 import audit as AUD
from mechinterp.sae_causal_transfer.native_transport_v1 import run_native as RUN


class HeadOnly:
    def __init__(self):
        self.lm_head = torch.nn.Linear(512, 11, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            for j in range(9):
                self.lm_head.weight[j + 2, j] = 1.0


class TransportTests(unittest.TestCase):
    def test_centered_nine_digit_readout_has_rank_eight(self):
        model = HeadOnly()
        wc, q, singular = RUN.readout_basis(model)
        self.assertEqual(wc.shape, (9, 512))
        self.assertEqual(q.shape, (512, 8))
        self.assertEqual(len(singular), 8)
        self.assertTrue(np.allclose(q.T @ q, np.eye(8), atol=1e-10))
        self.assertTrue(np.allclose(wc.sum(axis=0), 0, atol=1e-12))
        board = np.zeros((81,512),dtype=np.float64)
        board[:, 150] = 2  # Exact readout-nullspace input direction.
        projected = board @ q @ q.T
        self.assertLess(np.linalg.norm(projected), 1e-10)
        self.assertEqual(np.linalg.norm(board @ wc.T), 0.0)

    def test_signed_boundary_pressure_and_digit_flip(self):
        source = {"x": [0]+[1]*80, "target":[1]+[1]*80}
        clean = np.zeros((81,9),dtype=np.float64)
        clean[:, 0] = 2.0
        clean[0,1] = 1.0
        plus = clean.copy()
        plus[0,1] += .4
        m = RUN.pressure_and_flips(clean,plus,source)
        self.assertEqual(m["blank_flip_count"],0)
        self.assertAlmostEqual(m["boundary_pressure_max"],.4)
        self.assertAlmostEqual(m["minimum_clean_blank_margin"],1.0)
        independent = AUD.original_decision_stats(
            (clean.tolist(),plus.tolist()),source)
        self.assertAlmostEqual(independent["boundary_pressure_max"],.4)
        self.assertEqual(independent["blank_flip_count"],0)
        changed = clean.copy()
        changed[0,1] += 1.2
        m2 = RUN.pressure_and_flips(clean,changed,source)
        self.assertEqual(m2["blank_flip_count"],1)
        self.assertEqual(m2["blank_correct"],0)
        self.assertAlmostEqual(m2["boundary_pressure_max"],1.2)

    def test_frozen_descriptive_thresholds_and_zero_board(self):
        self.assertEqual(RUN.classification(.099,0,1.0,.05),
            {"contracted":True,"persistent_output_silent":False,
             "readout_small":True})
        self.assertEqual(RUN.classification(1.0,0,2.0,.2),
            {"contracted":False,"persistent_output_silent":True,
             "readout_small":False})
        self.assertEqual(RUN.classification(1.0,1,0,.0),
            {"contracted":False,"persistent_output_silent":False,
             "readout_small":False})

    def test_checker_rejects_corrupted_readout_and_norm_identities(self):
        self.assertTrue(AUD.close_square(5,3,4))
        self.assertFalse(AUD.close_square(5,3,5))
        self.assertTrue(AUD.close(1.0,1.0000001))
        self.assertFalse(AUD.close(1.0,2.0))

    def test_original_source_requires_exact_archived_sha(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            fake=Path(td)/"full.json"
            fake.write_text("{}")
            with self.assertRaises(ValueError):
                RUN.load_baseline(fake,pilot=False)


if __name__=="__main__":
    unittest.main()
