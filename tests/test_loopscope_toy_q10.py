import unittest
import numpy as np
import torch

from research_gym.scripts import loopscope_toy as L
from research_gym.scripts import loopscope_toy_q10 as Q10


class TestLoopscopeToyQ10(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = L.LoopedChase(n_nodes=6, d=32, layers=1, heads=2).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def test_full_ablation_none_matches_forward(self):
        tokens, hops = L.make_batch(6, 8, 123, 32)
        preds_clean = Q10.run_full_ablated(self.model, tokens, 32, None, None)
        with torch.no_grad():
            states = self.model(tokens, 32)
            preds_fwd = torch.stack([self.model.readout(s).argmax(-1) for s in states], dim=0)
        self.assertTrue(torch.equal(preds_clean, preds_fwd))
        win = Q10.acc_windows_32(preds_clean, hops)
        self.assertEqual(len(win["by_loop"]), 32)
        self.assertAlmostEqual(win["acc_9_32"], float(np.mean(win["by_loop"][8:32])), places=6)

    def test_clock_probes_and_subspace_shapes(self):
        tok_tr, _ = L.make_batch(6, 24, 11, 8)
        tok_te, _ = L.make_batch(6, 12, 22, 8)
        hs_tr = Q10.full_trajectory(self.model, tok_tr, 8)
        hs_te = Q10.full_trajectory(self.model, tok_te, 8)
        clocks = Q10.clock_probes_all(hs_tr, hs_te)
        for k in ("C_cursor", "C_edge", "C_edge_full"):
            self.assertGreaterEqual(clocks[k], 0.0)
            self.assertLessEqual(clocks[k], 1.0)
        mus, Ps = Q10.per_pos_clock_subspaces(hs_tr, k=4)
        self.assertEqual(tuple(mus.shape), (7, 32))
        self.assertEqual(tuple(Ps.shape), (7, 32, 4))
        # Orthonormal columns per position
        for pos in range(7):
            eye = Ps[pos].T @ Ps[pos]
            self.assertTrue(torch.allclose(eye, torch.eye(4), atol=1e-5))
        preds_abl = Q10.run_full_ablated(self.model, tok_te, 12, Ps, mus)
        self.assertEqual(tuple(preds_abl.shape), (12, 12))

    def test_own_echo_disagreement_bounds(self):
        res = Q10.own_echo_disagreement(self.model, n_items=8, max_loops=12, horizon=3, seed=55)
        self.assertGreaterEqual(res["D_echo_own"], 0.0)
        self.assertLessEqual(res["D_echo_own"], 1.0)
        self.assertEqual(res["items"], 8)

    def test_score_cohort_rules(self):
        rows = []
        for i in range(12):
            fail = i >= 8
            rows.append({
                "arm": f"lam0_s{10 + i}",
                "set": "q10",
                "training_valid": True,
                "extrap32": not fail,
                "C_cursor": 0.80 if fail else 0.35,
                "C_edge": 0.75 if fail else 0.30,
                "C_edge_full": 0.85 if fail else 0.40,
                "D_echo_own": 0.25 if fail else 0.00,
                "f1_over_f8": 0.12 if fail else 0.01,
                "clean": {"acc_9_16": 0.4 if fail else 1.0, "acc_17_32": 0.2 if fail else 1.0, "acc_9_32": 0.3 if fail else 1.0},
                "ablation_full_clock": {"acc_9_32": 0.5 if fail else 1.0},
                "ablation_full_random": {"acc_9_32": 0.3 if fail else 1.0},
                "effect_full_clock_9_32": 0.20 if fail else 0.00,
                "effect_full_clock_acc8": 0.00,
            })
        rep = Q10.score_cohort(rows, is_prospective=True)
        self.assertEqual(rep["P10a_C_cursor"]["verdict"], "supported")
        self.assertEqual(rep["P10a_C_cursor"]["correct"], 12)
        self.assertEqual(rep["P10b_D_echo_own"]["verdict"], "supported")
        self.assertEqual(rep["P10c_full_clock_ablation"]["verdict"], "supported")


if __name__ == "__main__":
    unittest.main()
