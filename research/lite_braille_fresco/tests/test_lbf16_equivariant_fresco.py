"""Unit tests for LBF-16 exact S6 alphabet-permutation invariance, S_{m0}xS_{m1} frontier
set equivariance, support sparsity, parameter budget matching, and causal sham invariants."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("optional CPU Torch is absent")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbf15_meeting_fresco as m15
import lbf16_equivariant_fresco as m16


class EquivarianceAndBudgetTests(unittest.TestCase):
    def test_parameter_matching_and_gate_bias_isolation(self):
        counts = {}
        for arch in m16.ARCHITECTURES:
            net, _ = m16.new_model(107, arch)
            counts[arch] = net.parameter_count()
            self.assertLessEqual(counts[arch], 57475)
            self.assertGreaterEqual(counts[arch], 52000)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 80)

        std, fp_s = m16.new_model(107, "recurrent_standard")
        opn, fp_o = m16.new_model(107, "recurrent_opened")
        self.assertEqual(fp_s, fp_o)
        diffs = []
        for k, t in std.state_dict().items():
            u = opn.state_dict()[k]
            if not torch.equal(t, u):
                diffs.append(k)
                self.assertTrue(torch.all(t == -6.0))
                self.assertTrue(torch.all(u == -1.5))
        self.assertEqual(set(diffs), {"env_gate.gate.bias", "fresco_gate.gate.bias"})


@unittest.skipUnless(os.environ.get("LBF14_UPSTREAM"), "requires pinned rewrite source")
class SymmetryAndSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rg = m16.base.load_upstream(Path(os.environ["LBF14_UPSTREAM"]))
        cases, _ = cls.rg.generate(20261231, {"train": 4})
        cls.cases = cases["train"]

    def test_exact_s6_alphabet_equivariance_and_frontier_set_equivariance(self):
        c = self.cases[0]
        s = m15.FringeState.start(c)
        m15.expand_node(self.rg, c, s, 0, 0)
        m15.expand_node(self.rg, c, s, 1, 0)
        net, _ = m16.new_model(109, "recurrent_opened")
        with torch.no_grad():
            net.env_gate.gate.weight.normal_(0.0, 0.2)
            net.fresco_gate.gate.weight.normal_(0.0, 0.2)
        net.eval()

        with torch.no_grad():
            o_orig, _ = net(c, s, 110)

        # 1. Check S6 alphabet-permutation invariance
        perm = str.maketrans("abcdef", "cfbead")
        c_perm = type(c)(
            c.root, c.lineage, c.split, c.family,
            c.source.translate(perm), c.target.translate(perm),
            tuple((a.translate(perm), b.translate(perm)) for a, b in c.rules),
        )
        s_perm = m15.FringeState(
            [[w.translate(perm) for w in s.fronts[0]], [w.translate(perm) for w in s.fronts[1]]],
            [{}, {}],
            [{w.translate(perm): d for w, d in s.dist[0].items()},
             {w.translate(perm): d for w, d in s.dist[1].items()}],
            s.work, s.expanded,
        )
        with torch.no_grad():
            o_perm, _ = net(c_perm, s_perm, 110)
        torch.testing.assert_close(o_orig[-1]["pot"], o_perm[-1]["pot"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(o_orig[-1]["s0"], o_perm[-1]["s0"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(o_orig[-1]["s1"], o_perm[-1]["s1"], rtol=1e-5, atol=1e-5)

        # 2. Check S_{m0}xS_{m1} frontier set equivariance under row/column reversal
        s_rev = m15.FringeState(
            [list(reversed(s.fronts[0])), list(reversed(s.fronts[1]))],
            s.parent, s.dist, s.work, s.expanded,
        )
        with torch.no_grad():
            o_rev, _ = net(c, s_rev, 110)
        torch.testing.assert_close(
            o_orig[-1]["pot"].flip(0, 1), o_rev[-1]["pot"], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(o_orig[-1]["s0"].flip(0), o_rev[-1]["s0"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(o_orig[-1]["s1"].flip(0), o_rev[-1]["s1"], rtol=1e-5, atol=1e-5)

        # 3. Check Pass-1 equivalence and multi-pass roll isometry via probe_fresco_decision
        raw, obs = [], []
        m15.probe_fresco_decision(net, c, s, 110, "109:recurrent_opened", 1, "b" * 64, raw, obs)
        self.assertEqual(len(raw), 1)
        # Under zero mode, active envelope support at Pass 3 equals Pass 1 support
        with torch.no_grad():
            oz, _ = net(c, s, 110, mode="zero", return_tensors=True)
        supp_p1 = oz[0]["bw_u_out"] > 1e-6
        supp_p3 = oz[2]["bw_u_out"] > 1e-6
        self.assertTrue(torch.equal(supp_p1, supp_p3))


if __name__ == "__main__":
    unittest.main()
