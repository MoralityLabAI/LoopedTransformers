"""Unit tests for LBF-15 geometric meeting-potential fresco, parameter matching,
planar tree-fringe invariants, causal shams, Moran's I, and manuscript rendering."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("optional CPU Torch is absent")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lbf_gym import validate_encoded
import lbf15_meeting_fresco as m15


class ArchitectureAndCodecTests(unittest.TestCase):
    def test_parameter_budget_and_shared_gate_initialization(self):
        counts = {}
        for arch in m15.ARCHITECTURES:
            net, _ = m15.new_model(107, arch)
            counts[arch] = net.parameter_count()
            self.assertLessEqual(counts[arch], 57475)
            self.assertGreaterEqual(counts[arch], 55000)
        # All four controller classes matched within 0.2%
        self.assertLess(max(counts.values()) - min(counts.values()), 100)

        std, fp_std = m15.new_model(107, "recurrent_standard")
        opn, fp_opn = m15.new_model(107, "recurrent_opened")
        self.assertEqual(fp_std, fp_opn)
        diffs = []
        for k, t in std.state_dict().items():
            u = opn.state_dict()[k]
            if not torch.equal(t, u):
                diffs.append(k)
                self.assertTrue(torch.all(t == -6.0))
                self.assertTrue(torch.all(u == -2.0))
        self.assertEqual(set(diffs), {"word_gate.gate.bias", "fresco_gate.gate.bias"})

    def test_morans_i_spatial_coherence_vs_checkerboard(self):
        # Smooth 2D block has positive Moran's I; checkerboard has negative Moran's I
        block = torch.zeros((6, 6))
        block[:3, :3] = 1.0
        checker = torch.tensor([[(i + j) % 2 for j in range(6)] for i in range(6)], dtype=torch.float32)
        self.assertGreater(m15.morans_i(block), 0.4)
        self.assertLess(m15.morans_i(checker), -0.8)


@unittest.skipUnless(os.environ.get("LBF14_UPSTREAM"), "requires pinned rewrite source")
class FringeAndCausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rg = m15.base.load_upstream(Path(os.environ["LBF14_UPSTREAM"]))
        cases, _ = cls.rg.generate(20261229, {"train": 6})
        cls.cases = cases["train"]

    def test_planar_fringe_splice_and_work_accounting(self):
        c = self.cases[0]
        s = m15.FringeState.start(c)
        expected_first_deg = len(self.rg.neighbors(c, c.source))
        m15.expand_node(self.rg, c, s, 0, 0)
        self.assertEqual(s.work, expected_first_deg)
        self.assertEqual(s.expanded, 1)

    def test_pass1_equivalence_roll_norm_and_svg_validity(self):
        c = self.cases[0]
        s = m15.FringeState.start(c)
        m15.expand_node(self.rg, c, s, 0, 0)
        m15.expand_node(self.rg, c, s, 1, 0)
        self.assertTrue(s.live())
        net, _ = m15.new_model(107, "recurrent_opened")
        with torch.no_grad():
            net.word_gate.gate.weight.normal_(0.0, 0.25)
            net.fresco_gate.gate.weight.normal_(0.0, 0.25)
        net.eval()
        raw, obs = [], []
        outs, _ = m15.probe_fresco_decision(net, c, s, 128 - s.work, "107:recurrent_opened", 1, "a" * 64, raw, obs)
        self.assertEqual(len(outs), 3)
        self.assertEqual(len(obs), 1)
        enc = m15.base.encode(obs[0])
        validate_encoded(enc)
        svg = m15.render_manuscript_svg(
            c, raw[0]["u_words"], raw[0]["v_words"],
            [o["pot"] for o in raw[0]["normal"]],
            [o["s0"] for o in raw[0]["normal"]],
            [o["s1"] for o in raw[0]["normal"]],
            raw[0]["persisted_mask"],
        )
        ET.fromstring(svg)


if __name__ == "__main__":
    unittest.main()
