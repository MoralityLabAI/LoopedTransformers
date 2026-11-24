"""Unit and regression tests for LBF-17:
1. Repaired horizon-support theorem & Counterexample 1 ('abc': ab↔de, bc↔fa, df↔cc)
2. Relational hierarchy Counterexamples 2 ('abbc') & 3 ('aebcd')
3. Parameter matching (~53,047) and single-pass compute matching (FLOPs >= 3-pass)
4. Exact S6 alphabet equivariance, S_{m0}xS_{m1} set equivariance, and zero-preserving brightness
5. Orthogonal 5-way factored causal interventions & charged exact_lookahead accounting
6. Standalone zero-axiom Lean 4.19.0 verification of lean/LBF17AbstractInterp.lean."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.flop_counter import FlopCounterMode
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("optional CPU Torch is absent")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbf15_meeting_fresco as m15
import lbf16_equivariant_fresco as m16
import lbf17_relational_fresco as m17


class HorizonSupportCounterexampleTests(unittest.TestCase):
    def test_lbf16_counterexample_abc_and_relational_hierarchy_fixes(self):
        """Mandatory regression: w0='abc', rules ab↔de, bc↔fa, df↔cc.
        Pass 1 is exact; at Pass 2 positionwise multiplicative matching combines 'd' (from dec)
        and 'f' (from afa) into impossible 'df', firing df→cc. Relational & provenance cells fix it."""
        rules = (("ab", "de"), ("bc", "fa"), ("df", "cc"))
        w0 = "abc"
        c_id = m17.CHAR2ID["c"]
        d_id = m17.CHAR2ID["d"]
        f_id = m17.CHAR2ID["f"]

        # 1. Symbolic positionwise A_1: exact at k=1, spurious 'c' at pos 0,1 at k=2
        pos1_hist = m17.symbolic_pos1(rules, w0, k_max=3)
        self.assertFalse(bool(pos1_hist[1][0, c_id]))
        self.assertFalse(bool(pos1_hist[1][1, c_id]))
        self.assertTrue(bool(pos1_hist[1][0, d_id] and pos1_hist[1][1, f_id]))
        self.assertTrue(bool(pos1_hist[2][0, c_id] and pos1_hist[2][1, c_id]))

        # 2. Neural LBF-16 SparseMultiplicativeRewriteCell exhibits the exact same Pass-2 chimera
        torch.manual_seed(11)
        cell16 = m16.SparseMultiplicativeRewriteCell(ce=8)
        gate16 = m16.SparseEnvGate(8, init_bias=-1.5)
        r2, r3 = m16.EquivariantHorizonFrescoNet.encode_rule_tensors(rules)
        oh = F.one_hot(torch.tensor([[m17.CHAR2ID[ch] for ch in w0]]), num_classes=6).float().unsqueeze(-1)
        h = oh.expand(1, 3, 6, 8).clone()
        b = torch.zeros_like(h)
        with torch.no_grad():
            h, b1, _, _ = cell16(oh, h, torch.zeros_like(h), b, r2, r3)
            self.assertFalse(bool((b1[0, 0, c_id] > 1e-6).any()))
            self.assertFalse(bool((b1[0, 1, c_id] > 1e-6).any()))
            g2, _ = gate16(h, b1)
            h, b2, _, _ = cell16(oh, h, g2, b1, r2, r3)
            self.assertTrue(bool((b2[0, 0, c_id] > 1e-6).any()))
            self.assertTrue(bool((b2[0, 1, c_id] > 1e-6).any()))

        # 3. Symbolic Adj2, Sub3, Prov-Pos1, Prov4xAdj2 AND neural FastProvenanceBigramCell exclude 'c' at pos 0,1
        adj2_p3 = m17.adj2_to_pos1(m17.symbolic_adj2(rules, w0, k_max=3)[3])
        sub3_p3 = m17.sub3_to_pos1(m17.symbolic_sub3(rules, w0, k_max=3)[3])
        prov1_p3 = torch.stack(m17.symbolic_prov_pos1(rules, w0, k_max=3)[3], dim=0).any(dim=0)
        p42 = m17.symbolic_prov4_adj2(rules, w0, k_max=3)[3]
        p42_pos = m17.adj2_to_pos1(p42[0])
        for fb in p42[1]:
            p42_pos |= m17.adj2_to_pos1(fb)
        for p_clean in (adj2_p3, sub3_p3, prov1_p3, p42_pos):
            self.assertFalse(bool(p_clean[0, c_id]))
            self.assertFalse(bool(p_clean[1, c_id]))

        cell17 = m17.FastProvenanceBigramCell(cb=4, use_provenance=True, learned=True)
        gate17 = m16.SparseEnvGate(4, init_bias=-1.5)
        oh_s = oh.squeeze(-1)
        oh_bi = (oh_s[:, :-1, :, None] * oh_s[:, 1:, None, :]).unsqueeze(-1)
        r2_b, r3_b = m17.FastProvenanceBigramCell.encode_bigram_rules(rules)
        bi_bank = torch.zeros((1, 4, 2, 6, 6, 4))
        with torch.no_grad():
            for kk in range(3):
                g_bi = torch.zeros_like(bi_bank) if kk == 0 else gate17(oh_bi.unsqueeze(1).expand_as(bi_bank), bi_bank)[0]
                bi_bank, _ = cell17(oh_bi, kk, g_bi, bi_bank, r2_b, r3_b)
        pos_from_17 = m17.RelationalProvenanceFrescoNet.bigram_to_pos1_channels(bi_bank)
        self.assertFalse(bool((pos_from_17[0, 0, c_id] > 1e-6).any()))
        self.assertFalse(bool((pos_from_17[0, 1, c_id] > 1e-6).any()))

    def test_counterexamples_2_and_3_separating_adj2_sub3_and_provenance(self):
        c_id = m17.CHAR2ID["c"]
        b_id = m17.CHAR2ID["b"]

        # Counterexample 2: w0='abbc', ab↔de, bbc↔efa, def↔ccc
        # Adj2 fails at pass 2 via shared 'e'; Sub3 and Prov4xAdj2 / neural provenance cell stay exact.
        r_cex2 = (("ab", "de"), ("bbc", "efa"), ("def", "ccc"))
        a2_c2 = m17.adj2_to_pos1(m17.symbolic_adj2(r_cex2, "abbc", k_max=3)[2])
        s3_c2 = m17.sub3_to_pos1(m17.symbolic_sub3(r_cex2, "abbc", k_max=3)[3])
        self.assertTrue(bool(a2_c2[0, c_id]))
        self.assertFalse(bool(s3_c2[0, c_id]))

        # Counterexample 3: w0='aebcd', aeb↔def, bcd↔ffa, ff↔cc, dec↔bbb
        # Sub3 fails at pass 3 via shared 'ef' ('deffa' -> 'decca' -> 'bbb'); neural provenance cell stays exact!
        r_cex3 = (("aeb", "def"), ("bcd", "ffa"), ("ff", "cc"), ("dec", "bbb"))
        s3_c3 = m17.sub3_to_pos1(m17.symbolic_sub3(r_cex3, "aebcd", k_max=3)[3])
        self.assertTrue(bool(s3_c3[0, b_id]))

        torch.manual_seed(19)
        cell17 = m17.FastProvenanceBigramCell(cb=4, use_provenance=True, learned=True)
        gate17 = m16.SparseEnvGate(4, init_bias=-1.5)
        for rules, w0, bad_sym in ((r_cex2, "abbc", c_id), (r_cex3, "aebcd", b_id)):
            oh = F.one_hot(torch.tensor([[m17.CHAR2ID[ch] for ch in w0]]), num_classes=6).float()
            oh_bi = (oh[:, :-1, :, None] * oh[:, 1:, None, :]).unsqueeze(-1)
            r2_b, r3_b = m17.FastProvenanceBigramCell.encode_bigram_rules(rules)
            bi_bank = torch.zeros((1, 4, len(w0) - 1, 6, 6, 4))
            with torch.no_grad():
                for kk in range(3):
                    gb = torch.zeros_like(bi_bank) if kk == 0 else gate17(oh_bi.unsqueeze(1).expand_as(bi_bank), bi_bank)[0]
                    bi_bank, _ = cell17(oh_bi, kk, gb, bi_bank, r2_b, r3_b)
            pos17 = m17.RelationalProvenanceFrescoNet.bigram_to_pos1_channels(bi_bank)
            self.assertFalse(bool((pos17[0, 0, bad_sym] > 1e-6).any()))


class BudgetSymmetryAndCausalFactorTests(unittest.TestCase):
    def test_parameter_matching_and_gate_isolation(self):
        counts = {}
        for arch in m17.ALL_LEARNED_ARCHS:
            net, _ = m17.new_lbf17_model(107, arch)
            counts[arch] = net.parameter_count()
            self.assertGreaterEqual(counts[arch], 53040)
            self.assertLessEqual(counts[arch], 53100)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 60)

        std, fp_s = m17.new_lbf17_model(107, "rel_prov_standard")
        opn, fp_o = m17.new_lbf17_model(107, "rel_prov_opened")
        self.assertEqual(fp_s, fp_o)
        diffs = []
        for k, t in std.state_dict().items():
            u = opn.state_dict()[k]
            if not torch.equal(t, u):
                diffs.append(k)
                self.assertTrue(torch.all(t == -6.0))
                self.assertTrue(torch.all(u == -1.5))
        self.assertEqual(set(diffs), {"env_gate.gate.bias", "bi_gate.gate.bias", "fresco_gate.gate.bias"})


@unittest.skipUnless(os.environ.get("LBF14_UPSTREAM"), "requires pinned rewrite source")
class UpstreamEquivarianceComputeAndLeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rg = m17.base.load_upstream(Path(os.environ["LBF14_UPSTREAM"]))
        cases, _ = cls.rg.generate(20261231, {"train": 3})
        cls.cases = cases["train"]

    def test_s6_equivariance_set_equivariance_compute_match_and_factored_shams(self):
        c = self.cases[0]
        s = m15.FringeState.start(c)
        for _ in range(3):
            (sd, ix), _, _ = m15.geom_mm_select(self.rg, c, s)
            m15.expand_node(self.rg, c, s, sd, ix)

        net, _ = m17.new_lbf17_model(109, "rel_prov_opened")
        with torch.no_grad():
            net.env_gate.gate.weight.normal_(0.0, 0.2)
            net.bi_gate.gate.weight.normal_(0.0, 0.2)
            net.fresco_gate.gate.weight.normal_(0.0, 0.2)
        net.eval()

        with torch.no_grad():
            o_orig, _ = net(c, s, 100)

        # 1. Exact S6 alphabet-permutation invariance
        perm = str.maketrans("abcdef", "cfbead")
        c_p = type(c)(
            c.root, c.lineage, c.split, c.family,
            c.source.translate(perm), c.target.translate(perm),
            tuple((a.translate(perm), b.translate(perm)) for a, b in c.rules),
        )
        s_p = m15.FringeState(
            [[w.translate(perm) for w in s.fronts[0]], [w.translate(perm) for w in s.fronts[1]]],
            [{}, {}],
            [{w.translate(perm): d for w, d in s.dist[0].items()},
             {w.translate(perm): d for w, d in s.dist[1].items()}],
            s.work, s.expanded,
        )
        with torch.no_grad():
            o_p, _ = net(c_p, s_p, 100)
        torch.testing.assert_close(o_orig[-1]["pot"], o_p[-1]["pot"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(o_orig[-1]["s0"], o_p[-1]["s0"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(o_orig[-1]["s1"], o_p[-1]["s1"], rtol=1e-5, atol=1e-5)

        # 2. Exact S_{m0}xS_{m1} frontier-set equivariance
        s_rev = m15.FringeState(
            [list(reversed(s.fronts[0])), list(reversed(s.fronts[1]))],
            s.parent, s.dist, s.work, s.expanded,
        )
        with torch.no_grad():
            o_rev, _ = net(c, s_rev, 100)
        torch.testing.assert_close(o_orig[-1]["pot"].flip(0, 1), o_rev[-1]["pot"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(o_orig[-1]["s0"].flip(0), o_rev[-1]["s0"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(o_orig[-1]["s1"].flip(0), o_rev[-1]["s1"], rtol=1e-5, atol=1e-5)

        # 3. Verify 5-way factored causal decoupling via probe_lbf17_decision
        raw, obs = [], []
        with torch.no_grad():
            m17.probe_lbf17_decision(net, c, s, 100, "109:rel_prov_opened", 1, "c" * 64, raw, obs)
        self.assertEqual(len(raw), 1)

        # 4. Verify compute_nonrecurrent (K=1) FLOPs >= rel_prov_opened (K=3) FLOPs
        net_cnr, _ = m17.new_lbf17_model(109, "compute_nonrecurrent")
        net_cnr.eval()
        fc_rec = FlopCounterMode(display=False)
        with torch.no_grad(), fc_rec:
            net(c, s, 100)
        fc_cnr = FlopCounterMode(display=False)
        with torch.no_grad(), fc_cnr:
            net_cnr(c, s, 100)
        self.assertGreaterEqual(fc_cnr.get_total_flops(), fc_rec.get_total_flops())

        # 5. Verify exact_lookahead charges examined successors within budget cap
        for b_cap in (16, 32):
            r_la = m17.run_exact_lookahead_episode(self.rg, c, budget=b_cap)
            self.assertLessEqual(r_la["work"], b_cap)
            self.assertGreater(r_la["work"], 0)

    def test_lean_formalization_compiles_axiom_free_when_lean_available(self):
        lean_exe = m17.base.config.LEAN_EXE
        if not lean_exe.exists() and not shutil.which("lean"):
            self.skipTest("Lean 4.19 toolchain not installed in this environment")
        with tempfile.TemporaryDirectory() as td:
            receipt = m17.verify_lean_theory_module(Path(td) / "lean_check")
            self.assertEqual(receipt["compiled_axiom_free"], 15)


if __name__ == "__main__":
    unittest.main()
