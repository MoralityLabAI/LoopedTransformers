"""Unit and regression tests for LBF-18:
1. Incremental word-envelope + static pairwise-ribbon equivalence (PyTorch & Rust backends)
2. Fail-closed cache invalidation on parameter mutation, rule change, mode change, and scrambles
3. 2x2 bank interaction & Shapley identities and root-level (R roots vs S seeds) statistics
4. Adversarial provenance collision Counterexamples 4A, 4B, 4C & multi-alphabet bitsets
5. HybridSymRecFrescoNet parameter budget, S6/set equivariance, and Lean 4.19 zero-axiom proof check
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
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
import lbf17_relational_fresco as m17
import lbf18_incremental_fresco as m18


class IncrementalCacheAndCollisionTests(unittest.TestCase):
    def test_2x2_factorial_shapley_and_root_level_stats(self):
        roots = ["r0", "r1", "r2", "r3"]
        # Non-additive case: I_EM != 0
        y11 = [1.0, 1.0, 0.9, 0.8]
        y01 = [0.7, 0.8, 0.7, 0.6]
        y10 = [0.9, 0.9, 0.8, 0.7]
        y00 = [0.4, 0.5, 0.5, 0.4]
        res = m18.analyze_2x2_bank_factorial(roots, y11, y01, y10, y00, higher_is_better=True)
        self.assertAlmostEqual(
            res["shapley_env"] + res["shapley_mat"],
            res["joint_recurrence_benefit"],
            places=6,
        )
        self.assertAlmostEqual(
            (res["marginal_drop_env_from_intact"] + res["marginal_drop_mat_from_intact"])
            - res["joint_recurrence_benefit"],
            res["interaction_I_EM"],
            places=6,
        )

        # Root-level vs seed-level variance & N_eff bounds
        eps = {
            107: {"r0": {"closed": True, "work": 20}, "r1": {"closed": True, "work": 40}, "r2": {"closed": False, "work": 128}},
            109: {"r0": {"closed": True, "work": 22}, "r1": {"closed": True, "work": 38}, "r2": {"closed": False, "work": 128}},
            113: {"r0": {"closed": True, "work": 21}, "r1": {"closed": True, "work": 41}, "r2": {"closed": False, "work": 128}},
        }
        rl = m18.root_level_summary(["r0", "r1", "r2"], eps, budget=128)
        self.assertEqual(rl["independent_roots"], 3)
        self.assertEqual(rl["total_seed_episodes"], 9)
        self.assertGreaterEqual(rl["effective_sample_size_n_eff"], 3.0 - 1e-6)
        self.assertLessEqual(rl["effective_sample_size_n_eff"], 9.0 + 1e-6)
        self.assertGreater(rl["work_icc"], 0.95)

    def test_counterexamples_4a_4b_4c_and_extended_alphabet_bitsets(self):
        # Counterexample 4A: width-5 rules at p1=0, p2=4 ("abbbebbbc"), 0 ≡ 4 (mod 4)
        w4a = "abbbebbbc"
        rules4a = (("abbbe", "dbbbf"), ("ebbbc", "fbbbd"))
        cset4a, trunc = m18.general_concrete_k_ball(rules4a, w4a, k_max=3)
        self.assertFalse(trunc)
        self.assertEqual(cset4a, {"abbbebbbc", "dbbbfbbbc", "abbbfbbbd"})
        self.assertNotIn("dbbbfbbbd", cset4a)

        if m18.RUST_BACKEND.available:
            bits4, _ = m18.RUST_BACKEND.sym_prov_adj2_u64([w4a], rules4a, alpha=6, fp=4, k_max=1)
            bits5, _ = m18.RUST_BACKEND.sym_prov_adj2_u64([w4a], rules4a, alpha=6, fp=5, k_max=1)
            chim = "dbbbfbbbd"

            def admits(bits_t, fp_v):
                for f in range(1, fp_v + 1):
                    if all(((int(bits_t[0, 1, f, q].item()) >> (m18.CHAR2ID[chim[q]] * 6 + m18.CHAR2ID[chim[q + 1]])) & 1) != 0
                           for q in range(len(chim) - 1)):
                        return True
                return False

            self.assertTrue(admits(bits4, 4))
            self.assertFalse(admits(bits5, 5))

            # Larger alphabet A = 8 and longer word L = 20 bitset vs general tensor exact match
            alpha8 = m18.EXTENDED_ALPHABET[:8]
            w_long = "abcdefghhgfedcbaabcd"
            rules8 = (("abc", "ghe"), ("fgh", "bcd"), ("de", "ha"))
            s_rs = m18.general_prov_fp_adj2_pos_support(rules8, w_long, alpha_str=alpha8, fp=4, k_max=3)
            # Force fallback tensor path by temporarily using alpha=12 superset embedding
            s_py12 = m18.general_prov_fp_adj2_pos_support(rules8, w_long, alpha_str=m18.EXTENDED_ALPHABET[:12], fp=4, k_max=3)
            self.assertTrue(torch.equal(s_rs, s_py12[:, :8]))

        # Counterexample 4B: secondary step-2 intra-fiber fork on "fabbc"
        w4b = "fabbc"
        rules4b = (("fa", "aa"), ("ab", "de"), ("bbc", "efa"), ("def", "ccc"))
        cset4b, _ = m18.general_concrete_k_ball(rules4b, w4b, k_max=3)
        self.assertTrue(all(w[1] != "c" for w in cset4b))
        supp4b_fp4 = m18.general_prov_fp_adj2_pos_support(rules4b, w4b, alpha_str="abcdef", fp=4, k_max=3)
        supp4b_fp8 = m18.general_prov_fp_adj2_pos_support(rules4b, w4b, alpha_str="abcdef", fp=8, k_max=3)
        self.assertTrue(bool(supp4b_fp4[1, m18.CHAR2ID["c"]]))
        self.assertTrue(bool(supp4b_fp8[1, m18.CHAR2ID["c"]]))

    def test_hybrid_parameter_budget(self):
        hyb = m18.HybridSymRecFrescoNet()
        self.assertGreaterEqual(hyb.parameter_count(), 53000)
        self.assertLessEqual(hyb.parameter_count(), 53200)


@unittest.skipUnless(os.environ.get("LBF14_UPSTREAM"), "requires pinned rewrite source")
class UpstreamIncrementalEquivalenceAndLeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rg = m18.base.load_upstream(Path(os.environ["LBF14_UPSTREAM"]))
        cases, _ = cls.rg.generate(20261231, {"train": 2})
        cls.cases = cases["train"]

    def test_cached_vs_uncached_scores_trajectories_and_invalidation(self):
        c0, c1 = self.cases[0], self.cases[1]
        model, ckpt_sha, _ = m18.load_frozen_lbf17_checkpoint(107, "rel_prov_opened")

        ctrl_py = m18.IncrementalFrescoController(model, ckpt_sha, use_rust=False)
        ctrl_rs = m18.IncrementalFrescoController(model, ckpt_sha, use_rust=True)

        s = m15.FringeState.start(c0)
        for _ in range(4):
            (sd, ix), _, _ = m15.geom_mm_select(self.rg, c0, s)
            m15.expand_node(self.rg, c0, s, sd, ix)

        with torch.no_grad():
            for md in ("normal", "ablate_axial", "env_zero", "env_roll", "mat_zero", "mat_roll", "env_mat_zero"):
                if md == "env_mat_zero":
                    o_py = ctrl_py.forward(c0, s, 96, mode=md)
                    o_rs = ctrl_rs.forward(c0, s, 96, mode=md)
                    for k in range(3):
                        torch.testing.assert_close(o_py[k]["pot"], o_rs[k]["pot"], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(o_py[k]["s0"], o_rs[k]["s0"], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(o_py[k]["s1"], o_rs[k]["s1"], rtol=1e-5, atol=1e-5)
                else:
                    o_ref, _ = model(c0, s, 96, mode=md)
                    o_py = ctrl_py.forward(c0, s, 96, mode=md)
                    o_rs = ctrl_rs.forward(c0, s, 96, mode=md)
                    for k in range(3):
                        torch.testing.assert_close(o_ref[k]["pot"], o_py[k]["pot"], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(o_ref[k]["s0"], o_py[k]["s0"], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(o_ref[k]["s1"], o_py[k]["s1"], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(o_ref[k]["sub_u"], o_py[k]["sub_u"], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(o_ref[k]["pot"], o_rs[k]["pot"], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(o_ref[k]["s0"], o_rs[k]["s0"], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(o_ref[k]["s1"], o_rs[k]["s1"], rtol=1e-5, atol=1e-5)

            # Scramble modes must be rejected from cross-size cache reuse
            rej_before = ctrl_rs.cache_rejections
            _ = ctrl_rs.forward(c0, s, 96, mode="scramble_window")
            _ = ctrl_rs.forward(c0, s, 96, mode="scramble_ribbon")
            self.assertEqual(ctrl_rs.cache_rejections, rej_before + 2)

            # Full search trajectory & witness equivalence across uncached, incremental PyTorch, and incremental Rust
            ep_unc = m18.run_clean_episode(self.rg, c0, uncached_model=model, budget=64)
            ep_py = m18.run_clean_episode(self.rg, c0, controller=ctrl_py, budget=64, cold_cache=True)
            ep_rs = m18.run_clean_episode(self.rg, c0, controller=ctrl_rs, budget=64, cold_cache=True)
            self.assertEqual(ep_unc["closed"], ep_py["closed"])
            self.assertEqual(ep_unc["closed"], ep_rs["closed"])
            self.assertEqual(ep_unc["work"], ep_py["work"])
            self.assertEqual(ep_unc["work"], ep_rs["work"])
            self.assertEqual(ep_unc["path"], ep_py["path"])
            self.assertEqual(ep_unc["path"], ep_rs["path"])
            self.assertGreater(ep_rs["word_hits"], 0)
            self.assertGreater(ep_rs["pair_hits"], 0)

            # Different rule system `c1` must never hit `c0` cache entries
            ep_c1_unc = m18.run_clean_episode(self.rg, c1, uncached_model=model, budget=64)
            ep_c1_warm = m18.run_clean_episode(self.rg, c1, controller=ctrl_rs, budget=64, cold_cache=False)
            self.assertEqual(ep_c1_unc["path"], ep_c1_warm["path"])

        # In-place weight mutation must fail closed immediately
        model_mut, sha_mut, _ = m18.load_frozen_lbf17_checkpoint(107, "rel_prov_opened")
        ctrl_mut = m18.IncrementalFrescoController(model_mut, sha_mut, use_rust=False)
        _ = ctrl_mut.forward(c0, s, 96, mode="normal")
        with torch.no_grad():
            model_mut.pot_head.bias.add_(1e-2)
        with self.assertRaises(m18.CacheInvalidationError):
            _ = ctrl_mut.forward(c0, s, 96, mode="normal")

    def test_hybrid_symmetries_and_sym_readout_equivalence(self):
        c = self.cases[0]
        s = m15.FringeState.start(c)
        for _ in range(3):
            (sd, ix), _, _ = m15.geom_mm_select(self.rg, c, s)
            m15.expand_node(self.rg, c, s, sd, ix)

        # Verify IncrementalSymController matches frozen sym_prov_readout
        sym_m, sym_sha, _ = m18.load_frozen_lbf17_checkpoint(107, "sym_prov_readout")
        sym_ctrl = m18.IncrementalSymController(sym_m, sym_sha, use_rust=True)
        with torch.no_grad():
            ref_o, _ = sym_m(c, s, 96)
            inc_o = sym_ctrl.forward(c, s, 96)
        torch.testing.assert_close(ref_o[-1]["pot"], inc_o[-1]["pot"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ref_o[-1]["s0"], inc_o[-1]["s0"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ref_o[-1]["s1"], inc_o[-1]["s1"], rtol=1e-5, atol=1e-5)

        # Verify HybridSymRecFrescoNet S6 alphabet equivariance and frontier-set equivariance
        torch.manual_seed(109)
        hyb = m18.HybridSymRecFrescoNet().eval()
        with torch.no_grad():
            hyb.fresco_gate.gate.weight.normal_(0.0, 0.2)
            o_orig, _ = hyb(c, s, 96)
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
            o_p, _ = hyb(c_p, s_p, 96)
            torch.testing.assert_close(o_orig[-1]["pot"], o_p[-1]["pot"], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(o_orig[-1]["s0"], o_p[-1]["s0"], rtol=1e-5, atol=1e-5)

            s_rev = m15.FringeState(
                [list(reversed(s.fronts[0])), list(reversed(s.fronts[1]))],
                s.parent, s.dist, s.work, s.expanded,
            )
            o_rev, _ = hyb(c, s_rev, 96)
            torch.testing.assert_close(o_orig[-1]["pot"].flip(0, 1), o_rev[-1]["pot"], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(o_orig[-1]["s0"].flip(0), o_rev[-1]["s0"], rtol=1e-5, atol=1e-5)

    def test_lean18_formalization_compiles_axiom_free(self):
        if m18.find_lean_exe() is None:
            self.skipTest("Lean 4 executable not installed in this environment")
        res = m18.verify_lbf18_lean_formalization()
        self.assertEqual(res["compiled_axiom_free"], 12)


if __name__ == "__main__":
    unittest.main()
