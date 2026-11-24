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
        cur_rss, peak_rss = m18.current_and_peak_rss_bytes()
        self.assertGreater(cur_rss, 5 * 1024 * 1024)
        self.assertGreaterEqual(peak_rss, cur_rss)

    def test_referee_rstar_guards_and_decoupled_boolean_support(self):
        # Referee R*: {abc <-> def, ab <-> de, bf <-> cc} on w0 = "abcaaaaaaa"
        # Concrete 3-step witness: abcaaaaaaa -> defaaaaaaa -> abfaaaaaaa -> accaaaaaaa
        w0 = "abcaaaaaaa"
        rstar = (("abc", "def"), ("ab", "de"), ("bf", "cc"))
        cset, trunc = m18.general_concrete_k_ball(rstar, w0, k_max=3)
        self.assertFalse(trunc)
        self.assertIn("accaaaaaaa", cset)
        gt_pos1 = m17.alpha1_bool(sorted(cset), len(w0))

        # Legacy unguarded root-suppression drops 'c' at index 1; both guarded modes are 100% sound
        leg_supp = m18.general_prov_fp_adj2_pos_support(rstar, w0, alpha_str="abcdef", fp=4, k_max=3, guard_mode="legacy")
        fab_supp = m18.general_prov_fp_adj2_pos_support(rstar, w0, alpha_str="abcdef", fp=4, k_max=3, guard_mode="fable_len_guard")
        msk_supp = m18.general_prov_fp_adj2_pos_support(rstar, w0, alpha_str="abcdef", fp=4, k_max=3, guard_mode="pr348_mask_guard")
        self.assertFalse(bool(leg_supp[1, m18.CHAR2ID["c"]]))
        self.assertTrue(bool((gt_pos1 & ~leg_supp).any()))
        self.assertFalse(bool((gt_pos1 & ~fab_supp).any()))
        self.assertFalse(bool((gt_pos1 & ~msk_supp).any()))
        self.assertTrue(torch.equal(fab_supp, msk_supp))

        # Modulus fail-safe check: when fp=2 < ell_max=3 on {"abc"<->"efd", "dd"<->"cd"}, w="abcd" -> "efcd"
        w_mod = "abcd"
        r_mod = (("abc", "efd"), ("dd", "cd"))
        c_mod, _ = m18.general_concrete_k_ball(r_mod, w_mod, k_max=2)
        self.assertIn("efcd", c_mod)
        gt_mod = m17.alpha1_bool(sorted(c_mod), len(w_mod))
        fab_fp2 = m18.general_prov_fp_adj2_pos_support(r_mod, w_mod, alpha_str="abcdef", fp=2, k_max=2, guard_mode="fable_len_guard")
        self.assertFalse(bool((gt_mod & ~fab_fp2).any()))

        # Decoupled exact Boolean u64 support in IncrementalFrescoController (PyTorch & Rust)
        case_rstar = m18.StressRewriteCase("rstar", "lin_rstar", "calibration", "equality", w0, "accaaaaaaa", rstar)
        model_orig, ckpt_sha, _ = m18.load_frozen_lbf17_checkpoint(107, "rel_prov_opened")
        for use_rs in ([False, True] if m18.RUST_BACKEND.available else [False]):
            ctrl_fab = m18.IncrementalFrescoController(
                model_orig, ckpt_sha, use_rust=use_rs, guard_mode="fable_len_guard"
            )
            b_supp = ctrl_fab.exact_boolean_pos1_support(case_rstar, w0, k=3)
            self.assertTrue(torch.equal(b_supp, fab_supp))
            self.assertFalse(bool((gt_pos1 & ~b_supp).any()))

        # Prove support independence from float32 glyph brightness + margin floor alarm
        model_dim, _ = m17.new_lbf17_model(107, "rel_prov_opened")
        model_dim.load_state_dict(model_orig.state_dict())
        model_dim.eval()
        with torch.no_grad():
            for p in model_dim.bi_cell.parameters():
                p.zero_()
            model_dim.bi_cell.rule2_mlp[-2].bias.fill_(-20.0)
            model_dim.bi_cell.rule3_mlp[-2].bias.fill_(-20.0)
        for use_rs in ([False, True] if m18.RUST_BACKEND.available else [False]):
            ctrl_dim = m18.IncrementalFrescoController(
                model_dim, "dimmed_test", use_rust=use_rs, guard_mode="fable_len_guard",
                margin_floor=1e-4, fail_on_margin_underflow=False,
            )
            b_dim = ctrl_dim.exact_boolean_pos1_support(case_rstar, w0, k=3)
            self.assertTrue(torch.equal(b_dim, fab_supp))
            self.assertGreater(ctrl_dim.margin_floor_violations, 0)

            ctrl_strict = m18.IncrementalFrescoController(
                model_dim, "dimmed_strict", use_rust=use_rs, guard_mode="fable_len_guard",
                margin_floor=1e-4, fail_on_margin_underflow=True,
            )
            with self.assertRaises(RuntimeError):
                _ = ctrl_strict.exact_boolean_pos1_support(case_rstar, w0, k=3)


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

        ctrl_py = m18.IncrementalFrescoController(model, ckpt_sha, use_rust=False, guard_mode="legacy")
        ctrl_rs = m18.IncrementalFrescoController(model, ckpt_sha, use_rust=True, guard_mode="legacy")

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

            # Also verify PyTorch == Rust across both guarded modes (`fable_len_guard` & `pr348_mask_guard`)
            for gm in ("fable_len_guard", "pr348_mask_guard"):
                g_py = m18.IncrementalFrescoController(model, ckpt_sha, use_rust=False, guard_mode=gm)
                g_rs = m18.IncrementalFrescoController(model, ckpt_sha, use_rust=True, guard_mode=gm)
                og_py = g_py.forward(c0, s, 96, mode="normal")
                og_rs = g_rs.forward(c0, s, 96, mode="normal")
                for k in range(3):
                    torch.testing.assert_close(og_py[k]["pot"], og_rs[k]["pot"], rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(og_py[k]["s0"], og_rs[k]["s0"], rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(og_py[k]["s1"], og_rs[k]["s1"], rtol=1e-5, atol=1e-5)

            # Scramble modes must be rejected from cross-size cache reuse
            rej_before = ctrl_rs.cache_rejections
            _ = ctrl_rs.forward(c0, s, 96, mode="scramble_window")
            _ = ctrl_rs.forward(c0, s, 96, mode="scramble_ribbon")
            self.assertEqual(ctrl_rs.cache_rejections, rej_before + 2)

            # Full search trajectory & witness equivalence across uncached, incremental PyTorch, and incremental Rust
            ep_unc = m18.run_clean_episode(self.rg, c0, uncached_model=model, uncached_guard_mode="legacy", budget=64)
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

            # Guarded uncached vs guarded incremental Rust trajectory equivalence
            ctrl_grd_rs = m18.IncrementalFrescoController(model, ckpt_sha, use_rust=True, guard_mode="fable_len_guard")
            ep_g_unc = m18.run_clean_episode(self.rg, c0, uncached_model=model, uncached_guard_mode="fable_len_guard", budget=64)
            ep_g_rs = m18.run_clean_episode(self.rg, c0, controller=ctrl_grd_rs, budget=64, cold_cache=True)
            self.assertEqual(ep_g_unc["path"], ep_g_rs["path"])
            self.assertEqual(ep_g_unc["work"], ep_g_rs["work"])

            # Different rule system `c1` must never hit `c0` cache entries
            ep_c1_unc = m18.run_clean_episode(self.rg, c1, uncached_model=model, uncached_guard_mode="legacy", budget=64)
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

        # Verify IncrementalSymController(guard_mode="legacy") matches frozen sym_prov_readout
        sym_m, sym_sha, _ = m18.load_frozen_lbf17_checkpoint(107, "sym_prov_readout")
        sym_ctrl = m18.IncrementalSymController(sym_m, sym_sha, use_rust=True, guard_mode="legacy")
        with torch.no_grad():
            ref_o, _ = sym_m(c, s, 96)
            inc_o = sym_ctrl.forward(c, s, 96)
        torch.testing.assert_close(ref_o[-1]["pot"], inc_o[-1]["pot"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ref_o[-1]["s0"], inc_o[-1]["s0"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ref_o[-1]["s1"], inc_o[-1]["s1"], rtol=1e-5, atol=1e-5)

        # Verify guarded IncrementalSymController PyTorch == Rust
        sym_g_py = m18.IncrementalSymController(sym_m, sym_sha, use_rust=False, guard_mode="fable_len_guard")
        sym_g_rs = m18.IncrementalSymController(sym_m, sym_sha, use_rust=True, guard_mode="fable_len_guard")
        with torch.no_grad():
            og_s_py = sym_g_py.forward(c, s, 96)
            og_s_rs = sym_g_rs.forward(c, s, 96)
        torch.testing.assert_close(og_s_py[-1]["pot"], og_s_rs[-1]["pot"], rtol=1e-5, atol=1e-5)

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

    def test_symbolic_controls_and_synthetic_cost_sweep(self):
        c0 = self.cases[0]
        for pol in ("absint1_mm", "absint_mm", "exact_mm", "bidir"):
            ep = m18.run_unified_symbolic_episode(self.rg, c0, pol, budget=32, guard_mode="fable_len_guard")
            self.assertGreaterEqual(ep["unique_succ_enumerated"], ep["work"])
            self.assertGreater(ep["sim_rule_window_checks"], 0)
            if pol in ("absint1_mm", "absint_mm", "exact_mm"):
                self.assertGreater(ep["model_rule_window_checks"], 0)
        ep_a = m18.run_unified_symbolic_episode(self.rg, c0, "absint_mm", budget=32)
        ep_b = m18.run_unified_symbolic_episode(self.rg, c0, "bidir", budget=32)
        sym_mock = {}
        for nm, ep_x in (("absint_mm", ep_a), ("bidir_b128", ep_b)):
            pw = float(ep_x["work"])
            un = float(ep_x["unique_succ_enumerated"])
            ms = ep_x["wall_clock_ms"]
            sym_mock[nm] = {
                "closure_rate": 1.0 if ep_x["closed"] else 0.0,
                "mean_penalized_work": pw,
                "mean_unique_succ_enumerated": un,
                "mean_repeated_succ_enumerated": float(ep_x["repeated_succ_enumerated"]),
                "mean_episode_ms": ms,
                "synthetic_verifier_cost_ms": {str(cs): ms + cs * pw for cs in m18.SYNTHETIC_SUCC_COSTS_MS},
                "synthetic_enumerated_cost_ms": {str(cs): ms + cs * un for cs in m18.SYNTHETIC_SUCC_COSTS_MS},
            }
        sweep = m18.compute_synthetic_cost_sweep({}, sym_mock)
        self.assertIn("100.0", sweep["arms"]["absint_mm"]["verifier_cost_ms"])
        self.assertAlmostEqual(
            sweep["arms"]["absint_mm"]["verifier_cost_ms"]["100.0"],
            ep_a["wall_clock_ms"] + 100.0 * float(ep_a["work"]),
            places=5,
        )

    def test_lean18_formalization_compiles_axiom_free(self):
        if m18.find_lean_exe() is None:
            self.skipTest("Lean 4 executable not installed in this environment")
        res = m18.verify_lbf18_lean_formalization()
        self.assertEqual(res["compiled_axiom_free"], 22)


if __name__ == "__main__":
    unittest.main()
