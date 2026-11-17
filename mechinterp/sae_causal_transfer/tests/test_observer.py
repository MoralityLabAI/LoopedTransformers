"""CPU-only regression tests for the LoopedTransformers SMC observer bridge."""
import json
from pathlib import Path
import unittest

import numpy as np

from mechinterp.sae_causal_transfer.adapter import (
    fingerprint_arrays, to_physical_dictionary,
)
from mechinterp.sae_causal_transfer.loop_depth import (
    Q, basis, difference, loop_gradient, loop_margin, transition, witness,
)
from mechinterp.sae_causal_transfer.observer import (
    ACTIVE, FEATURES, encode_top4, plan_edit, score_margin,
)


def fixture():
    """96/48 SAE with four active and 92 initially inactive covectors.

    Inactive block contains all 48 canonical basis rows plus 44 repeated
    e4 rows, giving full physical column rank and an inactive dose confound.
    """
    d = 48
    h = np.zeros(d)
    h[:4] = [1.3, 1.2, 1.1, 1.0]
    W = np.zeros((FEATURES, d))
    for j in range(4):
        W[j, j] = 1
    for j in range(d):
        W[4+j, j] = 1
    W[52:, 4] = 1
    b = np.zeros(FEATURES)
    b[4:] = -10
    D = np.zeros((FEATURES, d))
    D[3, 3] = 1
    return h, W, b, D


def looper(sign, steps=4):
    W = np.eye(48)
    W[4, 3] = sign / 2
    def continuation(state):
        s = np.array(state, dtype=np.float64, copy=True)
        for _ in range(steps):
            s = W @ s
        return 10 + float(s[4])
    return continuation


class PhysicalBridgeTests(unittest.TestCase):
    def test_normalization_to_physical_coordinates(self):
        h,W,b,D = fixture()
        mean = np.linspace(-1, 1, 48)
        scale = np.linspace(.5, 2, 48)
        enc = W * scale[None,:]
        enc_bias = b + W @ mean
        decoder = (D.T / scale[:,None])
        rows, bias, columns = to_physical_dictionary(
            enc, enc_bias, decoder, mean, scale)
        np.testing.assert_allclose(rows, W, atol=1e-12)
        np.testing.assert_allclose(bias, b, atol=1e-12)
        np.testing.assert_allclose(columns, D, atol=1e-12)
        np.testing.assert_allclose(
            enc @ ((h-mean)/scale)+enc_bias, rows @ h+bias, atol=1e-12)

    def test_array_transcript_changes_with_modified_weight(self):
        h,W,b,D = fixture()
        first = fingerprint_arrays(h,W,b,D)
        self.assertEqual(len(first),64)
        self.assertEqual(first, fingerprint_arrays(h,W,b,D))
        changed=W.copy()
        changed[0,0]+=1e-4
        self.assertNotEqual(first,fingerprint_arrays(h,changed,b,D))

    def test_complete_96_feature_admission_and_rank_obstruction(self):
        h,W,b,D = fixture()
        z,_ = encode_top4(h,W,b)
        self.assertEqual(np.flatnonzero(z>0).tolist(),[0,1,2,3])
        p=plan_edit(h,W,b,D,3,case_key="calib-frozen/puzzle10/goal3")
        self.assertTrue(p["eligible"])
        diag=p["diagnostics"]
        self.assertEqual(diag["active_ids"],[0,1,2,3])
        self.assertEqual(diag["inactive_encoder_rank"],48)
        self.assertEqual(diag["tangent_dimension"],44)
        self.assertEqual(len(diag["inactive_ids"]),92)
        self.assertEqual(len(diag["candidates"]),131)
        self.assertEqual(len(diag["all96_kill_preactivation_shift"]),96)
        self.assertEqual(len(diag["all96_best_preactivation_shift"]),96)
        self.assertGreater(diag["inactive_residual_floor"],0)
        self.assertLessEqual(
            diag["inactive_residual_floor"],diag["inactive_residual_best"]+1e-7)
        self.assertLessEqual(
            diag["inactive_residual_best"],diag["inactive_residual_baseline"]+1e-7)
        self.assertAlmostEqual(np.linalg.norm(p["kill"]),np.linalg.norm(p["best"]),places=9)
        self.assertAlmostEqual(np.linalg.norm(p["kill"]),np.linalg.norm(p["baseline"]),places=9)
        self.assertAlmostEqual(float(W[3]@p["kill"]),1.02,places=9)
        self.assertLess(float(np.linalg.norm(W[:3]@p["kill"])),1e-8)
        self.assertLess(float(np.linalg.norm(W[:4]@p["best"])),1e-8)
        after,_=encode_top4(h-p["kill"],W,b)
        shame,_=encode_top4(h-p["best"],W,b)
        self.assertEqual(float(after[3]),0)
        np.testing.assert_allclose(after[:3],z[:3],atol=1e-9)
        np.testing.assert_allclose(shame[:4],z[:4],atol=1e-9)

    def test_matched_vs_sham_causal_scores_with_tied_loop_depth(self):
        h,W,b,D = fixture()
        p=plan_edit(h,W,b,D,3,case_key="pinned/tied/4")
        self.assertTrue(p["eligible"])
        pos=score_margin(p,h,looper(+1,steps=4))
        neg=score_margin(p,h,looper(-1,steps=4))
        self.assertGreater(pos["clean_margin"],0)
        self.assertGreater(neg["clean_margin"],0)
        self.assertGreater(pos["kill_minus_best_margin_loss"],0)
        self.assertLess(neg["kill_minus_best_margin_loss"],0)
        # Numerical effects equal the exact affine linear gradient projection.
        for direction,observed in ((+1,pos),(-1,neg)):
            g=np.zeros(48)
            g[4]=1
            g[3]=2*direction
            predicted=float(g @ (p["kill"]-p["best"]))
            self.assertAlmostEqual(predicted,observed["kill_minus_best_margin_loss"],places=10)

    def test_failed_first_baseline_seed_retains_131_candidates(self):
        from unittest.mock import patch
        import mechinterp.sae_causal_transfer.observer as obs
        h,W,b,D = fixture()
        prior_project = obs._project
        calls = [0]

        def reject_one_baseline(vec, rows):
            if len(rows) == ACTIVE:
                calls[0] += 1
                if calls[0] == 1:
                    return None
            return prior_project(vec, rows)

        with patch.object(obs, "_project", side_effect=reject_one_baseline):
            p = obs.plan_edit(h,W,b,D,3,case_key="frozen/force-retry")
        self.assertTrue(p["eligible"])
        self.assertEqual(p["diagnostics"]["baseline_seed_attempts"],2)
        self.assertEqual(len(p["diagnostics"]["candidates"]),131)
        self.assertEqual(p["diagnostics"]["candidates"][0]["candidate"],"baseline")
        self.assertTrue(p["diagnostics"]["candidates"][0]["accepted"])

    def test_no_heldout_label_input_and_repeatability(self):
        h,W,b,D = fixture()
        p=plan_edit(h,W,b,D,3,case_key="training-frozen/solver1")
        q=plan_edit(h,W,b,D,3,case_key="training-frozen/solver1")
        self.assertEqual(p["diagnostics"]["selected_candidate"],
                         q["diagnostics"]["selected_candidate"])
        np.testing.assert_array_equal(p["kill"],q["kill"])
        np.testing.assert_array_equal(p["best"],q["best"])

    def test_fail_closed_for_inactive_target_and_zero_decoder(self):
        h,W,b,D=fixture()
        inactive=plan_edit(h,W,b,D,45,case_key="fixed/goal")
        self.assertFalse(inactive["eligible"])
        self.assertEqual(inactive["reason"],"matched_inactive")
        zero=D.copy()
        zero[3,:]=0
        cap=plan_edit(h,W,b,zero,3,case_key="fixed/goal")
        self.assertFalse(cap["eligible"])
        self.assertEqual(cap["reason"],"native_decoder_dose_cap")
        with self.assertRaises(ValueError):
            score_margin(inactive,h,looper(+1))

    def test_target_in_protected_span_is_rejected(self):
        h,W,b,D=fixture()
        W=W.copy()
        W[3]=W[2]
        p=plan_edit(h,W,b,D,3,case_key="frozen/span")
        self.assertFalse(p["eligible"])
        self.assertEqual(p["reason"],"protected_rank_or_target_span_obstruction")

    def test_bad_physical_input_rejected(self):
        h,W,b,D=fixture()
        with self.assertRaises(ValueError):
            plan_edit(h,W[:48],b,D,3,case_key="fixed")
        with self.assertRaises(ValueError):
            plan_edit(h,W,b,D,3,case_key="")
        with self.assertRaises(ValueError):
            to_physical_dictionary(W,b,D.T,np.zeros(48),np.zeros(48))


class RationalLoopWitnessTests(unittest.TestCase):
    def test_depth_growth_and_opposite_causal_signs(self):
        report=witness()
        self.assertEqual(report["status"],"synthetic_math_only")
        self.assertEqual(report["inactive_encoder_rank"],6)
        self.assertEqual(report["inactive_row_count"],92)
        self.assertEqual(report["unavoidable_inactive_residual_squared"],"25/16")
        self.assertEqual(report["equal_dose_sham_inactive_residual_squared"],"25/8")
        self.assertEqual([v["loops"] for v in report["effects_by_loop"]],[0,1,2,4,8])
        self.assertEqual(report["effects_by_loop"][-1]["positive_teacher_effect"],"5")
        self.assertEqual(report["effects_by_loop"][-1]["negative_teacher_effect"],"-5")
        self.assertEqual(loop_gradient(transition(+1),basis(4),4)[3],Q(2))
        self.assertEqual(loop_gradient(transition(-1),basis(4),4)[3],Q(-2))

    def test_invalid_depth_or_transition_rejected(self):
        with self.assertRaises(ValueError):
            transition(0)
        with self.assertRaises(ValueError):
            loop_margin(tuple(Q(0) for _ in range(6)),transition(1),basis(4),-1)


class HistoricalEvidenceTests(unittest.TestCase):
    def test_external_source_index_retains_negative_gates(self):
        p=Path(__file__).resolve().parents[1]/"source_evidence.json"
        d=json.loads(p.read_text())
        self.assertEqual(d["sources"]["smc14"]["registered_pairs_passing"],4)
        self.assertEqual(d["sources"]["smc15"]["registered_pairs_passing"],4)
        self.assertEqual(d["sources"]["smc16"]["registered_pairs_passing"],3)
        self.assertEqual(d["sources"]["smc17"]["registered_pairs_passing"],1)
        self.assertEqual(d["sources"]["smc17"]["eligible_goal_interventions"],267)
        self.assertEqual(len(d["smc17_family_rows"]),8)
        pairwise={}
        for row in d["smc17_family_rows"]:
            pairwise.setdefault(row["teacher"],[]).append(row["family_gate"])
            self.assertGreater(row["eligible_goals"],0)
            self.assertLessEqual(row["eligible_goals"],8*64)
        self.assertEqual(sum(all(accepted) for accepted in pairwise.values()),1)
        self.assertEqual({x for x,vals in pairwise.items() if all(vals)},
                         {"B701 SAE89"})
        for name in ("smc14","smc15","smc16","smc17"):
            self.assertEqual(len(d["sources"][name]["raw_sha256"]),64)


if __name__=="__main__":
    unittest.main()
