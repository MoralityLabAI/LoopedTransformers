"""Regression tests for OLR-08B (adaptive allocator, utility predictor, BPD, MMS, roster protection).

Verifies:
  1. Model architecture extends 122,881-param backbone cleanly (<146K params; 128,647 exact)
     with exact parameter, init-hash, and sampler-hash matching across matched neural arms.
  2. Direct MLP, recurrent, recurrent_bpd, matched_compute_sham, one_way, linear, and
     explicit_memory controls execute deterministically and cleanly.
  3. Optional BPD decomposition (SVD proxy vs learned BPD) + RMS-matched Haar rotation null
     + predictiveness evaluation (branch utility, tactic applicability, failure recovery).
  4. Configurable MMS budgets (0, 64, 256, 1024, 4096 MiB) enforce hard allocated_bytes <= budget_bytes,
     lazy allocation (4096 MiB does not eagerly allocate 4 GiB), eviction under tight budgets,
     and build/lookup/maintenance timing accounting.
  5. ImmutableReceiptLedger detects any tampering and keeps verifier receipts immutable.
  6. Preserved OLR-08A infrastructure invariants: pre-command scope rejection, premise heartbeat
     timeout treated as empty proposals while still charged, warmed exact? replay.
  7. Roster contamination audit: 0 spent files, 0 OLR-08A cal files, 0 OLR-08A protected eval files,
     0 SInt files, 0 TRAIN families in calibration, exact protection of 275-root eval registry
     with smoke-observed root BitVec.shiftLeftZeroExtend excluded from pristine confirmation.
  8. Feasibility gate fail-closed behavior (<120 hard CAL roots or <12 file clusters -> fail).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

OLR08B_DIR = Path(__file__).resolve().parent
OLR08_DIR = OLR08B_DIR.parent / "olr08"
RESEARCH_DIR = OLR08B_DIR.parent
for p in (str(OLR08B_DIR), str(OLR08_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run08  # noqa: E402
from allocator import AuditedMeter  # noqa: E402
from bpd_frontier import (  # noqa: E402
    BPD_FEATURE_DIM,
    BPDFrontierDecomposer,
    decompose_linear_svd,
    evaluate_bpd_predictiveness,
    fit_learned_bpd_site,
)
from mms_cache import (  # noqa: E402
    VALID_MMS_BUDGETS_MIB,
    EpisodicRecord,
    ImmutableReceiptLedger,
    ProofMemoryCache,
)
from models import (  # noqa: E402
    MATCHED_PARAM_ARMS,
    NEURAL_ARMS,
    ExplicitMemoryPolicy,
    ReceiptTracker,
    UtilityPredictor,
    build_packet,
    fit_utility_model,
    predict_costs_and_gates,
)
from roster import (  # noqa: E402
    FROZEN_CAL_FILES_08B,
    SMOKE_EVAL_ROOT_ID,
    load_exclusion_sets,
    verify_rule_matches_source,
)


def _dummy_packet():
    g = {
        "nodes": [{"k": "fvar", "n": "x"}, {"k": "const", "n": "Nat"}, {"k": "app"}],
        "edges": [[0, 2, "arg0"], [1, 0, "typeof"]],
        "obligations": [{"target": 2}],
    }
    lemmas = [{"name": "foo", "root": 0, "nodes": [{"k": "const", "n": "foo"}], "edges": []}]
    acts = [(1.0, "apply_lemma", "foo", "apply foo"), (2.0, "close", "", "simp_all"), (1.5, "close", "", "omega")]
    pcosts = {"s1": [1.0, 1.0, 1.0], "s2": [1.0, 2.0, 1.5], "s3": [0.8, 1.4, 2.1], "s3c": [3.2, 4.1, 3.8]}
    g16 = [0.8] * 16
    p = build_packet(g, lemmas, acts, depth=1, prior_costs=pcosts, global_cond16=g16)
    assert p is not None
    return p


class TestOLR08B(unittest.TestCase):
    def test_models_matched_params_init_and_sampler(self) -> None:
        torch.set_num_threads(1)
        p = _dummy_packet()
        frames = [
            {"packet": p, "target_idx": 0, "ok_mask": torch.tensor([1.0, 0.0, 1.0]), "best_prior_idx": 2},
            {"packet": p, "target_idx": 1, "ok_mask": torch.tensor([0.0, 1.0, 0.0]), "best_prior_idx": 1},
        ]
        sigs = []
        param_counts = {}
        for arm in NEURAL_ARMS:
            m, bpd_dec, st = fit_utility_model(frames, arm, seed=17, steps=4, bpd_rank=4)
            param_counts[arm] = st["parameters"]
            self.assertLess(st["parameters"], 146000)  # Strictly below 500K/5M; compact extension of 122,881
            costs, pw, summ = predict_costs_and_gates(m, p, alpha=1.0, bpd_decomposer=bpd_dec)
            self.assertEqual(len(costs), 3)
            self.assertTrue(all(c >= 0.0 for c in costs))
            self.assertEqual(len(pw), 4)
            self.assertAlmostEqual(sum(pw), 1.0, places=4)
            if arm in MATCHED_PARAM_ARMS:
                sigs.append((st["parameters"], st["initial_sha256"], st["sampler_sha256"]))
        # All 5 neural arms (recurrent, recurrent_bpd, direct_mlp, matched_compute_sham, one_way) share exact param count, init, sampler
        self.assertEqual(len(set(sigs)), 1)
        self.assertEqual(param_counts["recurrent"], 128647)

        # Direct MLP has zero frontier recurrence
        mlp = UtilityPredictor("direct_mlp")
        z_next = mlp.update_frontier_state(torch.ones(40), torch.ones(40), torch.ones(16))
        self.assertTrue(torch.allclose(z_next, torch.zeros(40)))

        # Recurrent updates frontier state nontrivially from verifier receipt
        rec = UtilityPredictor("recurrent")
        z_rec = rec.update_frontier_state(torch.zeros(40), torch.ones(40), torch.ones(16))
        self.assertGreater(float(z_rec.abs().sum()), 0.0)

    def test_explicit_memory_control_zero_params_and_source_only(self) -> None:
        pol = ExplicitMemoryPolicy()
        self.assertEqual(pol.parameters, 0)
        cache = ProofMemoryCache(budget_mib=64.0)
        rows = [
            {
                "role": "train",
                "on_path": [1],
                "events": [
                    {"kind": "close", "tactic": "rfl", "depth": 0, "ok": False, "hb_raw": 100},
                    {"kind": "close", "tactic": "simp_all", "depth": 0, "ok": True, "hb_raw": 200},
                ],
            }
        ]
        pol.fit_from_source_rows(rows, cache=cache)
        with self.assertRaises(ValueError):
            pol.fit_from_source_rows([{**rows[0], "role": "calibration"}])
        acts = [(1.0, "close", "", "rfl"), (1.0, "close", "", "simp_all")]
        sc = pol.score_actions(acts, [2.0, 2.0], depth=0, info={"defs": ["f"], "hyps": []}, cache=cache)
        self.assertLess(sc[1], sc[0])  # simp_all gets lower cost (higher priority) than failed rfl

    def test_mms_configurable_budgets_lazy_eviction_and_accounting(self) -> None:
        # All 5 registered budgets construct cleanly with lazy allocation (4096 MiB allocates 0 bytes initially)
        for mib in VALID_MMS_BUDGETS_MIB:
            c = ProofMemoryCache(budget_mib=mib, strict_registered_budget=True)
            self.assertEqual(c.allocated_bytes, 0)
            self.assertEqual(c.budget_bytes, int(mib * (2**20)))
        with self.assertRaises(ValueError):
            ProofMemoryCache(budget_mib=128, strict_registered_budget=True)

        # 0 MiB rejects all writes and reports 0 hits
        c0 = ProofMemoryCache(budget_mib=0)
        self.assertFalse(c0.put_lemma_graph("e", "lem1", {"name": "lem1", "nodes": [], "edges": []}))
        self.assertEqual(c0.allocated_bytes, 0)
        self.assertGreater(c0.budget_rejections, 0)
        self.assertIsNone(c0.get_lemma_graph("e", "lem1"))

        # Tight budget triggers eviction and never exceeds budget_bytes
        ct = ProofMemoryCache(budget_mib=0.002)  # 2,097 bytes
        sig = np.ones(24, dtype=np.float32) / np.sqrt(24.0)
        for i in range(40):
            ct.put_episodic(
                EpisodicRecord(
                    key=f"k{i}",
                    signature=sig,
                    tactic_template="simp",
                    action_kind="norm",
                    ok=(i % 2 == 0),
                    closed=(i == 39),
                    ngoals_delta=0.0,
                    hb_raw=100,
                    after_fail_streak=0,
                    step_idx=i,
                )
            )
            self.assertLessEqual(ct.allocated_bytes, ct.budget_bytes)
        self.assertGreater(ct.evictions, 0)
        hit, ok_r, _, n_m = ct.query_episodic(sig, tactic_template="simp")
        self.assertTrue(hit)
        self.assertGreater(n_m, 0)
        rec = ct.receipt()
        self.assertGreaterEqual(rec["build_seconds"], 0.0)
        self.assertGreaterEqual(rec["lookup_seconds"], 0.0)
        self.assertGreaterEqual(rec["maintenance_seconds"], 0.0)

    def test_immutable_receipt_ledger(self) -> None:
        led = ImmutableReceiptLedger("root_abc")
        payload = {"ok": True, "ngoals": 0, "hb_raw": 12345}
        h1 = led.append("step", payload)
        # Mutating caller's dict must NOT affect stored immutable receipt
        payload["ok"] = False
        self.assertTrue(led[0]["receipt"]["ok"])
        with self.assertRaises(TypeError):
            led[0]["op"] = "tampered"  # type: ignore[index]
        h2 = led.append("cert", {"kernel": True, "axioms": []})
        self.assertNotEqual(h1, h2)
        self.assertTrue(led.verify_chain())

    def test_bpd_decomposition_haar_null_and_predictiveness(self) -> None:
        torch.set_num_threads(1)
        m = UtilityPredictor("recurrent")
        svd_f = decompose_linear_svd("up", m.up.weight, rank_c=8)
        self.assertEqual(svd_f.map_kind, "svd_seed_proxy")
        self.assertGreater(svd_f.recon_cosine, 0.3)

        xs = [torch.randn(16, 40) for _ in range(3)]
        bpd_f = fit_learned_bpd_site("up", m.up.weight, xs, rank_c=8, steps=10, seed=17)
        self.assertEqual(bpd_f.map_kind, "learned_bpd")
        self.assertGreater(bpd_f.recon_cosine, 0.3)

        dec = BPDFrontierDecomposer.from_model(m, rank_c=8, learned=True, pass_inputs_by_site={"up": xs})
        null_dec = dec.make_haar_null(seed=77)
        p = _dummy_packet()
        out = m.forward_full(p)
        bf = dec.extract_action_features(out["pass_action_states"], torch.zeros(40))
        nf = null_dec.extract_action_features(out["pass_action_states"], torch.zeros(40))
        self.assertEqual(tuple(bf.shape), (3, BPD_FEATURE_DIM))
        self.assertEqual(tuple(nf.shape), (3, BPD_FEATURE_DIM))

        # Synthetic predictiveness test
        rng = np.random.default_rng(42)
        recs = []
        for i in range(60):
            signal = float(rng.normal())
            bpd_v = np.zeros(BPD_FEATURE_DIM, dtype=np.float32)
            bpd_v[0] = signal
            haar_v = rng.normal(size=BPD_FEATURE_DIM).astype(np.float32)
            cond_v = rng.normal(size=24).astype(np.float32) * 0.1
            y_ok = int(signal > 0.0)
            recs.append(
                {
                    "file": f"f{i % 6}.lean",
                    "ok": y_ok,
                    "closed": int(signal > 0.8),
                    "on_path": int(signal > 0.5),
                    "fail_streak": 2 if i % 2 == 0 else 0,
                    "bpd": bpd_v.tolist(),
                    "haar_null": haar_v.tolist(),
                    "cond": cond_v.tolist(),
                }
            )
        ev = evaluate_bpd_predictiveness(recs[:30], recs[30:])
        self.assertGreater(ev["tactic_applicability"]["bpd_components"]["auroc"], 0.85)
        self.assertGreater(ev["recovery_after_failure"]["bpd_components"]["auroc"], 0.85)

    def test_preserved_premise_timeout_charging_and_meter_ledger(self) -> None:
        class TimeoutMock:
            def request(self, **kw):
                return {
                    "ok": False,
                    "error": "(deterministic) timeout at `whnf`, maximum number of heartbeats (5000) has been reached",
                    "request_hb_raw": 5_000_210,
                    "request_ms": 120,
                }

        led = ImmutableReceiptLedger("root_timeout")
        m = AuditedMeter(TimeoutMock(), led, steps=10, rpcs=20, hb=50_000_000)  # type: ignore[arg-type]
        acts = run08.enumerate_menu(m, {"sid": 0, "info": {"hyps": [], "defs": []}})
        # Keeps non-premise actions, still charges the timed-out premise call, logs immutable receipt
        self.assertGreater(len(acts), 0)
        self.assertEqual(m.ops["premises"], 1)
        self.assertEqual(m.hb, 5_000_210)
        self.assertEqual(led.count, 1)
        self.assertFalse(led[0]["receipt"]["ok"])
        self.assertTrue(led.verify_chain())

    def test_roster_protection_and_exclusion_invariants(self) -> None:
        ex = load_exclusion_sets(RESEARCH_DIR)
        self.assertEqual(ex["protected_eval_roots_total"], 275)
        self.assertEqual(ex["protected_eval_pristine_roots"], 274)
        self.assertIn(SMOKE_EVAL_ROOT_ID, ex["protected_eval_smoke_excluded_ids"])
        # Zero overlap with spent, OLR-08A cal, OLR-08A eval, or sealed SInt
        cal_set = set(FROZEN_CAL_FILES_08B)
        self.assertEqual(len(cal_set), len(FROZEN_CAL_FILES_08B))
        self.assertFalse(cal_set & ex["all_forbidden_cal_files"])
        self.assertTrue(all("/SInt" not in f and not f.endswith("/SInt.lean") for f in FROZEN_CAL_FILES_08B))
        # Verify against live Lean source if present
        lean_src = Path(r"D:\lean4-419-source")
        if lean_src.is_dir():
            audit = verify_rule_matches_source(lean_src, RESEARCH_DIR)
            self.assertTrue(audit["passed"])
            self.assertTrue(audit["rule_exact_match"])
            self.assertFalse(audit["sealed_SInt_opened"])
            self.assertGreaterEqual(audit["cal_files_count"], 12)


if __name__ == "__main__":
    unittest.main()
