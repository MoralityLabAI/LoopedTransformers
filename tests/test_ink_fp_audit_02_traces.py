import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

from research_gym.ink_fp_audit_02 import acceptance as acc
from research_gym.ink_fp_audit_02 import curriculum, engine, gyms, hillclimb, traces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ink_fp_audit_02_acceptance import FIXTURE, independent_decision  # noqa: E402


class UpstreamTraceTests(unittest.TestCase):
    def setUp(self):
        self.rows = list(traces.from_upstream_record(json.loads(FIXTURE.read_text(encoding="utf-8"))))

    def test_every_candidate_is_a_step(self):
        self.assertEqual(len(self.rows), 9)
        self.assertEqual(sum(r["proposal"]["selected_as_winner"] for r in self.rows), 5)

    def test_discrepancies_on_winners_only(self):
        disc = [r["proposal"]["id"] for r in self.rows if r["discrepancy"]]
        self.assertEqual(disc, ["gen1-scale0.6", "gen2-scale0.6", "gen3-scale1.0", "gen4-scale1.0"])
        self.assertTrue(all(r["decision_recorded"] is None for r in self.rows if not r["proposal"]["selected_as_winner"]))

    def test_gen0_rule_explains_scale_choice(self):
        g0 = {r["proposal"]["id"]: r for r in self.rows[:2]}
        self.assertEqual(g0["gen0-scale1.0"]["binding_checks"], ["mode_drop_ok"])
        self.assertEqual(g0["gen0-scale0.6"]["decision_by_rule"], "accept")

    def test_logic_is_consistent_with_checks(self):
        for r in self.rows:
            fails = [ln for ln in r["logic"] if ln.endswith(": fail")]
            self.assertEqual(len(fails), len(r["binding_checks"]))
            self.assertIn(f"decision by published rule: {r['decision_by_rule']}", r["logic"])


class EngineTraceTests(unittest.TestCase):
    def test_engine_ledger_traces_match_independent_rule(self):
        P = curriculum.build_partitions(["dyck", "modular_arith"], {"accept": 6, "train": 6})
        with tempfile.TemporaryDirectory() as tmp:
            engine.run_arm(engine.ARMS["D"], engine.SimulatedBackend(["dyck", "modular_arith"], seed=2),
                           P["partitions"]["train"], P["partitions"]["accept"], Path(tmp), 3)
            led = Path(tmp) / "ledger.jsonl"
            steps = list(traces.from_acceptance_ledger(led, "t", "engineering_simulation"))
            from research_gym.ink_fp_audit_02.ledger import Ledger
            bodies = list(Ledger(led).kinds("acceptance"))
            self.assertEqual(len(steps), len(bodies))
            for s, b in zip(steps, bodies):
                ok = independent_decision(b["criteria"], b["predecessor_cases"], b["candidate_cases"])
                self.assertEqual(s["decision_by_rule"], "accept" if ok else "reject")
                self.assertFalse(s["discrepancy"])
                row = traces.to_research_row(s)
                self.assertEqual(row["target"]["decision"], s["decision_by_rule"])
                self.assertEqual(row["kind"], "research")


class ContinuousRuleTests(unittest.TestCase):
    C = {"min_mean_gain": 0.01, "max_family_drop": 0.05}

    def cs(self, vals):
        return [{"case_id": f"{f}-{i}", "family": f, "value": v} for f, row in vals.items() for i, v in enumerate(row)]

    def test_accept_and_family_veto(self):
        a = self.cs({"x": [0.0, 0.0], "y": [0.0, 0.0]})
        self.assertTrue(traces.continuous_decide(self.C, a, self.cs({"x": [0.1, 0.1], "y": [0.0, 0.0]}))["accepted"])
        d = traces.continuous_decide(self.C, a, self.cs({"x": [0.5, 0.5], "y": [-0.1, -0.1]}))
        self.assertFalse(d["accepted"])
        self.assertFalse(d["checks"]["family_regression_ok"])

    def test_unpaired(self):
        with self.assertRaises(ValueError):
            traces.continuous_decide(self.C, self.cs({"x": [0]}), self.cs({"y": [0]}))


class HillclimbTests(unittest.TestCase):
    def test_short_climb_traces_replay(self):
        r = hillclimb.climb(steps=3, dev_seeds=range(300, 302), heldout_seeds=range(400, 402), seed=0)
        rows = r["rows"]
        self.assertTrue(rows)
        inc = json.dumps(hillclimb.START, sort_keys=True)
        for s in rows:
            self.assertEqual(s["incumbent"]["id"], inc)  # chain: each step judges the current incumbent
            inc = s["next_incumbent"]
            self.assertEqual(s["next_incumbent"] == s["proposal"]["id"], s["decision_by_rule"] == "accept")
        self.assertEqual(json.dumps(r["report"]["final"], sort_keys=True), inc)


class JudgmentFamilyTests(unittest.TestCase):
    def test_labels_recomputed_from_certificate(self):
        ts = list(curriculum.generate("hillclimb_judgment", "dev", 40))
        labels = {t.answer for t in ts}
        self.assertGreaterEqual(len(labels), 4)
        for t in ts:
            ch = t.certificate["checks"]
            exp = "ACCEPT" if all(ch.values()) else "REJECT " + next(
                n for k, n in (("primary_gain_ok", "primary_gain"), ("family_regression_ok", "family_regression"),
                               ("repetition_ok", "repetition"), ("completion_ok", "completion")) if not ch[k])
            self.assertEqual(t.answer, exp)
            self.assertTrue(curriculum.verify(t, "ANSWER: " + t.answer)["correct"])
            for fam, loss in t.certificate["family_losses"].items():
                self.assertEqual(t.certificate["family_limits"][fam] >= 1, True)


if __name__ == "__main__":
    unittest.main()


class MettaTraceTests(unittest.TestCase):
    BASE = {"solved_and_stopped": 0, "new_violation_additions": 0, "unsafe_additions": 0, "exact_first_edit": 2,
            "premature_stops": 3, "verifier_queries": 0, "mean_remaining_canonical_edits": 2.5,
            "mean_inference_iterations": 4.0}

    def test_first_differing_component_decides(self):
        new = {**self.BASE, "exact_first_edit": 3, "premature_stops": 9}
        checks, logic, d = traces.metta_logic(self.BASE, new)
        self.assertEqual(d, "accept")
        self.assertTrue(any("exact_first_edit" in ln and "better" in ln for ln in logic))

    def test_safety_veto_beats_improvement(self):
        new = {**self.BASE, "solved_and_stopped": 5, "unsafe_additions": 1}
        checks, _, d = traces.metta_logic(self.BASE, new)
        self.assertEqual(d, "reject")
        self.assertFalse(checks["no_new_unsafe"])

    def test_tie_is_reject_and_rounding_matches_select_key(self):
        new = {**self.BASE, "mean_inference_iterations": 4.0001}  # rounds to the same millis
        self.assertEqual(traces.metta_logic(self.BASE, new)[2], "reject")

    def test_committed_metta_traces_agree_with_recorded(self):
        R = Path(__file__).resolve().parents[1] / "research_gym/ink_fp_audit_02/receipts/metta_hillclimb_20261128"
        files = sorted(R.glob("metta_*_c*.jsonl"))
        if not files:
            self.skipTest("no MeTTa receipts")
        for f in files:
            for s in traces.from_metta_steps(f, "t"):
                self.assertFalse(s["discrepancy"], (f.name, s["step"]))


class CurationTraceTests(unittest.TestCase):
    def rec(self, root, window, sel, reason, pg, rg, acc_, before=1.0, after=0.9):
        return {"provenance": {"root": root, "window": window, "experiment": "x", "arm": "D"},
                "input": {"edit_slot": 0}, "mechanistic_features": {"predicted_gain": pg},
                "outcome": {"selected": sel, "selection_reason": reason, "realized_gain": rg,
                            "sign_match": (pg > 0) == (rg > 0), "window_accepted": acc_,
                            "window_cal_before": before, "window_cal_after": after}}

    def test_window_rule_and_regret(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.jsonl"
            rows = [self.rec(1, 0, True, "ranked", 0.5, 0.2, True), self.rec(1, 0, False, "rejected_pool", -0.1, 0.4, True),
                    self.rec(1, 1, True, "ranked", 0.5, 0.1, True, before=1.0, after=1.1)]
            p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            steps = list(traces.from_curation_decisions(p))
            self.assertEqual([s["decision_by_rule"] for s in steps], ["accept", "reject"])
            self.assertAlmostEqual(steps[0]["evaluation"]["metrics"]["regret_per_slot"], 0.2)
            self.assertFalse(steps[0]["discrepancy"])
            self.assertTrue(steps[1]["discrepancy"])  # recorded accept, rule says reject
