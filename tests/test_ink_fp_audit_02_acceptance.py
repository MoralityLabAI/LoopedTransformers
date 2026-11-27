"""Independent replay of the acceptance ledger and the upstream discrepancy fixture.

The replay here deliberately re-implements the decision rule from the registered criteria
fields instead of calling acceptance.decide, so a bug in decide cannot hide itself.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

from research_gym.ink_fp_audit_02 import acceptance as acc
from research_gym.ink_fp_audit_02.ledger import Ledger

FIXTURE = Path(__file__).resolve().parents[1] / "research_gym/ink_fp_audit_02/fixtures/upstream_gspo_gate_765453d.json"


def cases(spec):
    """spec: {family: [(correct, completed, repetition), ...]}"""
    out = []
    for fam, rows in spec.items():
        for i, (c, comp, rep) in enumerate(rows):
            out.append({"task_id": f"{fam}-{i}", "family": fam, "correct": c, "completed": comp, "repetition": rep})
    return out


def independent_decision(crit, pred, cand):
    by_id = {c["task_id"]: c for c in pred}
    assert set(by_id) == {c["task_id"] for c in cand}
    fams = sorted({c["family"] for c in pred})
    gain = sum(c["correct"] for c in cand) - sum(c["correct"] for c in pred)
    ok = gain >= crit["min_primary_gain"]
    for f in fams:
        p = [c for c in pred if c["family"] == f]
        q = [c for c in cand if c["family"] == f]
        loss = sum(c["correct"] for c in p) - sum(c["correct"] for c in q)
        ok &= loss <= max(1, math.ceil(crit["max_family_loss_frac"] * len(p)))
    n = len(pred)
    rep = lambda xs: sum(bool(c["repetition"]) for c in xs) / n
    comp = lambda xs: sum(bool(c["completed"]) for c in xs) / n
    ok &= rep(cand) <= rep(pred) + crit["max_repetition_increase"] + 1e-12
    ok &= comp(cand) >= comp(pred) - crit["max_completion_drop"] - 1e-12
    return bool(ok)


def independent_replay(path):
    heads, out = {}, []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(raw)
        b = rec["body"]
        if rec["kind"] == "base":
            heads[b["arm"]] = b["checkpoint"]
        elif rec["kind"] == "acceptance":
            assert b["predecessor_checkpoint"] == heads[b["arm"]], "predecessor is not accepted head"
            ok = independent_decision(b["criteria"], b["predecessor_cases"], b["candidate_cases"])
            assert ok == b["decision"]["accepted"]
            heads[b["arm"]] = b["candidate_checkpoint"] if ok else b["predecessor_checkpoint"]
            assert heads[b["arm"]] == b["next_head"]
            out.append(ok)
    return out, heads


class DecisionTests(unittest.TestCase):
    C = acc.Criteria()

    def test_accept_gain(self):
        p = cases({"a": [(0, 1, 0)] * 10, "b": [(1, 1, 0)] * 10})
        q = cases({"a": [(1, 1, 0)] * 2 + [(0, 1, 0)] * 8, "b": [(1, 1, 0)] * 10})
        self.assertTrue(acc.decide(self.C, p, q)["accepted"])

    def test_reject_no_gain(self):
        p = cases({"a": [(1, 1, 0)] * 5})
        self.assertFalse(acc.decide(self.C, p, p)["accepted"])

    def test_reject_family_regression(self):
        p = cases({"a": [(1, 1, 0)] * 10, "b": [(0, 1, 0)] * 10})
        q = cases({"a": [(0, 1, 0)] * 2 + [(1, 1, 0)] * 8, "b": [(1, 1, 0)] * 10})
        d = acc.decide(self.C, p, q)
        self.assertFalse(d["accepted"])
        self.assertFalse(d["checks"]["family_regression_ok"])

    def test_reject_repetition(self):
        p = cases({"a": [(0, 1, 0)] * 20})
        q = cases({"a": [(1, 1, 1)] * 10 + [(0, 1, 0)] * 10})
        self.assertFalse(acc.decide(self.C, p, q)["checks"]["repetition_ok"])

    def test_reject_completion(self):
        p = cases({"a": [(0, 1, 0)] * 20})
        q = cases({"a": [(1, 1, 0)] * 5 + [(0, 0, 0)] * 15})
        self.assertFalse(acc.decide(self.C, p, q)["checks"]["completion_ok"])

    def test_unpaired_rejected(self):
        p = cases({"a": [(0, 1, 0)] * 3})
        with self.assertRaises(ValueError):
            acc.decide(self.C, p, p[:2])


class LedgerReplayTests(unittest.TestCase):
    def _ledger(self, tmp):
        led = Ledger(Path(tmp) / "ledger.jsonl")
        acc.register_base(led, "D", "base")
        p0 = cases({"a": [(0, 1, 0)] * 10, "b": [(1, 1, 0)] * 10})
        good = cases({"a": [(1, 1, 0)] * 3 + [(0, 1, 0)] * 7, "b": [(1, 1, 0)] * 10})
        bad = cases({"a": [(1, 1, 0)] * 10, "b": [(0, 1, 0)] * 10})
        acc.record_candidate(led, self.C, generation=0, arm="D", predecessor_checkpoint="base",
                             candidate_checkpoint="g0", predecessor_cases=p0, candidate_cases=good,
                             hyperparameters={"lr": 1e-5}, scales_examined=[1.0])
        acc.record_candidate(led, self.C, generation=1, arm="D", predecessor_checkpoint="g0",
                             candidate_checkpoint="g1", predecessor_cases=good, candidate_cases=bad,
                             hyperparameters={"lr": 1e-5}, scales_examined=[1.0, 0.5])
        return led

    C = acc.Criteria()

    def test_replay_matches_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = self._ledger(tmp)
            decisions, heads = independent_replay(led.path)
            self.assertEqual(decisions, [True, False])
            self.assertEqual(heads["D"], "g0")
            r = acc.replay(led)
            self.assertEqual(r["mismatches"], [])
            self.assertEqual(r["heads"], heads)

    def test_rejected_candidate_cannot_be_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = self._ledger(tmp)
            p = cases({"a": [(0, 1, 0)] * 2})
            with self.assertRaises(ValueError):
                acc.record_candidate(led, self.C, generation=2, arm="D", predecessor_checkpoint="g1",
                                     candidate_checkpoint="g2", predecessor_cases=p, candidate_cases=p,
                                     hyperparameters={}, scales_examined=[])

    def test_tampered_verdict_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = self._ledger(tmp)
            lines = led.path.read_text(encoding="utf-8").splitlines()
            rec = json.loads(lines[-1])
            rec["body"]["decision"]["accepted"] = True
            lines[-1] = json.dumps(rec, sort_keys=True)
            led.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):  # hash chain catches the edit
                list(led.iter_verified())


class UpstreamFixtureTests(unittest.TestCase):
    def test_four_discrepancies(self):
        record = json.loads(FIXTURE.read_text(encoding="utf-8"))
        rows = acc.audit_upstream(record)
        self.assertEqual([r["gen"] for r in rows if r["discrepancy"]], [1, 2, 3, 4])
        self.assertTrue(all(r["reported_accepted"] for r in rows))
        self.assertEqual([r["candidate_total"] for r in rows], [112, 114, 118, 114, 116])

    def test_independent_mode_drops(self):
        record = json.loads(FIXTURE.read_text(encoding="utf-8"))
        prev = {m: record["init_gate"][m]["correct"] for m in ("R3", "R4", "RK2")}
        worst = []
        for g in record["generations"]:
            w = [c for c in g["candidates"] if c["scale"] == g["winner_scale"]][0]
            cur = {m: round(w[m] * 64 / 100) for m in ("R3", "R4", "RK2")}
            worst.append(max(prev[m] - cur[m] for m in prev))
            prev = cur
        self.assertEqual(worst, [1, 4, 3, 6, 4])  # rule allows <= 2

    def test_our_gate_with_upstream_chain(self):
        """Feeding the upstream reported chain through our ledger records the rejections."""
        record = json.loads(FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            led = Ledger(Path(tmp) / "l.jsonl")
            led.append("upstream_audit", {"rows": acc.audit_upstream(record)})
            body = led.last("upstream_audit")
            self.assertEqual(sum(r["discrepancy"] for r in body["rows"]), 4)


if __name__ == "__main__":
    unittest.main()
