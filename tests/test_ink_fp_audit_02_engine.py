import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

from research_gym.ink_fp_audit_02 import acceptance as acc
from research_gym.ink_fp_audit_02 import bpd_proxy, curriculum, engine, exports, gyms, stats
from research_gym.ink_fp_audit_02.ledger import Ledger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ink_fp_audit_02_acceptance import independent_replay  # noqa: E402

FAST = ["modular_arith", "dyck", "pointer_chase", "register_overwrite"]


def pools():
    P = curriculum.build_partitions(FAST, {"sealed": 4, "accept": 8, "train": 8})
    return P["partitions"]["train"], P["partitions"]["accept"], P


class DegradingBackend(engine.SimulatedBackend):
    """Every update lowers every skill: all candidates must be rejected and rolled back."""

    def train(self, arm, items):
        for f in self.skill:
            self.skill[f] -= 1.0
        return {"degrading": True}


class FlakyBackend(engine.SimulatedBackend):
    def __init__(self, *a, fail_at=None, **k):
        super().__init__(*a, **k)
        self.fail_at = fail_at

    def sample(self, task, effort, n, seed):
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise ConnectionError("simulated API outage")
        return super().sample(task, effort, n, seed)


class GymTests(unittest.TestCase):
    def test_all_families_verify_reference(self):
        for fam in gyms.FAMILIES:
            if fam == "mcp_retrieval":
                continue  # subprocess generator; covered in test_mcp_generator
            for t in curriculum.generate(fam, "dev", 5):
                self.assertTrue(curriculum.verify(t, "x\nANSWER: " + t.answer)["correct"], t.task_id)
                self.assertFalse(curriculum.verify(t, "ANSWER: " + t.wrong_answer(random.Random(0)))["correct"])
                for k in ("source_root", "source_world", "generator", "generator_version", "verifier",
                          "difficulty", "verify_cost_ms", "partition"):
                    self.assertIsNotNone(getattr(t, k))

    def test_mcp_generator(self):
        try:
            ts = list(curriculum.generate("mcp_retrieval", "dev", 2))
        except Exception as exc:  # HybridTRMLDT checkout absent (CI)
            self.skipTest(f"HybridTRMLDT unavailable: {type(exc).__name__}")
        for t in ts:
            self.assertTrue(curriculum.verify(t, "ANSWER: " + t.answer)["correct"])

    def test_grader_requires_single_final_answer(self):
        t = next(curriculum.generate("modular_arith", "dev", 1))
        self.assertFalse(curriculum.verify(t, f"ANSWER: {t.answer}\nANSWER: {t.answer}")["completed"])
        self.assertFalse(curriculum.verify(t, f"ANSWER: {t.answer}\nmore text")["completed"])
        rep = " ".join(["the same eight words repeat over and over"] * 6)
        self.assertFalse(curriculum.verify(t, rep + f"\nANSWER: {t.answer}")["correct"])

    def test_storyworld_plan_simulated(self):
        ts = [t for t in curriculum.generate("storyworld", "train", 30) if t.certificate["reachable"]]
        self.assertTrue(ts)
        t = ts[0]
        self.assertFalse(curriculum.verify(t, "ANSWER: wait")["correct"])
        self.assertFalse(curriculum.verify(t, "ANSWER: NONE")["correct"])

    def test_partitions_disjoint(self):
        _, _, P = pools()
        rep = curriculum.leakage_report(P["partitions"])
        self.assertTrue(rep["ok"], rep["collisions"][:3])
        train_ids = {t.task_id for t in P["partitions"]["train"]}
        self.assertFalse(train_ids & set(P["sealed_task_ids"]))
        self.assertNotIn("sealed", P["partitions"])

    def test_leakage_detected(self):
        t = next(curriculum.generate("dyck", "train", 1))
        rep = curriculum.leakage_report({"train": [t], "dev": [t]})
        self.assertFalse(rep["ok"])

    def test_world_partition_contract(self):
        for part in curriculum.PARTITIONS:
            t = next(curriculum.generate("pointer_chase", part, 1))
            self.assertEqual(curriculum.world_partition(t.source_world), part)


class EngineTests(unittest.TestCase):
    def test_run_and_independent_replay(self):
        train, accept, _ = pools()
        with tempfile.TemporaryDirectory() as tmp:
            r = engine.run_arm(engine.ARMS["D"], engine.SimulatedBackend(FAST, seed=3), train, accept, Path(tmp), 3)
            self.assertEqual(r["replay"]["mismatches"], [])
            decisions, heads = independent_replay(Path(tmp) / "ledger.jsonl")
            self.assertEqual(heads["D"], r["head"])
            self.assertEqual(len(decisions), len(r["replay"]["rows"]))

    def test_rejection_rolls_back(self):
        train, accept, _ = pools()
        with tempfile.TemporaryDirectory() as tmp:
            r = engine.run_arm(engine.ARMS["C"], DegradingBackend(FAST, seed=3), train, accept, Path(tmp), 3)
            self.assertEqual(r["head"], "base")
            rows = r["replay"]["rows"]
            self.assertTrue(rows and not any(x["accepted"] for x in rows))
            led = Ledger(Path(tmp) / "ledger.jsonl")
            preds = [b["predecessor_checkpoint"] for b in led.kinds("acceptance")]
            self.assertEqual(set(preds), {"base"})  # rejected candidates never become predecessors
            self.assertEqual(len(list(led.kinds("checkpoint"))), len(rows))  # rejected ones persisted
            independent_replay(led.path)

    def test_resume_after_outage(self):
        train, accept, _ = pools()
        with tempfile.TemporaryDirectory() as tmp:
            b = FlakyBackend(FAST, seed=5, fail_at=150)
            with self.assertRaises(ConnectionError):
                engine.run_arm(engine.ARMS["D"], b, train, accept, Path(tmp), 3)
            done_before = [x["generation"] for x in Ledger(Path(tmp) / "ledger.jsonl").kinds("generation_done")]
            fresh = engine.SimulatedBackend(FAST, seed=5)  # new process: checkpoints re-adopted from ledger
            r = engine.run_arm(engine.ARMS["D"], fresh, train, accept, Path(tmp), 3)
            done = [x["generation"] for x in Ledger(Path(tmp) / "ledger.jsonl").kinds("generation_done")]
            self.assertEqual(sorted(done), [0, 1, 2])
            self.assertEqual(done[:len(done_before)], done_before)
            self.assertEqual(r["replay"]["mismatches"], [])
            independent_replay(Path(tmp) / "ledger.jsonl")

    def test_frozen_and_unsupported(self):
        train, accept, _ = pools()
        with tempfile.TemporaryDirectory() as tmp:
            r = engine.run_arm(engine.ARMS["A"], engine.SimulatedBackend(FAST), train, accept, Path(tmp) / "a", 2)
            self.assertEqual(r["head"], "base")
            h = engine.run_arm(engine.ARMS["H"], engine.SimulatedBackend(FAST), train, accept, Path(tmp) / "h", 2)
            self.assertEqual(h["status"], "unsupported")

    def test_six_samples_two_per_effort(self):
        t = next(curriculum.generate("dyck", "train", 1))
        g = engine.sample_group(engine.SimulatedBackend(["dyck"]), t, 1)
        self.assertEqual([s["effort"] for s in g], ["low", "low", "medium", "medium", "high", "high"])

    def test_exports_shape_and_dedupe(self):
        train, accept, _ = pools()
        with tempfile.TemporaryDirectory() as tmp:
            engine.run_arm(engine.ARMS["D"], engine.SimulatedBackend(FAST, seed=3), train, accept, Path(tmp), 2)
            man = json.loads((Path(tmp) / "exports" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(man["schema_version"], "rlt_gym.aleph_corpus.v1")
            sft = [json.loads(x) for x in (Path(tmp) / "exports" / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
            keys = [(r["semantic_task_id"], r["root"]) for r in sft]
            self.assertEqual(len(keys), len(set(keys)))
            for r in sft:
                self.assertTrue(r["verification"]["passed"])
                self.assertEqual(r["split"], "train")
                self.assertIn("prompt", r["input"]["task"])

    def test_explanation_kept_separate(self):
        text = "OBJECTIVE: find x\nVERIFICATION: recompute\nANSWER: 3"
        e = exports.explanation(text)
        self.assertEqual(e, {"objective": "find x", "verification": "recompute"})


class StatsAndProxyTests(unittest.TestCase):
    def test_power_monotone_in_effect(self):
        a = stats.power_preflight(min_effect=0.10)["required_independent_roots"]
        b = stats.power_preflight(min_effect=0.20)["required_independent_roots"]
        c = stats.power_preflight(min_effect=0.10, comparisons=7)["required_independent_roots"]
        self.assertLess(b, a)
        self.assertGreater(c, a)

    def test_registration_uses_exact_decimals(self):
        r = stats.registration("x", 0.1, 2, preregistered=True, sealed=True)
        self.assertEqual((r["alpha"], r["minimum_effect"]), ("0.05", "0.1"))

    def test_projected_update_exact_and_naive_leak(self):
        import numpy as np
        d = bpd_proxy.make(bpd_proxy.Scenario(conflict=0.5), np.random.default_rng(0))
        D = bpd_proxy.m_projected(d)
        rel = np.linalg.norm(d["Xk"] @ D) / np.linalg.norm(d["Xk"] @ d["W"].T)
        self.assertLess(rel, 1e-12)
        self.assertLess(np.linalg.norm(d["Xk_test"] @ D), 1e-10)
        naive = bpd_proxy.m_projected_naive(d)
        self.assertGreater(np.linalg.norm(d["Xk"] @ naive) / np.linalg.norm(d["Xk"] @ d["W"].T), 1e-6)

    def test_float32_rounding_bound(self):
        import numpy as np
        d = bpd_proxy.make(bpd_proxy.Scenario(conflict=0.5, dtype="float32"), np.random.default_rng(1))
        m = bpd_proxy.metrics(d, bpd_proxy.m_projected(d))
        self.assertLess(m["keep_residual_rel"], 1e-4)

    def test_adaptive_rank_beats_uniform(self):
        r = bpd_proxy.adaptive_vs_uniform(0)
        self.assertGreaterEqual(r["captured_adaptive"], r["captured_uniform"])
        self.assertEqual(sum(r["allocation"]), 16)


if __name__ == "__main__":
    unittest.main()
