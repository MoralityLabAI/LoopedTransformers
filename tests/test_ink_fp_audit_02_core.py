import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research_gym.ink_fp_audit_02 import resources as res
from research_gym.ink_fp_audit_02 import rewards as rw
from research_gym.ink_fp_audit_02 import tinker_backend as tb
from research_gym.ink_fp_audit_02.ledger import Ledger, file_sha256

GIB = res.GIB


class GovernorTests(unittest.TestCase):
    def test_pause_resume_hysteresis(self):
        g = res.Governor()
        self.assertEqual(g.observe(7 * GIB, 0), res.RUN)
        self.assertEqual(g.observe(int(2.9 * GIB), 1), res.PAUSE)
        self.assertEqual(g.observe(4 * GIB, 2), res.PAUSE)        # between 3 and 5: stay paused
        self.assertEqual(g.observe(int(5.1 * GIB), 10), res.PAUSE)
        self.assertEqual(g.observe(int(5.1 * GIB), 69), res.PAUSE)
        self.assertEqual(g.observe(int(4.9 * GIB), 70), res.PAUSE)  # dip resets the 60 s window
        self.assertEqual(g.observe(int(5.2 * GIB), 80), res.PAUSE)
        self.assertEqual(g.observe(int(5.2 * GIB), 140), res.RUN)
        self.assertEqual(g.min_available, int(2.9 * GIB))

    def test_emergency_sticky(self):
        g = res.Governor()
        self.assertEqual(g.observe(int(1.4 * GIB), 0), res.EMERGENCY)
        self.assertEqual(g.observe(8 * GIB, 100), res.EMERGENCY)

    def test_monitor_raises(self):
        m = res.ResourceMonitor(reader=lambda: int(1.0 * GIB), sleeper=lambda s: None)
        with self.assertRaises(res.EmergencyStop):
            m.gate()

    def test_monitor_waits_then_runs(self):
        seq = iter([int(2 * GIB)] + [int(6 * GIB)] * 20)
        clock = iter(range(0, 1000, 10))
        m = res.ResourceMonitor(reader=lambda: next(seq), sleeper=lambda s: None, clock=lambda: next(clock))
        m.gate()
        self.assertEqual(m.governor.state, res.RUN)

    def test_startup_check(self):
        self.assertFalse(res.startup_check(available=int(5.6 * GIB))["ok"])
        self.assertTrue(res.startup_check(available=7 * GIB)["ok"])
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(res.gpu_allocation_bytes(), 0)


@unittest.skipUnless(sys.platform == "win32", "Job Objects are Windows-only")
class JobObjectTests(unittest.TestCase):
    def test_child_allocation_capped(self):
        """A child in a 64 MiB job cannot commit 256 MiB; a 16 MiB allocation succeeds."""
        code = ("import sys,time;sys.stdin.readline();"
                "a=bytearray(16*2**20);print('small-ok',flush=True)\n"
                "try:\n b=bytearray(256*2**20);print('big-ok',flush=True)\n"
                "except MemoryError:\n print('big-denied',flush=True)")
        job = res.JobObject(64 * res.MIB)
        try:
            p = subprocess.Popen([sys.executable, "-I", "-c", code], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
            job.assign_pid(p.pid)
            out, _ = p.communicate("go\n", timeout=60)
            self.assertIn("small-ok", out)
            self.assertIn("big-denied", out)
            self.assertNotIn("big-ok", out)
            # PeakJobMemoryUsed is readable; on this host it also counts the refused commit
            # attempt, so it is not asserted to stay below the cap.
            self.assertGreater(job.peak_job_memory(), 0)
        finally:
            job.close()


class RewardTests(unittest.TestCase):
    def s(self, effort, correct, completed=True, rep=False, tok=100):
        return {"effort": effort, "correct": correct, "completed": completed, "repetition": rep,
                "completion_tokens": tok}

    def test_rescue_only_when_low_fails(self):
        g = [self.s("low", 0), self.s("low", 0), self.s("medium", 0), self.s("medium", 1),
             self.s("high", 1), self.s("high", 0)]
        self.assertEqual(rw.rewards(g), [0, 0, 0, 1.0, 1.35, 0])
        g2 = [self.s("low", 1)] + g[1:]
        self.assertEqual(rw.rewards(g2)[4], 1.0)  # low already solved: no rescue

    def test_rescue_is_not_a_length_reward(self):
        g = [self.s("low", 0, tok=10), self.s("high", 1, tok=10), self.s("high", 1, tok=500)]
        r = rw.rewards(g)
        self.assertEqual(r[1], r[2])

    def test_penalties_and_cost(self):
        self.assertEqual(rw.rewards([self.s("low", 0, completed=False, rep=True)]), [-0.75])
        cfg = rw.RewardConfig(token_cost_per_1k=0.1)
        self.assertAlmostEqual(rw.rewards([self.s("high", 1, tok=500)], cfg)[0], 0.95)

    def test_shuffled_control_preserves_bonus_count(self):
        g = [self.s("low", 0), self.s("low", 0), self.s("high", 1), self.s("high", 1)]
        base = sum(rw.rewards(g))
        import random
        sh = rw.rewards(g, rw.RewardConfig(shuffle_rescue=True), random.Random(3))
        self.assertAlmostEqual(sum(sh), base)

    def test_advantages(self):
        a = rw.group_advantages([1, 0, 0, 1])
        self.assertAlmostEqual(sum(a), 0, places=9)
        self.assertEqual(rw.group_advantages([1, 1]), [0.0, 0.0])
        self.assertTrue(rw.is_frontier([self.s("low", 1), self.s("low", 0)]))
        self.assertFalse(rw.is_frontier([self.s("low", 1)]))

    def test_gspo_ratio(self):
        self.assertAlmostEqual(rw.gspo_ratio([-1.0, -2.0], [-1.0, -2.0]), 1.0)
        self.assertAlmostEqual(rw.gspo_ratio([-0.9, -1.9], [-1.0, -2.0]), 2.718281828 ** 0.1, places=6)
        self.assertAlmostEqual(rw.gspo_objective(2.0, 1.0), 1.0004)


class BackendGuardTests(unittest.TestCase):
    def test_paid_model_blocked(self):
        with self.assertRaises(tb.PaidModelBlocked):
            tb.guard_model("thinkingmachines/Inkling-2-Mini")
        self.assertEqual(tb.guard_model(tb.MODEL), tb.MODEL)

    def test_context_ceiling(self):
        tb.guard_context(11_000, 512)
        with self.assertRaises(ValueError):
            tb.guard_context(12_000, 512)
        with self.assertRaises(ValueError):
            tb.guard_context(10, 10, limit=20_000)

    def test_oai_sampler_with_fake_transport(self):
        u = tb.Usage()
        def fake(key, payload, timeout):
            self.assertEqual(payload["model"], tb.MODEL)
            return {"model": tb.MODEL, "choices": [{"message": {"content": "ANSWER: 3"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 7}}
        s = tb.OAISampler(u, transport=fake)
        r = s.sample("q", "high", 512, "k")
        self.assertEqual(r["text"], "ANSWER: 3")
        self.assertEqual((u.sample_requests, u.prompt_tokens, u.completion_tokens), (1, 40, 7))
        self.assertEqual(u.paid_spend_usd, 0.0)

    def test_sdk_capabilities_static(self):
        caps = tb.sdk_capabilities()
        if caps["installed"]:
            self.assertIn("gspo", caps["loss_fns"])


class LedgerTests(unittest.TestCase):
    def test_resume_and_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "l.jsonl"
            a = Ledger(p)
            a.append("x", {"i": 1})
            a.append("x", {"i": 2})
            b = Ledger(p)
            self.assertEqual(b.seq, 2)
            b.append("y", {"i": 3})
            self.assertEqual([r["seq"] for r in Ledger(p).iter_verified()], [0, 1, 2])

    def test_torn_tail_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "l.jsonl"
            Ledger(p).append("x", {"i": 1})
            with p.open("a", encoding="utf-8") as f:
                f.write('{"seq": 1, "kind"')
            self.assertEqual(Ledger(p).seq, 1)

    def test_lf_normalised_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.write_bytes(b"x\ny\n")
            b.write_bytes(b"x\r\ny\r\n")
            self.assertEqual(file_sha256(a), file_sha256(b))


if __name__ == "__main__":
    unittest.main()
