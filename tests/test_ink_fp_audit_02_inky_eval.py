import json
import os
import tempfile
import unittest
import urllib.request
from pathlib import Path

from research_gym.ink_fp_audit_01 import tinker_remote as r01
from research_gym.ink_fp_audit_02 import curriculum, inky_eval


class UserAgentTests(unittest.TestCase):
    def test_runner_sends_generic_user_agent(self):
        seen = {}

        class FakeRes:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n):
                return json.dumps({"choices": [{"message": {"content": "ANSWER: 1"}}]}).encode()

        class FakeOpener:
            def open(self, req, timeout):
                seen["ua"] = req.get_header("User-agent")
                seen["auth"] = req.get_header("Authorization")
                return FakeRes()

        orig = urllib.request.build_opener
        urllib.request.build_opener = lambda *a: FakeOpener()
        try:
            r01.sample("k", r01.request_payload("q", "low", 16))
        finally:
            urllib.request.build_opener = orig
        self.assertTrue(seen["ua"].startswith("ink-fp-audit/"))
        self.assertNotIn("urllib", seen["ua"].lower())


class InkyEvalTests(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.get("TINKER_API_KEY")
        os.environ["TINKER_API_KEY"] = "test-key"

    def tearDown(self):
        if self.old is None:
            os.environ.pop("TINKER_API_KEY", None)
        else:
            os.environ["TINKER_API_KEY"] = self.old

    def tasks(self):
        return list(curriculum.generate("modular_arith", "dev", 2))

    def test_resume_and_grading(self):
        ts = self.tasks()
        answers = {t.prompt: t.answer for t in ts}

        def fake(key, payload, timeout):
            self.assertEqual(payload["model"], r01.MODEL)
            ans = answers[payload["messages"][0]["content"]] if payload["reasoning_effort"] != "low" else "999"
            return {"model": r01.MODEL, "choices": [{"message": {"content": f"work\nANSWER: {ans}"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 10}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            r = inky_eval.run(ts, out, delay=0, max_requests=3, transport=fake)
            self.assertEqual(r["sent"], 3)
            r = inky_eval.run(ts, out, delay=0, transport=fake)
            self.assertEqual(r["sent"], 3)  # only the remaining pairs
            s = inky_eval.summarize(out / "records.jsonl")
            self.assertEqual(s["paired_tasks"], 2)
            self.assertEqual(s["rescued_high_over_low"], 2)
            self.assertEqual(s["totals"]["low"]["correct"], 0)
            self.assertNotIn("test-key", (out / "records.jsonl").read_text(encoding="utf-8"))

    def test_http_error_stops_without_retry(self):
        import urllib.error
        calls = []

        def fake(key, payload, timeout):
            calls.append(1)
            raise urllib.error.HTTPError("u", 429, "rate", None, None)
        with tempfile.TemporaryDirectory() as tmp:
            r = inky_eval.run(self.tasks(), Path(tmp), delay=0, transport=fake)
            self.assertEqual(len(calls), 1)
            self.assertIn("429", r["stopped"])
            f = json.loads((Path(tmp) / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual((f["http_status"], f["automatic_retries"], f["paid_fallback"]), (429, 0, False))


if __name__ == "__main__":
    unittest.main()
