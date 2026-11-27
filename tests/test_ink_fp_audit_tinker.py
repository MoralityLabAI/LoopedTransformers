import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from research_gym.ink_fp_audit_01.tinker_remote import MODEL, plan, request_payload, run, score_response


class TinkerTests(unittest.TestCase):
    def test_budget(self):
        p=plan(8,512)
        self.assertEqual(p["request_cap"],24)
        self.assertEqual(p["max_requested_completion_tokens"],12288)
        self.assertFalse(p["sealed_readout"])
        self.assertEqual(p,plan(8,512))

    def test_bounds(self):
        for args in [(9,512),(2,2049),(2,0),(2,128,("low","low"))]:
            with self.assertRaises(ValueError):
                plan(*args)

    def test_payload(self):
        p=request_payload("test","high",512)
        self.assertEqual(p["model"],MODEL)
        self.assertEqual(p["reasoning_effort"],"high")
        self.assertNotIn("num_passes",p)

    def test_grade_final_only(self):
        r={"choices":[{"message":{"content":"ANSWER: 1","reasoning_content":"ANSWER: 2"},"finish_reason":"stop"}]}
        self.assertTrue(score_response(r,1)["correct"])
        self.assertFalse(score_response(r,2)["correct"])

    def test_success_fake_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"out"
            calls=[]
            def fake(key,payload,timeout):
                self.assertEqual(payload["model"],MODEL)
                calls.append(payload)
                return {"model":MODEL,"choices":[{"message":{"content":"ANSWER: 1"}}]}
            run(plan(),path,"fake",fake,delay=0)
            self.assertEqual(len(calls),6)
            self.assertEqual(json.loads((path/"complete.json").read_text())["requests"],6)
            self.assertNotIn("fake",(path/"records.jsonl").read_text())

    def test_429_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"out"
            calls=[]
            def fake(key,payload,timeout):
                calls.append(payload)
                raise urllib.error.HTTPError("https://example.com",429,"rate",None,None)
            with self.assertRaises(RuntimeError):
                run(plan(),path,"fake",fake,delay=0)
            self.assertEqual(len(calls),1)
            failure=json.loads((path/"failure.json").read_text())
            self.assertEqual(failure["http_status"],429)
            self.assertFalse(failure["paid_fallback"])

    def test_expanded_jobs_blocked(self):
        p=plan()
        p["jobs"].append(p["jobs"][0])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run(p,Path(tmp)/"out","fake",delay=0)


if __name__ == "__main__":
    unittest.main()
