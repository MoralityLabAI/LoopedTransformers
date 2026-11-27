"""Inky proposer for the MeTTa hill climb, exercised with fake transports (no network)."""
import unittest

from research_gym.ink_fp_audit_01.tinker_remote import MODEL
from research_gym.ink_fp_audit_02 import inky_proposer as ip


class FakeRH:
    OPTIONS = {"trm": {"lr": (0.08, 0.2), "steps": (16, 32)}, "ldt_proxy": {"lr": (0.08, 0.2)}}

    @staticmethod
    def validate_proposal(p, current):
        if p["field"] not in FakeRH.OPTIONS[p["arm"]] or p["value"] not in FakeRH.OPTIONS[p["arm"]][p["field"]]:
            raise ValueError("outside grammar")
        if current[p["arm"]][p["field"]] == p["value"]:
            raise ValueError("no-op")
        return {**current[p["arm"]], p["field"]: p["value"]}


CURRENT = {"trm": {"lr": 0.2, "steps": 16}, "ldt_proxy": {"lr": 0.2}}


def reply(text):
    return {"model": MODEL, "choices": [{"message": {"content": text}}], "usage": {"completion_tokens": 9}}


class InkyProposerTests(unittest.TestCase):
    def test_valid_edit_and_free_model(self):
        sent = []

        def fake(key, payload, timeout):
            sent.append(payload)
            return reply('thinking\nANSWER: {"arm": "trm", "field": "steps", "value": 32}')
        p = ip.InkyProposer(transport=fake, key="k")
        self.assertEqual(p.propose(FakeRH, CURRENT, [], set()), {"arm": "trm", "field": "steps", "value": 32})
        self.assertEqual(sent[0]["model"], MODEL)
        self.assertEqual(sent[0]["reasoning_effort"], "medium")
        self.assertIn("lexicographic key", sent[0]["messages"][0]["content"])
        self.assertNotIn("k", [v for v in sent[0].values() if isinstance(v, str)])

    def test_out_of_grammar_and_noop_recorded_not_retried(self):
        calls = []
        for text in ('ANSWER: {"arm": "trm", "field": "lr", "value": 0.5}',
                     'ANSWER: {"arm": "trm", "field": "lr", "value": 0.2}', "no answer line"):
            def fake(key, payload, timeout, text=text):
                calls.append(1)
                return reply(text)
            p = ip.InkyProposer(transport=fake, key="k")
            self.assertIsNone(p.propose(FakeRH, CURRENT, [], set()))
            self.assertEqual(len(p.invalid), 1)
        self.assertEqual(len(calls), 3)

    def test_requires_key(self):
        import os
        old = os.environ.pop("TINKER_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                ip.InkyProposer(transport=lambda *a: None)
        finally:
            if old is not None:
                os.environ["TINKER_API_KEY"] = old

    def test_prompt_lists_history(self):
        hist = [{"arm": "trm", "proposal": {"arm": "trm", "field": "lr", "value": 0.08}, "decision": "reject_no_gain",
                 "candidate_metrics": {"solved_and_stopped": 0, "exact_first_edit": 1, "premature_stops": 15}}]
        text = ip.build_prompt(FakeRH.OPTIONS, CURRENT, hist)
        self.assertIn("reject_no_gain", text)
        self.assertTrue(text.rstrip().endswith('"value": ...}'))


if __name__ == "__main__":
    unittest.main()
