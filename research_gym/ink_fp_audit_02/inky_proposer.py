"""Inky (Inkling-2-Mini, free Tinker tier) as the edit proposer for the MeTTa RAM hill climb.

One request per turn through the audit-01 OpenAI-compatible runner (free-model guard, no
retries, no redirects). An unparsable or out-of-grammar reply ends the climb and is recorded,
never retried. Stdlib only, so metta_driver.py can load it with DataGoblin on sys.path.
"""
from __future__ import annotations

import json
import os
import re

KEY_ORDER = ("solved_and_stopped (higher), new_violation_additions (lower), unsafe_additions (lower), "
             "exact_first_edit (higher), premature_stops (lower), verifier_queries (lower), "
             "mean_remaining_canonical_edits (lower), mean_inference_iterations (lower)")
ANSWER = re.compile(r"(?m)^ANSWER:\s*(\{.*\})\s*$")


def build_prompt(options: dict, current: dict, history: list[dict]) -> str:
    hist = [{"arm": h["arm"], "edit": h["proposal"], "decision": h["decision"],
             "solved_and_stopped": h["candidate_metrics"]["solved_and_stopped"],
             "exact_first_edit": h["candidate_metrics"]["exact_first_edit"],
             "premature_stops": h["candidate_metrics"]["premature_stops"]} for h in history[-12:]]
    grammar = {a: {f: list(v) for f, v in fs.items()} for a, fs in options.items()}
    return "\n".join([
        "You are proposing one hyperparameter edit in a hill climb over two tiny CPU students.",
        f"An edit is accepted only if it improves this lexicographic key on development cases: {KEY_ORDER}.",
        "Any rise in violations or unsafe additions is rejected.",
        f"Allowed values: {json.dumps(grammar)}",
        f"Current configs: {json.dumps(current, sort_keys=True)}",
        f"Recent steps: {json.dumps(hist)}",
        "Propose one edit that changes a single field to a different allowed value.",
        'Finish with a separate final line: ANSWER: {"arm": "...", "field": "...", "value": ...}',
    ])


def parse_edit(text: str) -> dict:
    m = ANSWER.findall(text or "")
    if not m:
        raise ValueError("no ANSWER line")
    e = json.loads(m[-1])
    return {"arm": e["arm"], "field": e["field"], "value": e["value"]}


class InkyProposer:
    name = "inky"

    def __init__(self, effort: str = "medium", max_tokens: int = 512, transport=None, key: str | None = None):
        from research_gym.ink_fp_audit_01 import tinker_remote as r01
        self.r01, self.effort, self.max_tokens = r01, effort, max_tokens
        self.transport = transport or r01.sample
        self.key = key or os.environ.get("TINKER_API_KEY")
        if not self.key:
            raise RuntimeError("TINKER_API_KEY not present in environment")
        self.invalid: list[dict] = []
        self.usage: list = []

    def propose(self, rh, current, history, tried):
        payload = self.r01.request_payload(build_prompt(rh.OPTIONS, current, history), self.effort, self.max_tokens)
        reply = self.transport(self.key, payload, 120)
        self.usage.append(reply.get("usage"))
        text = reply["choices"][0]["message"].get("content") or ""
        try:
            edit = parse_edit(text)
            rh.validate_proposal({**edit, "source": "inky", "model_id": "inkling-2-mini-free"}, current)
            return edit
        except Exception as exc:
            self.invalid.append({"turn": len(history), "error": type(exc).__name__, "reply_chars": len(text)})
            return None
