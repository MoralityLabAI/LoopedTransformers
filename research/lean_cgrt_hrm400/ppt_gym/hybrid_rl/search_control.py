"""Shared decision rules for verifier-backed proof search."""

import json


class DirectFactContinuation:
    """Restrict a continuation to available exact local proofs when present."""

    def __init__(self, policy):
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def __call__(self, packet, menu, *args, **kwargs):
        logits, value = self.policy(packet, menu, *args, **kwargs)
        public = json.loads(packet)
        if public["checks_remaining"] == 8:
            return logits, value
        goal = public["state"]["goals"][0]
        direct = {f"apply {hyp['name']}" for hyp in goal["hypotheses"] if hyp["type"] == goal["target"]}
        if direct.intersection(menu):
            keep = logits.new_tensor([name in direct for name in menu]).bool()
            logits = logits.masked_fill(~keep, float("-inf"))
        return logits, value
