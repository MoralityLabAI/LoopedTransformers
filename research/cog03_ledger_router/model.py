"""A tied transformer cell routes public rule tokens in a finite proof graph."""
from __future__ import annotations

import torch
from torch import nn

from research.lean_cgrt_hrm400.ppt_gym.core import block


FEATURES = (
    "all_premises_known", "fraction_premises_known", "conclusion_known",
    "conclusion_is_goal", "premise_count_scaled", "consumers_scaled",
    "producers_scaled", "remaining_budget_fraction", "failed_here",
    "last_action_was_this_rule",
)


def encode_observation(observation, remaining, budget):
    """Metadata, labels, reference paths and goal ancestry never enter this view."""
    world = observation["world"]
    rules = world["rules"]
    known = set(observation["known"])
    history = observation.get("history", [])
    failed = set()
    for event in reversed(history):
        if event.get("new_fact") is not None:
            break
        if event.get("accepted") is False:
            failed.add(event["action"])
    m = len(rules)
    edges = torch.zeros((m, m), dtype=torch.bool)
    rows = []
    for i, rule in enumerate(rules):
        consumers = [j for j, other in enumerate(rules)
                     if rule["conclusion"] in other["premises"]]
        producers = [j for j, other in enumerate(rules)
                     if other["conclusion"] in rule["premises"]]
        edges[i, i] = True
        edges[i, consumers] = True
        # Backward messages carry the public goal cue toward its prerequisites.
        count = sum(p in known for p in rule["premises"])
        rows.append([
            float(count == len(rule["premises"])), count / len(rule["premises"]),
            float(rule["conclusion"] in known), float(rule["conclusion"] == world["goal"]),
            len(rule["premises"]) / 4.0, len(consumers) / max(1, m),
            len(producers) / max(1, m), remaining / max(1, budget),
            float(i in failed), float(bool(history) and history[-1]["action"] == i),
        ])
    return torch.tensor(rows, dtype=torch.float32), ~edges


class LedgerRouter(nn.Module):
    def __init__(self, width=24, heads=4):
        super().__init__()
        self.width = width
        self.heads = heads
        self.input = nn.Linear(len(FEATURES), width)
        self.carry = nn.Linear(width, width, bias=False)
        self.carry_gate = nn.Parameter(torch.tensor(-2.0))
        self.recurrent = block(width, heads, 2)
        self.norm = nn.LayerNorm(width)
        self.policy = nn.Linear(width, 1)
        self.acceptance = nn.Linear(width, 1)

    def forward(self, features, attention_mask, previous=None, *, depth=4,
                reset_inner=False, shuffled_history=False):
        if depth < 1:
            raise ValueError("positive inner depth required")
        x = self.input(features).unsqueeze(0)
        if previous is not None:
            if previous.shape != x.shape:
                raise ValueError("history belongs to a different rule menu")
            if shuffled_history:
                previous = previous.roll(1, dims=1)
            x = x + torch.sigmoid(self.carry_gate) * self.carry(previous)
        state = x
        for _ in range(depth):
            state = self.recurrent(x if reset_inner else state, src_mask=attention_mask)
        state = self.norm(state)
        return {"logits": self.policy(state).squeeze(0).squeeze(-1),
                "acceptance_logits": self.acceptance(state).squeeze(0).squeeze(-1),
                "state": state}

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
