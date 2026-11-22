"""A tied two-token proof cell with explicit carry between Lean tactics."""

import torch
from torch import nn

from .core import ByteEncoder, block
from .data import canonical
from .rewrite import rewrite_hypothesis
from .two_slot_fixture import CHECKS


class TwoSlotPolicy(nn.Module):
    def __init__(self, width=32, passes=2):
        super().__init__()
        self.width = width
        self.passes = passes
        self.text = ByteEncoder(width, 384)
        self.roles = nn.Parameter(torch.randn(2, width) * .02)
        self.budget = nn.Linear(1, width)
        self.cell = block(width, 4, 2)
        self.write = nn.Linear(2 * width, width)
        self.actor = nn.Linear(width, 1)

    def macs(self, actions):
        d = self.width
        text = (4 + actions) * self.text.max_bytes * 3 * d * d
        return text + actions * self.passes * (16 * d * d + 8 * d) + 8 * d * d + (actions + 1) * d

    def encode(self, goals):
        if not 1 <= len(goals) <= 2:
            raise ValueError("two-slot policy requires one or two open goals")
        texts = [canonical(goal["target"]) + " " + " ".join(
            f"{hyp['name']}:{hyp['type'].get('raw', canonical(hyp['type']))}"
            for hyp in goal["hypotheses"]) for goal in goals]
        if any(len(item.encode("utf-8")) > self.text.max_bytes for item in texts):
            raise ValueError("two-slot goal exceeds the registered text budget")
        return self.text(texts) + self.roles[:len(goals)]

    def forward(self, goals, menu, checks_left, carry=None):
        observed = self.encode(goals)
        hypotheses = {hyp["name"]: hyp["type"] for hyp in goals[0]["hypotheses"]}
        actions = []
        for action in menu:
            name = (rewrite_hypothesis(action.removeprefix("all_goals "))
                    or action.removeprefix("exact ").removeprefix("apply "))
            actions.append(f"{action} {canonical(hypotheses.get(name))}")
        cue = self.text(actions) + self.budget(self.roles.new_tensor([checks_left / CHECKS]))
        if carry is not None:
            observed = observed + carry[:len(goals)]
        slots = observed.unsqueeze(0) + cue.unsqueeze(1)
        for _ in range(self.passes):
            slots = self.cell(slots)
        return self.actor(slots[:, 0]).squeeze(-1), slots

    def advance(self, predicted, goals):
        observed = self.encode(goals)
        if len(predicted) == 2 and len(goals) == 1:
            predicted = predicted[1:]
        elif len(predicted) == 1 and len(goals) == 2:
            predicted = torch.cat((predicted, torch.zeros_like(predicted)))
        return torch.lerp(predicted, observed, self.write(torch.cat((predicted, observed), dim=-1)).sigmoid())
