"""A tied two-token proof cell with explicit carry between Lean tactics."""

import torch
from torch import nn

from lcgrt.tactic_lm import PostBlock

from .core import ByteEncoder, block
from .data import canonical
from .rewrite import rewrite_hypothesis
from .slot_bindings import BINDING_CHANNELS, action_bindings, binding_features
from .two_slot_fixture import CHECKS


class TwoSlotInputs(nn.Module):
    def __init__(self, width, passes, *, bindings):
        super().__init__()
        self.width = width
        self.passes = passes
        self.text = ByteEncoder(width, 384)
        self.roles = nn.Parameter(torch.randn(2, width) * .02)
        self.budget = nn.Linear(1, width)
        self.binding = nn.Linear(BINDING_CHANNELS, width, bias=False) if bindings else None

    def encode(self, goals):
        if not 1 <= len(goals) <= 2:
            raise ValueError("two-slot policy requires one or two open goals")
        texts = [canonical(goal["target"]) + " " + " ".join(
            f"{hyp['name']}:{hyp['type'].get('raw', canonical(hyp['type']))}"
            for hyp in goal["hypotheses"]) for goal in goals]
        if any(len(item.encode("utf-8")) > self.text.max_bytes for item in texts):
            raise ValueError("two-slot goal exceeds the registered text budget")
        return self.text(texts) + self.roles[:len(goals)]

    def cues(self, goals, menu, checks_left, carry=None):
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
        if self.binding is not None:
            features = slots.new_tensor([action_bindings(goals, action) for action in menu])
            slots = slots + self.binding(features)
        return slots


class TwoSlotPolicy(TwoSlotInputs):
    def __init__(self, width=32, passes=2, *, bindings=False):
        super().__init__(width, passes, bindings=bindings)
        self.cell = block(width, 4, 2)
        self.write = nn.Linear(2 * width, width)
        self.actor = nn.Linear(width, 1)

    def macs(self, actions):
        d = self.width
        text = (4 + actions) * self.text.max_bytes * 3 * d * d
        binding = 2 * (actions + 1) * BINDING_CHANNELS * d if self.binding is not None else 0
        return text + actions * self.passes * (16 * d * d + 8 * d) + 8 * d * d + (actions + 1) * d + binding

    def forward(self, goals, menu, checks_left, carry=None):
        slots = self.cues(goals, menu, checks_left, carry)
        for _ in range(self.passes):
            slots = self.cell(slots)
        return self.actor(slots[:, 0]).squeeze(-1), slots

    def advance(self, predicted, goals):
        observed = self.encode(goals)
        if self.binding is not None:
            observed = observed + self.binding(observed.new_tensor([binding_features(goal) for goal in goals]))
        if len(predicted) == 2 and len(goals) == 1:
            predicted = predicted[1:]
        elif len(predicted) == 1 and len(goals) == 2:
            predicted = torch.cat((predicted, torch.zeros_like(predicted)))
        return torch.lerp(predicted, observed, self.write(torch.cat((predicted, observed), dim=-1)).sigmoid())


class TwoSlotTRM(TwoSlotInputs):
    """TRM's tied answer/latent updates on the same two public goal tokens."""

    def __init__(self, width=28, passes=1, *, bindings=False):
        super().__init__(width, passes, bindings=bindings)
        self.trm = nn.Sequential(PostBlock(width, 4, 2), PostBlock(width, 4, 2))
        self.actor = nn.Linear(width, 1)
        self.register_buffer("answer_init", torch.randn(width) * .02)
        self.register_buffer("latent_init", torch.randn(width) * .02)

    def macs(self, actions):
        d = self.width
        text = (2 + actions) * self.text.max_bytes * 3 * d * d
        binding = 2 * actions * BINDING_CHANNELS * d if self.binding is not None else 0
        return text + 6 * actions * self.passes * (16 * d * d + 8 * d) + (actions + 1) * d + binding

    def forward(self, goals, menu, checks_left, carry=None):
        cue = self.cues(goals, menu, checks_left)
        answer, latent = self.answer_init.expand_as(cue), self.latent_init.expand_as(cue)
        for _ in range(self.passes):
            for _ in range(2):
                latent = self.trm(cue + answer + latent)
            answer = self.trm(answer + latent)
        return self.actor(answer[:, 0]).squeeze(-1), answer

    def advance(self, predicted, goals):
        return None
