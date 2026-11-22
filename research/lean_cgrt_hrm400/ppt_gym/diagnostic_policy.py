"""A two-slot proof controller that writes only the newly observed diagnostic."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


# Output actions are commit zero, commit one, probe left, and probe right.
HISTORIES = ((), (0,), (1,), (0, 1), (1, 0))


def evidence(events):
    """Return two value channels and a write mask for each observation."""
    aliases, values = events.unbind(-1)
    valid = aliases >= 0
    write = (aliases.unsqueeze(-1) == torch.arange(2, device=events.device)) & valid.unsqueeze(-1)
    tokens = F.one_hot(values.clamp_min(0), 2).to(torch.float32)
    return tokens.unsqueeze(-2) * write.unsqueeze(-1), write


class DiagnosticPolicy(nn.Module):
    def __init__(self, warm, context_width, *, trm=False):
        super().__init__()
        self.trm = trm
        self.roles = nn.Parameter(warm.roles.detach().clone())
        self.cell = deepcopy(warm.trm if trm else warm.cell)
        self.context = nn.Linear(context_width - 3, warm.width)
        self.clock = nn.Linear(3, warm.width)
        self.signal = nn.Linear(2, warm.width, bias=False)
        self.action = nn.Linear(2 * warm.width, 4)
        if trm:
            self.register_buffer("answer_init", warm.answer_init.clone())
            self.register_buffer("latent_init", warm.latent_init.clone())

    def loop(self, slots):
        flat = slots.reshape(-1, 2, slots.shape[-1])
        if self.trm:
            answer, latent = self.answer_init, self.latent_init
            for _ in range(2):
                latent = self.cell(flat + answer + latent)
            return self.cell(answer + latent).reshape_as(slots)
        return self.cell(self.cell(flat)).reshape_as(slots)

    def memory(self, context, events, *, reset=False):
        base = self.context(context[..., :-3]).unsqueeze(-2) + self.roles
        signals, writes = evidence(events)
        if self.trm:
            # TRM receives the complete typed history in its two input tokens.
            return base + self.signal(signals.sum(-3))
        slots = base
        for index in range(events.shape[-2]):
            observed = signals.select(-3, index)
            mask = writes.select(-2, index).unsqueeze(-1)
            previous = base if reset else slots
            proposals = self.loop(previous + self.signal(observed))
            updated = torch.where(mask, proposals, previous)
            slots = torch.where(mask.any(dim=-2, keepdim=True), updated, slots)
        return slots

    def read(self, state, context):
        read = state + self.clock(context[..., -3:]).unsqueeze(-2)
        features = self.loop(read).flatten(-2)
        return self.action(features), features

    def forward(self, context, events, *, reset=False):
        state = self.memory(context, events, reset=reset)
        logits, _ = self.read(state, context)
        return logits
