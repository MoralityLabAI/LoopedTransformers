"""Tied checkpointed slots and a full-history TRM for the same Lean evidence."""

import torch
from torch import nn

from lcgrt.tactic_lm import PostBlock

from .scope_recovery import N_ACTIONS


WIDTH = 24
EVENT = 4
PUBLIC = 6


def inputs(state, edited):
    context = [*(state.phase == i for i in range(3)), edited == 0, edited == 1, state.remaining / 3]
    evidence = []
    for time, value in enumerate(state.values()):
        address = time if time < 2 else edited
        evidence.append([value >= 0, 2 * value - 1 if value >= 0 else 0., address == 0, address == 1])
    return context, evidence


class ScopeSlots(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(EVENT, WIDTH)
        self.context = nn.Linear(PUBLIC, WIDTH)
        self.roles = nn.Parameter(torch.randn(4, WIDTH) * .02)
        self.loop = PostBlock(WIDTH, 4, 2)
        self.link = nn.Linear(WIDTH, WIDTH, bias=False)
        self.gate = nn.Linear(2 * WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, N_ACTIONS)

    def written(self, events):
        shape = events.shape[:-1]
        cue = self.encode(events.reshape(-1, EVENT)).unsqueeze(1)
        value = self.loop(self.loop(cue) + cue).reshape(*shape, WIDTH)
        return torch.where(events[..., :1].bool(), value, torch.zeros_like(value))

    def memory(self, context, history):
        checkpoint = self.written(history[:, :2])
        pending = self.written(history[:, 2])
        resolved = context[:, 2].bool()
        keep = resolved & (history[:, 3, 1] > 0)
        address = context[:, 3:5].bool()
        active = torch.where((keep[:, None] & address)[..., None], pending[:, None], checkpoint)
        visible_pending = torch.where(resolved[:, None], torch.zeros_like(pending), pending)
        return active, visible_pending

    def read(self, context, active, pending):
        left, right = active.unbind(dim=1)
        cue = self.context(context) + self.link(left * right)
        start = torch.cat((active, pending[:, None], cue[:, None]), dim=1) + self.roles
        state = start
        for _ in range(2):
            message = self.link(state[:, [1, 0, 2, 3]])
            gate = self.gate(torch.cat((state, start), dim=-1)).sigmoid()
            state = self.loop(state + start + message * gate)
        return state[:, -1]

    def forward(self, context, history):
        return self.head(self.read(context, *self.memory(context, history)))


class ScopeTRM(nn.Module):
    """Tied answer and latent updates over the identical four historical events."""

    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(EVENT, WIDTH)
        self.context = nn.Linear(PUBLIC, WIDTH)
        self.roles = nn.Parameter(torch.randn(5, WIDTH) * .02)
        self.loop = nn.Sequential(PostBlock(WIDTH, 4, 2), PostBlock(WIDTH, 4, 2))
        self.head = nn.Linear(WIDTH, N_ACTIONS)
        self.register_buffer("answer_init", torch.randn(WIDTH) * .02)
        self.register_buffer("latent_init", torch.randn(WIDTH) * .02)

    def forward(self, context, history):
        cue = torch.cat((self.encode(history), self.context(context)[:, None]), dim=1) + self.roles
        answer, latent = self.answer_init.expand_as(cue), self.latent_init.expand_as(cue)
        for _ in range(2):
            latent = self.loop(cue + answer + latent)
        return self.head(self.loop(cue + answer + latent)[:, -1])


def calibrated_features(context, history, *, aligned=False):
    signal = torch.cat((context, history[:, :, :2].flatten(1)), dim=1)
    pairwise = torch.triu_indices(signal.shape[-1], signal.shape[-1])
    products = signal[:, pairwise[0]] * signal[:, pairwise[1]]
    if aligned:
        keep = context[:, 2].bool() & (history[:, 3, 1] > 0)
        active = torch.where((keep[:, None] & context[:, 3:5].bool())[:, :, None],
                             history[:, 2:3, :2], history[:, :2, :2])
        products = torch.cat((products, active.flatten(1), active[:, 0] * active[:, 1],
                              (active[:, 0, 0] * active[:, 1, 0])[:, None]), dim=1)
    return torch.cat((torch.ones_like(signal[:, :1]), signal, products), dim=1)
