"""Recurrent policies that keep memory between paid Lean actions."""

import torch
from torch import nn

from lcgrt.tactic_lm import PostBlock

from .live_quote_policy import FEATURES, WIDTH


class PersistentRecoverySlots(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(FEATURES, WIDTH)
        self.roles = nn.Parameter(torch.randn(3, WIDTH) * .02)
        self.loop = PostBlock(WIDTH, 4, 2)
        self.message = nn.Linear(WIDTH, WIDTH, bias=False)
        self.gate = nn.Linear(2 * WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def visit(self, state, cue):
        swapped = [1, 0, 2] if state.shape[1] == 3 else [1, 0]
        message = self.message(state[:, swapped])
        gate = self.gate(torch.cat((cue, state), dim=-1)).sigmoid()
        return self.loop(state + cue + gate * message)

    def initialize(self, public):
        cue = self.embed(public[:, :2]) + self.roles[:2]
        return self.visit(cue, cue)

    def observe(self, memory, event):
        paid = self.embed(event) + self.roles[0]
        cue = torch.stack((paid, self.roles[1].expand_as(paid)), dim=1)
        return self.visit(memory, cue)

    def score(self, memory, choices):
        batch, actions = choices.shape[:2]
        cue = self.embed(choices.flatten(0, 1)) + self.roles
        retained = memory.repeat_interleave(actions, dim=0)
        state = torch.cat((retained, cue[:, 2:]), dim=1)
        for _ in range(2):
            state = self.visit(state, cue)
        return self.head(state[:, 2]).reshape(batch, actions)


class PersistentRecoveryTRM(nn.Module):
    """Answer and latent states consume the same paid events and legal menus."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(FEATURES, WIDTH)
        self.roles = nn.Parameter(torch.randn(3, WIDTH) * .02)
        self.loop = nn.Sequential(PostBlock(WIDTH, 4, 2), PostBlock(WIDTH, 4, 2))
        self.head = nn.Linear(WIDTH, 1)
        self.register_buffer("answer_init", torch.randn(WIDTH) * .02)
        self.register_buffer("latent_init", torch.randn(WIDTH) * .02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def advance(self, memory, cue):
        answer, latent = memory.unbind(dim=1)
        latent = self.loop((cue + answer[:, None] + latent[:, None]))[:, -1]
        answer = self.loop((cue + answer[:, None] + latent[:, None]))[:, -1]
        return torch.stack((answer, latent), dim=1)

    def initialize(self, public):
        cue = self.embed(public[:, :2]) + self.roles[:2]
        initial = torch.stack((self.answer_init, self.latent_init))
        return self.advance(initial.expand(len(cue), -1, -1), cue)

    def observe(self, memory, event):
        cue = self.embed(event)[:, None] + self.roles[:1]
        return self.advance(memory, cue)

    def score(self, memory, choices):
        batch, actions = choices.shape[:2]
        cue = self.embed(choices.flatten(0, 1)) + self.roles
        states = self.advance(memory.repeat_interleave(actions, dim=0), cue)
        return self.head(states[:, 0]).reshape(batch, actions)
