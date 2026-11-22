"""A compact feedback state and a full-history control for MathCan tactics."""

from copy import deepcopy

import torch
from torch import nn

from .mathcan_sparse import CHECKS, CHECK_COST


class RoutePolicy(nn.Module):
    def __init__(self, warm, *, trm=False):
        super().__init__()
        self.trm = trm
        self.width = warm.width
        self.roles = nn.Parameter(warm.roles.detach().clone())
        self.cell = deepcopy(warm.trm if trm else warm.cell)
        self.input = nn.Linear(32, warm.width)
        if warm.width == 32:
            nn.init.eye_(self.input.weight)
            nn.init.zeros_(self.input.bias)
        self.clock = nn.Linear(1, warm.width)
        self.success = deepcopy(warm.actor)
        self.work = nn.Linear(warm.width, 1)
        if trm:
            self.history = nn.Linear(warm.width + 2, warm.width)
            self.register_buffer("answer_init", warm.answer_init.clone())
            self.register_buffer("latent_init", warm.latent_init.clone())
        else:
            self.write = nn.Linear(warm.width + 1, 2 * warm.width)

    def encode(self, goals, actions, past, feedback, remaining):
        proof = self.input(goals) + self.roles
        cue = self.input(actions) + self.clock(remaining[:, None] / CHECKS).unsqueeze(1)
        tokens = proof[:, None] + cue[:, :, None]
        trace = self.input(past)
        if self.trm:
            age = torch.arange(CHECKS, device=past.device).expand_as(feedback) / CHECKS
            observed = self.history(torch.cat((trace, feedback[..., None], age[..., None]), dim=-1))
            observed = observed * (feedback != 0).unsqueeze(-1)
            return torch.cat((tokens, observed[:, None].expand(-1, actions.shape[1], -1, -1)), dim=-2)
        memory = torch.zeros_like(proof[:, 0])
        weights = self.write(torch.cat((trace, feedback[..., None]), dim=-1))
        scales, shifts = weights.chunk(2, dim=-1)
        for index in range(CHECKS):
            gate = (feedback[:, index] != 0).unsqueeze(-1)
            updated = scales[:, index].sigmoid() * memory + shifts[:, index].tanh()
            memory = torch.where(gate, updated, memory)
        return torch.stack((tokens[:, :, 0], tokens[:, :, 1] + memory.unsqueeze(1)), dim=-2)

    def read(self, tokens, remaining):
        flat = tokens.flatten(0, 1)
        if self.trm:
            answer, latent = self.answer_init, self.latent_init
            for _ in range(2):
                latent = self.cell(flat + answer + latent)
            hidden = self.cell(answer + latent)
        else:
            hidden = self.cell(self.cell(flat))
        features = hidden[:, 0].reshape(*tokens.shape[:2], self.width)
        probability = self.success(features).squeeze(-1).sigmoid()
        checks = 1. + self.work(features).squeeze(-1).sigmoid() * (remaining[:, None] - 1.)
        return torch.stack((probability, checks), dim=-1), features

    def forward(self, goals, actions, past, feedback, remaining):
        return self.read(self.encode(goals, actions, past, feedback, remaining), remaining)[0]


def utility(factors):
    return factors[..., 0] - CHECK_COST * factors[..., 1]
