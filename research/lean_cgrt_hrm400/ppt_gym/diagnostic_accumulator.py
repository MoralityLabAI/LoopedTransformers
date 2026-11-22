"""Fixed-size diagnostic memory with shared affine writes and a separate read."""

from copy import deepcopy

import torch
from torch import nn


MAX_OBSERVATIONS = 5
OPERATORS = 4


class DiagnosticAccumulator(nn.Module):
    """Keep the goal token fixed and update only the diagnostic accumulator."""

    def __init__(self, warm, context_width, *, kind="affine"):
        super().__init__()
        self.kind = kind
        self.width = warm.width
        self.roles = nn.Parameter(warm.roles.detach().clone())
        self.cell = deepcopy(warm.trm if kind == "trm" else warm.cell)
        self.context = nn.Linear(context_width, self.width)
        self.clock = nn.Linear(2, self.width)
        if kind != "trm":
            self.seed = nn.Embedding(OPERATORS, self.width)
        if kind != "affine":
            self.signal = nn.Embedding(2, self.width)
        self.commit = nn.Linear(self.width * 2, 2)
        self.acquire = nn.Linear(self.width * 2, 1)
        if kind == "affine":
            self.scale = nn.Embedding(OPERATORS * 2, self.width)
            self.shift = nn.Embedding(OPERATORS * 2, self.width)
            nn.init.normal_(self.scale.weight, std=1.)
            nn.init.normal_(self.shift.weight, std=.1)
        if kind == "trm":
            self.register_buffer("answer_init", warm.answer_init.clone())
            self.register_buffer("latent_init", warm.latent_init.clone())

    def encode(self, context, operators, values):
        """Return all prefix states using only observations in each prefix."""
        first = self.context(context) + self.roles[0]
        observed = values.clamp_min(0)
        width = values.shape[-1]
        if self.kind == "trm":
            prefix = torch.arange(width + 1, device=values.device)[:, None]
            time = torch.arange(width, device=values.device)
            visible = (time < prefix) & (values[:, None] >= 0)
            diagnostics = (self.signal(observed)[:, None] + self.roles[1]) * visible.unsqueeze(-1)
            goal = first[:, None, None].expand(-1, width + 1, 1, -1)
            return torch.cat((goal, diagnostics), dim=-2)
        current = self.seed(operators) + self.roles[1]
        states = [current]
        for step in range(width):
            bit = observed[:, step]
            if self.kind == "affine":
                transition = operators * 2 + bit
                update = self.scale(transition).tanh() * current + self.shift(transition)
            else:
                tokens = torch.stack((first, current + self.signal(bit)), dim=-2)
                update = self.cell(self.cell(tokens))[:, 1]
            current = torch.where((values[:, step] >= 0).unsqueeze(-1), update, current)
            states.append(current)
        held = torch.stack(states, dim=1)
        return torch.stack((first[:, None].expand_as(held), held), dim=-2)

    def reads(self, tokens, clock):
        flat = tokens.flatten(0, -3)
        if self.kind == "trm":
            answer, latent = self.answer_init, self.latent_init
            for _ in range(2):
                latent = self.cell(flat + answer + latent)
            output = self.cell(answer + latent)
            # The first token reads the complete observed prefix.
            stats = torch.stack((output[:, 0], output[:, 1:].mean(1)), dim=-2)
        else:
            stats = self.cell(self.cell(flat))
        context = stats.flatten(-2).reshape(*tokens.shape[:-2], 2 * self.width)
        clock_signal = self.clock(clock)
        gate = context + torch.cat((clock_signal, clock_signal), dim=-1)
        return context, gate

    def forward(self, context, operators, values, clock):
        states = self.encode(context, operators, values)
        commit, acquire = self.reads(states, clock)
        return torch.cat((self.commit(commit), self.acquire(acquire)), dim=-1)
