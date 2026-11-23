"""Compare affordable recovery routes around a shared retained-witness slot."""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from lcgrt.tactic_lm import PostBlock

from .failed_prefix import CHARGE
from .failed_prefix_learning import COUNT
from .live_quote_policy import FEATURES, WIDTH


BOUND = .10
TEMPERATURE = .08


class MenuActor(nn.Module):
    def __init__(self, trm=False):
        super().__init__()
        self.saved = nn.Linear(FEATURES, WIDTH)
        self.route = nn.Linear(2 * FEATURES, WIDTH)
        if trm:
            self.loop = nn.ModuleList(PostBlock(WIDTH, 4, 2) for _ in range(2))
            self.register_buffer("answer_init", torch.randn(WIDTH) * .02)
            self.register_buffer("latent_init", torch.randn(WIDTH) * .02)
        else:
            self.loop = PostBlock(WIDTH, 4, 2)
            self.message = nn.Linear(WIDTH, WIDTH, bias=False)
            self.gate = nn.Linear(2 * WIDTH, WIDTH)
        self.trm = trm
        self.head = nn.Linear(WIDTH, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens, legal):
        count = legal.sum(-1).clamp_min(1).unsqueeze(-1)
        saved = (tokens[:, :, 0] * legal.unsqueeze(-1)).sum(1) / count
        candidate = self.route(tokens[:, :, 1:].flatten(2))
        cue = torch.cat((self.saved(saved).unsqueeze(1), candidate), dim=1)
        mask = F.pad(legal, (1, 0), value=True)[:, None, None, :]
        if self.trm:
            def block(value):
                for layer in self.loop:
                    value = layer(value, mask)
                return value

            answer = self.answer_init.expand_as(cue)
            latent = self.latent_init.expand_as(cue)
            for _ in range(2):
                latent = block(cue + answer + latent)
            state = block(answer + latent)
        else:
            state = cue
            for _ in range(3):
                updated = self.loop(cue + state, mask)
                pool = (updated[:, 1:] * legal.unsqueeze(-1)).sum(1) / count
                opposite = torch.cat((pool.unsqueeze(1), updated[:, :1].expand(-1, 3, -1)), dim=1)
                state = updated + self.message(opposite) * self.gate(torch.cat((updated, opposite), dim=-1)).sigmoid()
        return BOUND * self.head(state[:, 1:]).squeeze(-1).tanh()


class AffineMenu(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(6 * FEATURES, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens, legal):
        flat = tokens.flatten(2)
        pool = (flat * legal.unsqueeze(-1)).sum(1) / legal.sum(-1).clamp_min(1).unsqueeze(-1)
        return BOUND * self.head(torch.cat((flat, pool.unsqueeze(1).expand_as(flat)), dim=-1)).squeeze(-1).tanh()


def group_candidates(tokens, codes, base):
    states = sorted({code // 3 for code in codes})
    positions = {state: i for i, state in enumerate(states)}
    menus = tokens.new_zeros(len(states), 3, 3, FEATURES)
    values = tokens.new_full((len(states), 3), -torch.inf)
    for row, (code, value) in enumerate(zip(codes, base, strict=True)):
        state, arm = divmod(code, 3)
        index = positions[state]
        menus[index, arm] = tokens[row]
        menus[index, arm, 2, 27] = value
        values[index, arm] = value
    return menus, states, values


def corrections(model, tokens, base):
    legal = base.isfinite()
    delta = model(tokens.flatten(0, 1), legal.flatten(0, 1)).reshape_as(base)
    return delta.masked_fill(~legal, 0.)


def policy(base, delta, states, acting):
    """Use the frozen stop gate and normalize only over affordable alternatives."""
    logits = ((base + delta) / TEMPERATURE).masked_fill(~acting.unsqueeze(-1), 0.)
    probability = logits.softmax(-1) * acting.unsqueeze(-1)
    dense = probability.new_zeros(len(base), COUNT + 1, 3)
    return dense.scatter(1, states.unsqueeze(-1).expand_as(probability), probability)[:, :-1]


def rank_oracle(routes, truth, anchor):
    @lru_cache(maxsize=None)
    def best(remaining, failed):
        values = anchor.get(remaining * 8 + failed, {})
        top = max(values.values(), default=0.)
        if top <= 0.:
            return 0., None

        def gain(arm):
            route = routes[arm]
            if truth[arm]:
                return 1 - CHARGE * route["pass_cost"]
            cost = route["fail_cost"]
            return best(remaining - cost, failed | (1 << arm))[0] - CHARGE * cost

        selected = max((arm for arm, value in values.items() if value + BOUND >= top - BOUND), key=gain)
        return gain(selected), selected

    return lambda remaining, failed, _menu: best(remaining, failed)[1]
