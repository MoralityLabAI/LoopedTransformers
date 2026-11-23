"""A bounded learned correction to a frozen Lean recovery planner."""

from functools import lru_cache
from math import fsum, sqrt

import torch
from torch import nn

from .failed_prefix import CHARGE, WORLDS, planned_values
from .failed_prefix_learning import COUNT


BOUND = .15


class RecoveryCorrection(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        head = actor.head if hasattr(actor, "head") else actor[-1]
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def forward(self, tokens):
        return BOUND * self.actor(tokens).tanh()


def augment(tokens, indices, routes, weights):
    """Append conditional belief and exact route values without reading native outcomes."""
    values = planned_values(routes, weights)
    rows, scores = tokens.clone(), []
    for row, code in enumerate(indices):
        state, arm = divmod(code, 3)
        remaining, failed = divmod(state, 8)
        q = values(remaining, failed)
        prior = [weight if all(not (failed & (1 << index)) or not bit for index, bit in enumerate(world)) else 0.
                 for world, weight in zip(WORLDS, weights, strict=True)]
        total = fsum(prior)
        chances = [fsum(weight for weight, world in zip(prior, WORLDS, strict=True) if world[index]) / total
                   for index in range(3)]
        context = [*(q.get(index, 0.) for index in range(3)), *chances, *(float(index in q) for index in range(3)),
                   q[arm] - max(0., *q.values())]
        rows[row, 2, 17:27] = rows.new_tensor(context)
        rows[row, 0, 25] = chances[0]
        rows[row, 1, 25] = chances[arm]
        scores.append(q[arm])
    values.cache_clear()
    return rows, scores


def bounded_scores(model, tokens, codes, base):
    correction = model(tokens.flatten(0, 1)).reshape(codes.shape)
    dense = correction.new_zeros(len(tokens), COUNT * 3 + 1).scatter(1, codes, correction)
    return base + dense[:, :-1].reshape_as(base)


def possible_choices(values):
    best = max(values.values(), default=0.)
    moves = [arm for arm, value in values.items() if value + BOUND > 0. and value + BOUND >= best - BOUND]
    return (*moves, None) if best <= BOUND else tuple(moves)


def bounded_oracle(routes, weights, truth):
    """Perfect-information upper bound restricted to the score correction's action envelope."""
    planner = planned_values(routes, weights)

    @lru_cache(maxsize=None)
    def choose(remaining, failed):
        def gain(arm):
            if arm is None:
                return 0.
            route = routes[arm]
            if truth[arm]:
                return 1 - CHARGE * route["pass_cost"]
            cost = route["fail_cost"]
            return choose(remaining - cost, failed | (1 << arm))[0] - CHARGE * cost

        envelope = possible_choices(planner(remaining, failed))
        best = max(envelope, key=gain)
        return gain(best), best

    return lambda remaining, failed, _menu: choose(remaining, failed)[1]


def redistribute(residual, selected, frozen, sham_sign=None):
    """Swap the two chosen addends or put the same norm on the orthogonal contrast."""
    changes = dict(residual)
    difference = changes[frozen] - changes[selected]
    if sham_sign is None:
        changes[selected], changes[frozen] = changes[frozen], changes[selected]
    else:
        other, = (arm for arm in changes if arm not in (selected, frozen))
        shift = sham_sign * difference / sqrt(3.)
        changes[selected] += shift
        changes[frozen] += shift
        changes[other] -= 2 * shift
    return changes, sqrt(2.) * abs(difference)
