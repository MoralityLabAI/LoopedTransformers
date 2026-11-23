"""Bounded recurrent corrections to an intact supervised guard planner."""

from functools import lru_cache

import torch
from torch import nn

from .guard_recovery import CHARGE, transition
from .guard_recovery_learning import action_features


RADIUS = .0375


def packed_action(descriptors, probability, warm, state, action, base, maximum):
    row = action_features(descriptors, probability, warm, state, action)
    row[2][-2:] = [base, maximum - base]
    return row


class BoundedCorrection(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        nn.init.zeros_(self.actor.head.weight)
        nn.init.zeros_(self.actor.head.bias)

    def residual(self, inputs):
        return RADIUS * self.actor(inputs).tanh()

    def forward(self, inputs, base):
        return base + self.residual(inputs)


def admissible_choices(values):
    """Overestimate the selectable actions by permitting ties at the radius."""
    incumbent = max(values.values(), default=0.)
    floor = max(0., incumbent - RADIUS)
    moves = [move for move, value in values.items() if value + RADIUS >= floor]
    if incumbent <= RADIUS:
        moves.append(None)
    return moves


def hindsight_policy(planner_values, truth):
    """Finite upper bound for any bounded correction on the same planner."""
    @lru_cache(maxsize=None)
    def value(state):
        return max((outcome(state, move) for move in admissible_choices(planner_values(state))), default=0.)

    def outcome(state, move):
        if move is None:
            return 0.
        following, bit = transition(state, move, truth)
        return float(move == "close" and bit is True) - CHARGE + (value(following) if following is not None else 0.)

    def choose(state):
        return max(admissible_choices(planner_values(state)), key=lambda move: outcome(state, move))

    return choose
