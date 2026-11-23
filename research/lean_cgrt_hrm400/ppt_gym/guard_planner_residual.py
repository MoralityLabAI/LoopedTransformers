"""Bounded recurrent corrections to an intact supervised guard planner."""

import torch
from torch import nn

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
