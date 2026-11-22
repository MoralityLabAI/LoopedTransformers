"""Separate immediate acceptance, continuation value, and one-check retries."""

import torch
from torch import nn

from .mathcan_route_policy import RoutePolicy


class ProgressRoute(RoutePolicy):
    def __init__(self, warm, *, trm=False):
        super().__init__(warm, trm=trm, deadline=True)
        self.accept = nn.Linear(2 * self.width, 1)

    def factors(self, tokens):
        features = self.features(tokens)
        accepted = self.accept(torch.cat((features, tokens[..., 0, :]), dim=-1)).squeeze(-1)
        conditional = self.first_hit(features).softmax(dim=-1).cumsum(dim=-1)[..., :-1]
        return accepted, conditional


def progress_curve(acceptance, conditional, *, retry=True):
    direct = acceptance.unsqueeze(-1) * conditional
    if not retry or direct.shape[-2] <= 1:
        return direct
    # The best remaining option is available after a rejection, but the
    # rejected tactic itself cannot be tried again on that prefix.
    values, indices = direct.topk(2, dim=-2)
    actions = torch.arange(direct.shape[-2], device=direct.device)[..., None]
    best_other = torch.where(actions == indices[..., :1, :], values[..., 1:, :], values[..., :1, :])
    fallback = torch.cat((torch.zeros_like(best_other[..., :1]), best_other[..., :-1]), dim=-1)
    return direct + (1. - acceptance).unsqueeze(-1) * fallback
