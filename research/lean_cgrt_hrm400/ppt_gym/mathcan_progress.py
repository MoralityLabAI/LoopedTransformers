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


def retry_curve(direct):
    # The best remaining option is available after a rejection, but the
    # rejected tactic itself cannot be tried again on that prefix.
    if direct.shape[-2] <= 1:
        return torch.zeros_like(direct)
    values, indices = direct.topk(2, dim=-2)
    actions = torch.arange(direct.shape[-2], device=direct.device)[..., None]
    best_other = torch.where(actions == indices[..., :1, :], values[..., 1:, :], values[..., :1, :])
    return torch.cat((torch.zeros_like(best_other[..., :1]), best_other[..., :-1]), dim=-1)


def progress_curve(acceptance, conditional, *, retry=True):
    direct = acceptance.unsqueeze(-1) * conditional
    return direct + (1. - acceptance).unsqueeze(-1) * retry_curve(direct) if retry else direct


def split_completion_curve(immediate, delayed):
    """Keep empirical first-check mass and Bellman timing conditional on surviving it."""
    first = immediate[..., :1]
    old_first = delayed[..., :1]
    survival = 1. - old_first
    tail = (delayed[..., 1:] - old_first) / survival.clamp_min(torch.finfo(delayed.dtype).tiny)
    return torch.cat((first, first + (1. - first) * tail), dim=-1)
