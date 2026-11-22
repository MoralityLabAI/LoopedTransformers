"""Estimate proof success and future diagnostic work separately."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AcquisitionBatch:
    costs: torch.Tensor
    remaining: torch.Tensor
    legal: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def from_regimes(cls, horizons, regimes, width):
        fees, withheld = zip(*regimes, strict=True)
        costs = torch.tensor(fees)[:, None, None, None]
        cap = (horizons[None, :, None] - torch.tensor(withheld)[:, None, None]).clamp_min(0)
        time = torch.arange(width + 1)[None, None]
        remaining = (cap - time).clamp_min(0)
        valid = time <= cap
        legal = torch.stack((torch.ones_like(valid), torch.ones_like(valid), remaining > 0), dim=-1)
        return cls(costs, remaining, legal, valid)

    def utility(self, factors):
        return factors[..., 0] - self.costs * factors[..., 1]


def action_factors(batch, native, groups, policy=None):
    """Back up terminal outcomes under a continuation policy or the optimal policy."""
    fees, worlds, times, actions = batch.legal.shape
    q = native.new_zeros((fees, worlds, times, actions, 2))
    q[..., :2, 0] = native[None, :, None, :]
    continuation = native.new_zeros((fees, worlds, 2))
    for time in range(times - 1, -1, -1):
        q[:, :, time, 2] = continuation + native.new_tensor([0., 1.])
        index, count = groups[time]
        grouped = q.new_zeros((fees, len(count), actions, 2)).index_add(1, index, q[:, :, time])
        mean = (grouped / count[None, :, :, None])[:, index]
        if policy is None:
            utilities = mean[..., 0] - batch.costs.squeeze(-1) * mean[..., 1]
            choice = utilities.masked_fill(~batch.legal[:, :, time], -torch.inf).argmax(-1)
            continuation = mean.gather(2, choice[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        else:
            continuation = (policy[:, :, time, :, None] * q[:, :, time]).sum(-2)
        q[:, :, time] = mean
    return q


def acquisition_features(batch, inputs, forecast):
    width = inputs.values.shape[-1]
    context = inputs.context[None, :, None].expand(len(batch.costs), -1, width + 1, -1)
    remaining = batch.remaining.unsqueeze(-1).float() / width
    costs = batch.costs.expand(-1, context.shape[1], context.shape[2], -1)
    return torch.cat((context, remaining, costs, forecast[..., 0], forecast[..., 1] / width), dim=-1)


class AcquisitionHead(nn.Module):
    def __init__(self, inputs, *, scalar=False):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(inputs, 32), nn.GELU(), nn.Linear(32, 3 if scalar else 4))
        nn.init.zeros_(self.hidden[-1].weight)
        nn.init.zeros_(self.hidden[-1].bias)
        self.scalar = scalar

    def forward(self, batch, features, forecast):
        update = self.hidden(features)
        if self.scalar:
            return batch.utility(forecast) + update, None
        anchor = torch.logit(forecast[..., 0].clamp(.05, .95))
        success = (anchor + update[..., :3]).sigmoid()
        extra_capacity = (batch.remaining.float() - 1).clamp_min(0)
        share = ((forecast[..., 2, 1] - 1) / extra_capacity.clamp_min(1)).clamp(.05, .95)
        work = 1 + (torch.logit(share) + update[..., 3]).sigmoid() * extra_capacity
        work = torch.cat((torch.zeros_like(success[..., :2]), work.unsqueeze(-1)), dim=-1)
        factors = torch.stack((success, work), dim=-1)
        return batch.utility(factors), factors
