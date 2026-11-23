"""A split tactic readout trained on the return of the whole Lean policy."""

import torch
from torch import nn
from torch.nn import functional as F

from .guard_belief import BeliefPlanner
from .guard_recovery import ACTIONS, ALLOWANCES, ENTER, PROBE, VERIFY, initial
from .guard_recovery_learning import GUARD_WORLDS, supports
from .live_quote_policy import FEATURES


DOSE = .075
TEMPERATURE = .045
CHANNELS = 4


class TacticCorrection(nn.Module):
    """Share the recurrent guard representation across four tactic readouts."""

    def __init__(self, actor=None):
        super().__init__()
        if actor is None:
            self.actor = nn.Sequential(nn.Flatten(1), nn.Linear(3 * FEATURES, CHANNELS))
            head = self.actor[-1]
        else:
            self.actor = actor
            head = self.actor.head = nn.Linear(actor.head.in_features, CHANNELS)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def forward(self, tokens):
        outputs = self.actor(tokens.flatten(0, 1)).reshape(len(tokens), 4, CHANNELS)
        return DOSE * outputs.tanh()


def tactic_ports(state, action):
    """Indices in the flattened false-guard by tactic-channel output."""
    if action in ENTER:
        first = 2 * ENTER.index(action)
        channel = int(state.remaining > 4)
        return first * CHANNELS + channel, (first + 1) * CHANNELS + channel
    if action in PROBE:
        index = PROBE.index(action) * CHANNELS + 2
        return index, index
    if action in VERIFY and state.branch >= 0:
        index = (2 * state.branch + VERIFY.index(action)) * CHANNELS + 3
        return index, index
    return CHANNELS * 4, CHANNELS * 4


class ReturnPlanner(BeliefPlanner):
    def __init__(self):
        super().__init__()
        self.starts = [self.index[initial(budget)] for budget in ALLOWANCES]
        sentinel = CHANNELS * 4
        self.ports = torch.tensor([[(sentinel, sentinel)] * len(ACTIONS)]
                                  + [[tactic_ports(state, action) for action in ACTIONS] for state in self.states])
        self.unresolved = torch.tensor([False] + [sum(supports(state, bits) for bits in GUARD_WORLDS) > 1
                                                  for state in self.states])

    def addends(self, heads):
        ports = F.pad(heads.flatten(1), (0, 1))
        return ports[:, self.ports].mean(dim=-1) * self.unresolved[:, None]

    def rollout(self, false_guard, scores=None):
        """Evaluate all transitions exactly for each source world in parallel.

        The policy is fixed by public scores before an outcome indexes an edge.
        With no policy scores, return full-information action targets for the
        matched regression control.
        """
        count = len(false_guard)
        values = self.reward.new_zeros(count, 1)
        targets = [self.reward.new_zeros(count, 1, len(ACTIONS))]
        offset = 1
        for _, children, legal in self.layers:
            span = slice(offset, offset + len(children))
            next_state = children[:, :, false_guard].permute(2, 0, 1)
            q = values.gather(1, next_state.flatten(1)).reshape_as(next_state) + self.reward
            if scores is None:
                q = q.masked_fill(~legal, -torch.inf)
                targets.append(q)
                layer = q.amax(dim=-1).clamp_min(0.)
            else:
                logits = F.pad(scores[:, span] / TEMPERATURE, (0, 1))
                prob = logits.softmax(dim=-1)[..., :-1]
                layer = (prob * q).sum(dim=-1)
            values = torch.cat((values, layer), dim=1)
            offset = span.stop
        return torch.cat(targets, dim=1) if scores is None else values[:, self.starts]
