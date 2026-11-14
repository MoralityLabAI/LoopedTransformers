"""AF-04: graph-typed native recurrent proof-state model and code interchange.

The original TypedLoop implementation is source-hash verified by AF-02 and
imported without modification. Training may update its parameters. The
bottleneck is the actual state carried between every native cell application.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from experiments.activation_funnel.model import LearnedCodec
from experiments.activation_funnel_af02.model import native_cell_class
from .gym import GraphBatch, ATOMS, RULES


class GraphCodec(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        if width not in (1, 2):
            raise ValueError('registered width is one or two floats per token')
        self.width = width
        self.atom = LearnedCodec(width=16, code_width=width)
        self.rule = LearnedCodec(width=16, code_width=width)

    def encode(self, a, r):
        return self.atom.encode(a), self.rule.encode(r)

    def decode(self, az, rz):
        return self.atom.decode(az), self.rule.decode(rz)


@dataclass
class State:
    a: torch.Tensor
    r: torch.Tensor
    az: torch.Tensor
    rz: torch.Tensor


@dataclass
class Output:
    goal: torch.Tensor
    subgoals: torch.Tensor
    initial_subgoals: torch.Tensor
    pre: State
    post: State


class GraphLooper(nn.Module):
    def __init__(self, code_width=2, pre_loops=5, post_loops=3):
        super().__init__()
        self.code_width = code_width
        self.pre_loops = pre_loops
        self.post_loops = post_loops
        self.atom_encoder = nn.Linear(5, 16)
        self.rule_encoder = nn.Linear(3, 16)
        self.cell = native_cell_class()(16)
        self.codec = GraphCodec(code_width)
        self.goal_head = nn.Linear(16, 2)
        self.subgoal_head = nn.Linear(16, 2)

    def _step(self, a, r, prem, head):
        a, r = self.cell(a, r, prem, head)
        az, rz = self.codec.encode(a, r)
        a, r = self.codec.decode(az, rz)
        return State(a, r, az, rz)

    def _read(self, a, positions):
        width = a.shape[-1]
        picked = a.gather(1, positions.unsqueeze(-1).expand(-1, -1, width))
        return self.subgoal_head(picked[:, :2]), self.goal_head(picked[:, 2])

    def prefix(self, batch: GraphBatch, loops=None):
        a = self.atom_encoder(batch.atom_features)
        r = self.rule_encoder(batch.rule_features)
        for _ in range(self.pre_loops if loops is None else loops):
            s = self._step(a, r, batch.prem_up, batch.head_up)
            a, r = s.a, s.r
        return s

    def suffix(self, batch: GraphBatch, start: State):
        a, r = start.a, start.r
        for _ in range(self.post_loops):
            s = self._step(a, r, batch.prem_down, batch.head_down)
            a, r = s.a, s.r
        q, goal = self._read(s.a, batch.read_slots)
        return goal, q, s

    def forward(self, batch: GraphBatch, pre_loops=None):
        pre = self.prefix(batch, pre_loops)
        first, _ = self._read(pre.a, batch.read_slots)
        goal, q, post = self.suffix(batch, pre)
        return Output(goal, q, first, pre, post)

    def patched(self, recipient: GraphBatch, donor: GraphBatch,
                selected: torch.Tensor, *, condition='addressed'):
        if recipient.read_slots.shape != donor.read_slots.shape:
            raise ValueError('donor and recipient batch shapes differ')
        # Explicitly guard the public graph, downstream goal, and untouched facts.
        for name in ('rule_features', 'prem_up', 'head_up', 'prem_down', 'head_down', 'read_slots'):
            if not torch.equal(getattr(recipient, name), getattr(donor, name)):
                raise ValueError(f'mismatched structural donor: {name}')
        if selected.shape != (len(recipient.targets),) or \
           not bool(((selected == 0) | (selected == 1)).all()):
            raise ValueError('invalid selected subgoal')
        r = self.prefix(recipient)
        d = self.prefix(donor)
        n = len(selected)
        if condition == 'addressed':
            addresses = selected
        elif condition == 'other':
            addresses = 1 - selected
        elif condition == 'whole':
            addresses = None
        elif condition == 'self':
            addresses = selected
        else:
            raise ValueError('unknown patch condition')
        if condition == 'whole':
            az, rz = d.az, d.rz
        elif condition == 'self':
            az, rz = r.az, r.rz
        else:
            positions = recipient.read_slots.gather(1, addresses[:, None]).squeeze(-1)
            mask = F.one_hot(positions, num_classes=ATOMS).to(torch.bool)
            az = torch.where(mask.unsqueeze(-1), d.az, r.az)
            rz = r.rz
        a, rule = self.codec.decode(az, rz)
        goal, q, post = self.suffix(recipient, State(a, rule, az, rz))
        return Output(goal, q, self._read(r.a, recipient.read_slots)[0],
                      State(r.a, r.r, az, rz), post)


def parameter_count(model):
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'native_cell_parameters': sum(p.numel() for p in model.cell.parameters()),
            'raw_state_floats': (ATOMS+RULES)*16,
            'carried_state_floats': (ATOMS+RULES)*model.code_width,
            'compressed_ratio': 16 / model.code_width}
