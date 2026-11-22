"""Trainable directed evidence propagation for two-slot equality search."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
import math

import torch
from torch import nn

from .data import canonical
from .slot_bindings import projected_goal


MAX_TERMS = 32
MAX_EDGES = 64
HOPS_PER_UPDATE = 2


@dataclass(frozen=True)
class PathView:
    sources: torch.Tensor
    destinations: torch.Tensor
    edge_features: torch.Tensor
    seeds: torch.Tensor
    read_indices: torch.Tensor
    read_slots: torch.Tensor
    counts: torch.Tensor
    shape: tuple[int, int]


@lru_cache(maxsize=128)
def path_view(packet, actions):
    goals = json.loads(packet)
    sources, destinations, features, seeds, read_indices, read_slots = [], [], [], [], [], []
    counts = []
    for action in actions:
        for slot, goal in enumerate(goals):
            scoped_action = action if slot == 0 or action.startswith("all_goals ") else None
            targets, edges, _ = projected_goal(goal, scoped_action)
            counts.append(len(targets))
            ordered_edges = sorted(edges)
            outgoing = Counter(source for source, _ in ordered_edges)
            for left, right in targets:
                terms = sorted({left, right, *(term for pair in ordered_edges for term in pair)})
                if len(terms) > MAX_TERMS or len(edges) > MAX_EDGES:
                    raise ValueError("the directed slot bank exceeded its registered graph cap")
                ids = {term: len(seeds) + i for i, term in enumerate(terms)}
                for source, destination in ordered_edges:
                    sources.append(ids[source])
                    destinations.append(ids[destination])
                    features.append((float(destination == right), float(outgoing[source] > 1),
                                     float(outgoing[destination] > 1)))
                read_slots.append(len(counts) - 1)
                read_indices.append(ids[left])
                seeds.extend(float(term == right) for term in terms)
    return PathView(torch.tensor(sources, dtype=torch.long), torch.tensor(destinations, dtype=torch.long),
                    torch.tensor(features, dtype=torch.float32).reshape(-1, 3),
                    torch.tensor(seeds, dtype=torch.float32), torch.tensor(read_indices, dtype=torch.long),
                    torch.tensor(read_slots, dtype=torch.long), torch.tensor(counts, dtype=torch.float32),
                    (len(actions), len(goals)))


class SlotPathBank(nn.Module):
    """Read the current and newly arrived evidence after each directed update."""

    def __init__(self, width):
        super().__init__()
        self.edge_gate = nn.Linear(3, 1)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.constant_(self.edge_gate.bias, math.log(4.))
        self.read = nn.Linear(2, width, bias=False)

    def macs(self, actions, updates):
        # At most two slots with two atomic targets each.
        return 4 * actions * (MAX_EDGES * (3 + HOPS_PER_UPDATE * updates) + 2 * updates * self.read.out_features)

    def start(self, goals, actions):
        view = path_view(canonical(goals), tuple(actions))
        return view, view.seeds, self.edge_gate(view.edge_features).squeeze(-1).sigmoid()

    def advance(self, view, bank, gates):
        previous = bank
        for _ in range(HOPS_PER_UPDATE):
            messages = bank[view.destinations] * gates
            bank = bank.scatter_reduce(0, view.sources, messages, reduce="amax")
        current = bank[view.read_indices]
        observations = torch.stack((current, current - previous[view.read_indices]), dim=-1)
        pooled = bank.new_zeros((len(view.counts), 2)).index_add(0, view.read_slots, observations)
        signals = pooled / view.counts.clamp_min(1).unsqueeze(1)
        return bank, self.read(signals).reshape(*view.shape, -1)
