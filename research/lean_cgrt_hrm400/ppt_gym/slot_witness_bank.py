"""Typed witness substitution followed by tied backward-proof messages."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json

import torch

from .data import canonical
from .dependent_witness import action_projection, horn_graph, proof_hypotheses
from .slot_path_bank import MAX_EDGES, MAX_TERMS, PathView, SlotPathBank


@dataclass(frozen=True)
class WitnessWorkspace:
    keys: tuple[tuple[str, str], ...]
    values: torch.Tensor


@lru_cache(maxsize=128)
def witness_graph(packet, actions):
    """Use a single graph per local context for every candidate and slot."""
    goals = json.loads(packet)
    contexts = tuple(canonical(list(proof_hypotheses(goal))) for goal in goals)
    targets = [target for action in actions for target in action_projection(goals, action)]
    read_indices = [0] * len(targets)
    keys = []
    sources, inputs, partners, features, seeds = [], [], [], [], []
    closed = canonical({"closed": True})
    for context in dict.fromkeys(contexts):
        positions = [i for i in range(len(targets)) if contexts[i % len(goals)] == context]
        group_targets = [targets[i] for i in positions]
        terms, facts, edges, active = horn_graph(goals[contexts.index(context)], group_targets)
        if len(terms) > MAX_TERMS or len(edges) > MAX_EDGES:
            raise ValueError("the witness bank exceeded its registered graph cap")
        outgoing = Counter(head for head, _, _ in edges)
        ordered_terms = sorted(terms)
        ids = {key: len(seeds) + i for i, key in enumerate(ordered_terms)}
        for head, left, right in sorted(edges):
            sources.append(ids[head])
            inputs.append(ids[left])
            partners.append(ids[right] if right else -1)
            features.append((float(bool(right)), float(outgoing[head] > 1), float(outgoing[left] > 1)))
        seeds.extend(1. if key == closed else .8 if key in facts else 0. for key in ordered_terms)
        keys.extend((context, term) for term in ordered_terms)
        for i, goal in zip(positions, active, strict=True):
            read_indices[i] = ids[goal]
    count = len(targets)
    view = PathView(torch.tensor(sources, dtype=torch.long), torch.tensor(inputs, dtype=torch.long),
                    torch.tensor(features, dtype=torch.float32).reshape(-1, 3),
                    torch.tensor(seeds, dtype=torch.float32), torch.tensor(read_indices, dtype=torch.long),
                    torch.arange(count), torch.ones(count, dtype=torch.float32),
                    (len(actions), len(goals)), torch.tensor(partners, dtype=torch.long))
    return view, tuple(keys)


def witness_view(packet, actions):
    return witness_graph(packet, actions)[0]


class SlotWitnessBank(SlotPathBank):
    """Use the same tied message update on instantiated unary and conjunctive goals."""

    def start(self, goals, actions):
        view, bank, gates, _ = self.resume(goals, actions, None)
        return view, bank, gates

    def resume(self, goals, actions, previous):
        view, keys = witness_graph(canonical(goals), tuple(actions))
        bank = view.seeds
        if previous is not None:
            positions = {key: i + 1 for i, key in enumerate(previous.keys)}
            gather = torch.tensor([positions.get(key, 0) for key in keys], dtype=torch.long)
            padded = torch.cat((previous.values.new_zeros(1), previous.values))
            bank = torch.maximum(bank, padded[gather])
        gates = self.edge_gate(view.edge_features).squeeze(-1).sigmoid()
        return view, bank, gates, keys
