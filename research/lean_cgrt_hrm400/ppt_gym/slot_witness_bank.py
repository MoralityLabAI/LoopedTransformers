"""Typed witness substitution followed by tied backward-proof messages."""

from collections import Counter
from functools import lru_cache
import json

import torch

from .data import canonical
from .dependent_witness import action_projection
from .slot_path_bank import MAX_EDGES, MAX_TERMS, PathView, SlotPathBank


@lru_cache(maxsize=128)
def witness_view(packet, actions):
    goals = json.loads(packet)
    sources, inputs, partners, features, seeds = [], [], [], [], []
    read_indices, counts = [], []
    for action in actions:
        targets = action_projection(goals, action)
        for goal, target in zip(goals, targets, strict=True):
            terms, facts, edges = set(), set(), set()

            def register(prop):
                key = canonical(prop)
                if key not in terms:
                    terms.add(key)
                    if "and" in prop:
                        lhs, rhs = (register(item) for item in prop["and"])
                        edges.add((key, lhs, rhs))
                return key

            for hyp in goal["hypotheses"]:
                ty = hyp["type"]
                if "imp" in ty:
                    body, head = (register(item) for item in ty["imp"])
                    edges.add((head, body, ""))
                else:
                    facts.add(register(ty))
            active = register(target or {"closed": True})
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
            seeds.extend(1. if key == active and target is None else .8 if key in facts else 0. for key in ordered_terms)
            read_indices.append(ids[active])
            counts.append(1.)
    return PathView(torch.tensor(sources, dtype=torch.long), torch.tensor(inputs, dtype=torch.long),
                    torch.tensor(features, dtype=torch.float32).reshape(-1, 3),
                    torch.tensor(seeds, dtype=torch.float32), torch.tensor(read_indices, dtype=torch.long),
                    torch.arange(len(counts)), torch.tensor(counts), (len(actions), len(goals)),
                    torch.tensor(partners, dtype=torch.long))


class SlotWitnessBank(SlotPathBank):
    """Use the same tied message update on instantiated unary and conjunctive goals."""

    def start(self, goals, actions):
        view = witness_view(canonical(goals), tuple(actions))
        return view, view.seeds, self.edge_gate(view.edge_features).squeeze(-1).sigmoid()
