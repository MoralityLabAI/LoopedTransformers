"""Rank mixed proof openings with a shared recurrence over typed obligations."""

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import torch
from torch import nn

from .data import canonical
from .hybrid_rl.graph_model import visible_graph
from .live_lean import candidate_menu
from .rewrite import rewrite_problem


def obligation_views(packet, menu):
    parsed = json.loads(packet)
    original, = parsed["state"]["goals"]
    hypotheses = {hyp["name"]: hyp["type"] for hyp in original["hypotheses"]}
    views = []
    for tactic in menu:
        equality, implication = hypotheses[tactic.removeprefix("apply ")]["imp"]
        proposition, conclusion = implication["imp"]
        if conclusion != original["target"]:
            raise ValueError("the root rule concludes a different proposition")
        pair = []
        for target in (equality, proposition):
            goal = {**original, "target": target}
            state = {"goals": [{"target": target, "hyps": original["hypotheses"]}]}
            actions = candidate_menu(SimpleNamespace(contexts={}), state, rewrite=rewrite_problem(goal) is not None)
            view = visible_graph(canonical({**parsed, "state": {"goals": [goal]}}), actions)
            if view is None or not actions or not all(view.actions):
                raise ValueError("a projected obligation lacks its public local rules")
            pair.append(view)
        views.append(tuple(pair))
    return tuple(views)


class MixedObligationPolicy(nn.Module):
    """Use the same evidence recurrence for both proposed obligations."""

    def __init__(self, source):
        super().__init__()
        self.architecture = source.architecture
        self.graph = copy.deepcopy(source.graph)
        self.action = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.action.weight.fill_(0.25)

    def macs(self, candidates, passes):
        return candidates * (2 * self.graph.upper_macs(passes) + self.action.in_features)

    def summaries(self, packet, actions, passes):
        branches = []
        for views in obligation_views(packet, actions):
            summaries = []
            for view in views:
                scores, value = self.graph(view, passes)
                summaries.extend((scores.max(), value))
            branches.append(torch.stack(summaries))
        return torch.stack(branches)

    def forward(self, packet, actions, passes):
        logits = self.action(self.summaries(packet, actions, passes)).squeeze(-1)
        return logits, logits.mean()


class MixedPhaseCostPolicy(MixedObligationPolicy):
    """Predict the equality and proposition check counts separately."""

    def __init__(self, source):
        super().__init__(source)
        self.phase_bias = nn.Parameter(torch.ones(2))

    def phase_costs(self, packet, actions, passes):
        features = self.summaries(packet, actions, passes).reshape(len(actions), 2, -1)
        projected = (features * self.action.weight.reshape(2, -1)).sum(-1) + self.phase_bias
        return 1 + torch.nn.functional.softplus(projected)

    def forward(self, packet, actions, passes):
        logits = -self.phase_costs(packet, actions, passes).sum(-1)
        return logits, logits.mean()


class MixedArrivalCostPolicy(MixedPhaseCostPolicy):
    """Use evidence arrival across recurrent passes to estimate each obligation's cost."""

    depths = (1, 2, 3, 4)

    def __init__(self, source):
        super().__init__(source)
        self.action = nn.Linear(4 * len(self.depths), 1, bias=False)
        nn.init.zeros_(self.action.weight)

    def macs(self, candidates, _passes):
        graph_cost = sum(self.graph.upper_macs(depth) for depth in self.depths)
        return candidates * (4 * graph_cost + self.action.in_features)

    def summaries(self, packet, actions, _passes):
        branches = []
        for views in obligation_views(packet, actions):
            history = []
            for view in views:
                neutral = replace(view, features=[[0., active, pending] for _, active, pending in view.features])
                for depth in self.depths:
                    scores, value = self.graph(view, depth)
                    no_evidence_scores, no_evidence_value = self.graph(neutral, depth)
                    history.extend(((scores - no_evidence_scores).max(), value - no_evidence_value))
            branches.append(torch.stack(history))
        return torch.stack(branches)
