"""Shared false-guard beliefs and differentiable finite-horizon Lean planning."""

import torch
from torch import nn

from .guard_recovery import ACTIONS, ALLOWANCES, CHARGE, feasible_menu, initial, transition
from .guard_recovery_learning import GUARD_WORLDS, supports
from .live_quote_policy import FEATURES, pad_features


LOGIT_BOUND = .75


def belief_inputs(descriptors, probabilities, warm):
    """Orient the two witness tokens around each hypothesized false guard."""
    rows = []
    for focus in range(4):
        other = focus ^ 2
        order = (focus, focus ^ 1, other, other ^ 1)

        def witness(index):
            return pad_features([*descriptors[index], *descriptors[index ^ 1][:5],
                                 probabilities[index], probabilities[index ^ 1]])

        context = [*(probabilities[index] for index in order), *warm[focus // 2], *warm[other // 2]]
        rows.append([witness(focus), witness(other), pad_features(context)])
    return rows


class BeliefCorrection(nn.Module):
    def __init__(self, actor=None):
        super().__init__()
        self.actor = actor if actor is not None else nn.Sequential(nn.Flatten(1), nn.Linear(3 * FEATURES, 1))
        head = actor.head if actor is not None else self.actor[-1]
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def correction(self, tokens):
        outputs = self.actor(tokens.flatten(0, 1)).reshape(len(tokens), 4)
        return LOGIT_BOUND * outputs.tanh()

    def forward(self, tokens, prior):
        return prior + self.correction(tokens)


class BeliefPlanner:
    """Batch the public transition graph so Q loss can train a root belief."""

    def __init__(self):
        pending = [initial(budget) for budget in ALLOWANCES]
        graph = {}
        while pending:
            state = pending.pop()
            menu = feasible_menu(state)
            if not menu or state in graph:
                continue
            support = [supports(state, truth) for truth in GUARD_WORLDS]
            moves = {}
            for action in menu:
                following = [transition(state, action, truth)[0] if allowed else None
                             for allowed, truth in zip(support, GUARD_WORLDS, strict=True)]
                moves[action] = following
                pending.extend(child for child in following if child is not None)
            graph[state] = support, moves

        self.states = sorted(graph)
        self.index = {state: index for index, state in enumerate(self.states, 1)}
        self.layers = []
        for remaining in sorted({state.remaining for state in self.states}):
            states = [state for state in self.states if state.remaining == remaining]
            support, children, legal = [], [], []
            for state in states:
                worlds, moves = graph[state]
                support.append(worlds)
                legal.append([action in moves for action in ACTIONS])
                children.append([[self.index.get(child, 0) for child in moves.get(action, (None,) * 4)]
                                 for action in ACTIONS])
            self.layers.append(tuple(torch.tensor(rows) for rows in (support, children, legal)))
        self.reward = torch.tensor([float(action == "close") - CHARGE for action in ACTIONS])

    def scores(self, logits):
        """Return dense Q by indexed public state and action, with row zero terminal."""
        values = logits.new_zeros(len(logits), 1)
        scores = [logits.new_zeros(len(logits), 1, len(ACTIONS))]
        for support, children, legal in self.layers:
            posterior = logits[:, None, :].masked_fill(~support, -torch.inf).softmax(dim=-1)
            q = (values[:, children] * posterior.unsqueeze(2)).sum(dim=-1)
            q = (q + self.reward.to(logits)).masked_fill(~legal, -torch.inf)
            scores.append(q)
            values = torch.cat((values, q.amax(dim=-1).clamp_min(0.)), dim=1)
        return torch.cat(scores, dim=1)
