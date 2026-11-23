"""State-conditioned guard probes with a frozen Lean continuation."""

from functools import lru_cache
from math import exp, fsum, log

import torch
from torch.nn import functional as F

from .guard_belief import BeliefPlanner, belief_inputs
from .guard_recovery import ALLOWANCES, CHARGE, PROBE, feasible_menu, initial, transition
from .guard_recovery_learning import GUARD_WORLDS, baked_scores, false_guard_weights, supports


PROBE_TEMPERATURE = .035


def probe_inputs(descriptors, probability, warm, state):
    tokens = belief_inputs(descriptors, probability, warm)
    for focus, (witness, other, context) in enumerate(tokens):
        rival = focus ^ 2
        order = (focus, focus ^ 1, rival, rival ^ 1)
        for seat, index in ((witness, focus), (other, rival)):
            seat[27:29] = [(1 + state.known[index]) / 2, (1 + state.known[index ^ 1]) / 2]
        context[20:25] = [state.remaining / 6, *((1 + state.known[index]) / 2 for index in order)]
    return tokens


def best_action(scores):
    action = max(scores, key=scores.__getitem__, default=None)
    return action if action is not None and scores[action] > 0. else None


class ProbeGraph:
    """Reduce the public planner to the forks where probe identity can change."""

    def __init__(self):
        self.states = [state for state in BeliefPlanner().states
                       if sum(move in PROBE for move in feasible_menu(state)) >= 2
                       and sum(supports(state, world) for world in GUARD_WORLDS) >= 2]
        self.index = {state: index for index, state in enumerate(self.states, 1)}
        self.starts = [self.index.get(initial(budget), 0) for budget in ALLOWANCES]
        self.support = torch.tensor([[supports(state, world) for world in GUARD_WORLDS] for state in self.states])
        self.available = torch.tensor([[probe in feasible_menu(state) for probe in PROBE] for state in self.states])

    def encode(self, descriptors, probability, warm):
        scores = baked_scores(probability, action_space=feasible_menu)

        @lru_cache(maxsize=None)
        def continuation(state, world):
            if state is None:
                return 0.
            move = best_action(scores(state))
            if move is None:
                return 0.
            child, seen = transition(state, move, GUARD_WORLDS[world])
            return float(move == "close" and seen is True) - CHARGE + continuation(child, world)

        enabled = [best_action(scores(state)) in PROBE for state in self.states]
        payoffs, edges, immediate, offsets = [], [], [], []
        for state, active in zip(self.states, enabled, strict=True):
            menu = scores(state)
            state_values, state_edges, state_rewards = [], [], []
            for probe in PROBE:
                values, children, rewards = [], [], []
                for world, truth in enumerate(GUARD_WORLDS):
                    value, index, reward = 0., 0, 0.
                    if probe in menu and supports(state, truth):
                        child, _ = transition(state, probe, truth)
                        value = continuation(child, world) - CHARGE
                        index = self.index.get(child, 0)
                        if not index or not enabled[index - 1]:
                            index = 0
                        reward = -CHARGE if index else value
                    values.append(value)
                    children.append(index)
                    rewards.append(reward)
                state_values.append(values)
                state_edges.append(children)
                state_rewards.append(rewards)
            payoffs.append(state_values)
            edges.append(state_edges)
            immediate.append(state_rewards)
            offsets.append([menu.get(probe, 0.) for probe in PROBE])

        tokens = [probe_inputs(descriptors, probability, warm, state) for state in self.states]
        prior = [log(weight) for weight in false_guard_weights(probability)]
        baseline = [[continuation(initial(budget), world) for budget in ALLOWANCES]
                    for world in range(len(GUARD_WORLDS))]
        scores.cache_clear()
        continuation.cache_clear()
        return tokens, prior, enabled, offsets, payoffs, edges, immediate, baseline

    def candidate_values(self, prior, correction, offsets, payoffs):
        logits = prior[:, None, :].masked_fill(~self.support, -torch.inf)
        base = logits.softmax(dim=-1)
        revised = (logits + correction).softmax(dim=-1)
        delta = (payoffs * (revised - base).unsqueeze(2)).sum(dim=-1)
        return (offsets + delta).masked_fill(~self.available, -torch.inf)

    def rollout(self, policy, world, enabled, edges, immediate, baseline):
        """Follow actual source checks while routing by public policy probabilities."""
        count = len(world)
        rows = torch.arange(count)
        values = policy.new_zeros(count, 1)
        for state in range(len(self.states)):
            following = edges[rows, state, :, world]
            reward = immediate[rows, state, :, world]
            choice_value = reward + values.gather(1, following)
            expected = (policy[:, state] * choice_value).sum(dim=-1, keepdim=True)
            values = torch.cat((values, expected), dim=1)
        use_policy = F.pad(enabled, (1, 0))[:, self.starts]
        return torch.where(use_policy, values[:, self.starts], baseline[rows, world]) - 2 * CHARGE


def revised_scores(offsets, payoff, prior, correction, support):
    """Scalar anchor preserves the exact frozen tie order at zero correction."""
    base = [exp(weight) if valid else 0. for weight, valid in zip(prior, support, strict=True)]
    revised = [weight * exp(delta) for weight, delta in zip(base, correction, strict=True)]
    old_total, new_total = fsum(base), fsum(revised)
    shift = [after / new_total - before / old_total for before, after in zip(base, revised, strict=True)]
    return {probe: value + fsum(weight * reward for weight, reward in zip(shift, payoff[PROBE.index(probe)], strict=True))
            for probe, value in offsets.items()}
