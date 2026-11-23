"""Source exploration, matched credit targets, and scored guard choices."""

from collections import defaultdict
from functools import lru_cache
from math import fsum
import random

import torch

from .data import digest
from .guard_fork_policy import guard_tokens, numeric_description
from .guard_recovery import ALLOWANCES, CHARGE, ENTER, PROBE, VERIFY, initial, menu, transition
from .live_quote_policy import pad_features


def behavior(state, rng, kind, preferred):
    choices = menu(state)
    if kind == 0:
        return rng.choice(choices)
    if state.branch >= 0:
        return rng.choice(choices)
    probes = [move for move in choices if move in PROBE and PROBE.index(move) // 2 == preferred]
    if kind == 1 and ENTER[preferred] in choices and state.remaining >= 5 and probes:
        return rng.choice(probes)
    return ENTER[preferred] if ENTER[preferred] in choices else next(move for move in choices if move in ENTER)


def source_targets(case):
    rng = random.Random(int(digest(["guard-tapes", case["module"]])[:15], 16))
    edges, returns = {}, defaultdict(list)
    for allowance in ALLOWANCES:
        for run in range(24):
            state = initial(allowance)
            preferred = rng.randrange(2)
            tape = []
            while state is not None and menu(state):
                move = behavior(state, rng, run % 3, preferred)
                following, observation = transition(state, move, case["truth"])
                reward = float(observation is True and move == "close") - CHARGE
                tape.append(((state, move), reward))
                edges[state, move] = following, reward
                state = following
            utility = 0.
            for key, reward in reversed(tape):
                utility += reward
                returns[key].append(utility)

    @lru_cache(maxsize=None)
    def value(state):
        return max((0., *(score(state, move) for move in menu(state) if (state, move) in edges)))

    def score(state, move):
        following, reward = edges[state, move]
        return reward + (value(following) if following is not None else 0.)

    return [(state, move, score(state, move), fsum(returns[state, move]) / len(returns[state, move]))
            for state, move in edges]


def raw_guards(case):
    tokens = [guard_tokens(claim) for group in case["guards"] for claim in group]
    if any(token is None for token in tokens):
        raise ValueError("a native guard expression was lost by the public encoder")
    return [numeric_description(token) for token in tokens]


def action_features(descriptors, predicted, warm, state, action):
    if action in ENTER:
        branch = ENTER.index(action)
        first, other = 2 * branch, 2 * branch + 1
    elif action in PROBE:
        first = PROBE.index(action)
        other = first ^ 1
        branch = first // 2
    elif action in VERIFY:
        branch = state.branch
        first = 2 * branch + VERIFY.index(action)
        other = first ^ 1
    else:
        branch = state.branch
        first, other = 2 * branch, 2 * branch + 1
    def port(index):
        return [*descriptors[index], predicted[index], (1 + state.known[index]) / 2,
                bool(state.failed & (1 << (index // 2)))]
    controls = [state.remaining / 6, action in ENTER, action in PROBE, action in VERIFY, action == "close",
                state.branch == 0, state.branch == 1, state.checked / 3, state.failed / 3,
                *((bit + 1) / 2 for bit in state.known), *predicted, *warm[branch]]
    return [pad_features(port(first)), pad_features(port(other)), pad_features(controls)]


def baked_policy(predicted, *, action_space=menu):
    """Use the direct supervised guard probabilities as a full model-based control."""
    truths = [tuple(i != bad for i in range(4)) for bad in range(4)]
    weights = [(1 - predicted[i]) * product_probability((predicted[j] for j in range(4) if j != i))
               for i in range(4)]

    def posterior(state):
        ids = [i for i, row in enumerate(truths) if all(bit < 0 or row[k] == bit for k, bit in enumerate(state.known))]
        return [(truths[i], weights[i]) for i in ids]

    @lru_cache(maxsize=None)
    def value(state):
        return max((0., *(score(state, move) for move in action_space(state))))

    def score(state, action):
        beliefs = posterior(state)
        if not beliefs:
            return 0.
        expected = []
        for _truth, weight in beliefs:
            child, seen = transition(state, action, _truth)
            terminal = float(action == "close" and seen is True)
            expected.append(weight * (terminal + (value(child) if child is not None else 0.)))
        return fsum(expected) / fsum(weight for _, weight in beliefs) - CHARGE

    def choose(state):
        best = max(action_space(state), key=lambda action: score(state, action))
        return best if score(state, best) > 0. else None

    return choose


def product_probability(values):
    result = 1.
    for factor in values:
        result *= factor
    return result


def trace_policy(case, allowance, select, *, swap_after_probe=False, action_space=menu):
    state, trace = initial(allowance), []
    proof = []
    changed, patched_at = None, None
    while state is not None and action_space(state):
        visible = state
        if changed is not None:
            bit = state.known[changed]
            inputs = (*state.known[:changed], 1 - bit, *state.known[changed + 1:])
            visible = state._replace(known=inputs)
            patched_at = len(trace)
            changed = None
        move = select(visible, action_space(state))
        if move is None:
            break
        following, observed = transition(state, move, case["truth"])
        trace.append({"action": move, "branch": state.branch, "observation": observed, "remaining": state.remaining})
        if move == "close":
            last_opening = max(idx for idx, event in enumerate(trace) if event["action"] == ENTER[state.branch])
            first_check = next(event["action"] for event in trace[last_opening:] if event["action"] in VERIFY)
            proof = case["routes"][f"{state.branch}_{VERIFY.index(first_check)}"]["proof"]
            if not proof:
                raise ValueError("a successful policy trace has no checked native script")
        if swap_after_probe and move in PROBE and patched_at is None:
            changed = PROBE.index(move)
        state = following
    checks = len(trace) + 2
    return {"allowance": allowance, "proved": bool(proof), "proof": proof, "trace": trace, "checks": checks,
            "utility": bool(proof) - CHARGE * checks, "patched_at": patched_at}
