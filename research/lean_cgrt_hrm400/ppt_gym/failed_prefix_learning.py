"""Typed saved-prefix inputs and exact batched policy returns."""

import torch
from torch.nn import functional as F

from .failed_prefix import ALLOWANCES, CHARGE, legal
from .guard_fork_policy import guard_tokens, numeric_description
from .live_quote_policy import pad_features


STATES = tuple((remaining, failed) for remaining in range(max(ALLOWANCES) + 1) for failed in range(8))
COUNT = len(STATES)
STARTS = tuple(budget * 8 for budget in ALLOWANCES)
TEMPERATURE = .12


def observations(routes, warm):
    descriptors = []
    for route in routes:
        token = guard_tokens(route["guard"])
        if token is None:
            raise ValueError("the public recovery guard is outside the typed grammar")
        descriptors.append(numeric_description(token))
    visited, queue = set(), [(budget, 0) for budget in ALLOWANCES]
    while queue:
        remaining, failed = queue.pop()
        if (remaining, failed) in visited:
            continue
        visited.add((remaining, failed))
        for arm in legal(routes, remaining, failed):
            queue.append((remaining - routes[arm]["fail_cost"], failed | (1 << arm)))
    tokens, indices = [], []
    menu = [[False] * 3 for _ in STATES]
    next_state = [[0] * 3 for _ in STATES]
    for remaining, failed in sorted(visited):
        index = remaining * 8 + failed

        def describe(arm):
            route = routes[arm]
            return pad_features([*descriptors[arm], route["pass_cost"] / 7, route["fail_cost"] / 7,
                                 bool(failed & (1 << arm)), arm == 0, remaining / 7])

        retired = [float(bool(failed & (1 << arm))) for arm in range(3)]
        for arm in legal(routes, remaining, failed):
            tokens.append([describe(0), describe(arm),
                           pad_features([remaining / 7, *retired, *(route["pass_cost"] / 7 for route in routes),
                                         arm == 0, arm == 1, *warm[arm]])])
            indices.append(index * 3 + arm)
            menu[index][arm] = True
            following = remaining - routes[arm]["fail_cost"]
            next_state[index][arm] = following * 8 + (failed | (1 << arm))
    return tokens, indices, menu, next_state


def dense_scores(model, tokens, indices, menu):
    raw = model(tokens.flatten(0, 1)).reshape(indices.shape)
    flat = raw.new_zeros(len(tokens), COUNT * 3 + 1).scatter(1, indices, raw)
    return flat[:, :-1].reshape_as(menu).masked_fill(~menu, -torch.inf)


def returns(next_state, menu, pass_cost, fail_cost, truth, policy=None):
    """Replay each native outcome over the known acyclic saved-snapshot graph."""
    values = pass_cost.new_zeros(len(truth), 8)
    optimum = [values.new_zeros(len(truth), 8, 3)]
    for remaining in range(1, max(ALLOWANCES) + 1):
        span = slice(8 * remaining, 8 * (remaining + 1))
        following = next_state[:, span]
        continuation = values.gather(1, following.flatten(1)).reshape_as(following)
        q = torch.where(truth[:, None], 1 - CHARGE * pass_cost[:, None], continuation - CHARGE * fail_cost[:, None])
        legal_q = q.masked_fill(~menu[:, span], -torch.inf)
        if policy is None:
            optimum.append(legal_q)
            layer = legal_q.amax(-1).clamp_min(0.)
        else:
            layer = (q * policy[:, span]).sum(-1)
        values = torch.cat((values, layer), dim=1)
    return torch.cat(optimum, dim=1) if policy is None else values[:, STARTS]


def policy_distribution(scores):
    return F.pad(scores / TEMPERATURE, (0, 1)).softmax(dim=-1)[..., :3]
