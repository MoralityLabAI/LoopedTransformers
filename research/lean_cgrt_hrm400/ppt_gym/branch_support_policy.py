"""Encode native feedback in direct and composed evidence slots."""

import torch

from .branch_support import ACTIONS, BRANCHES, OPEN, PROBES, parity_paths
from .live_quote_policy import pad_features


KEYS = tuple(range(1, 8))
PATHS = tuple(parity_paths(key) for key in KEYS)


def tokens(state):
    signs = [1 - 2 * bit if bit >= 0 else 0 for bit in state.observed]
    deck = [float(key in state.deck) / 2 for key in KEYS]
    query = [float(key == state.key) for key in KEYS]
    direct = [signs[PROBES.index(key)] if key in PROBES else 0 for key in KEYS]
    pairs = [next((signs[i] * signs[j] for i, j in links if signs[i] and signs[j]), 0) for links in PATHS]
    rows = []
    for action in range(ACTIONS):
        adding = action < OPEN
        support = [bool(sign) or adding and action == i for i, sign in enumerate(signs)]
        ready = [key in PROBES and support[PROBES.index(key)] or any(support[i] and support[j] for i, j in links)
                 for key, links in zip(KEYS, PATHS, strict=True)]
        key_bits = [(PROBES[action] >> i) & 1 if adding else 0 for i in range(3)]
        context = [state.remaining / 2, state.key != 0, adding, action == OPEN,
                   action == BRANCHES[0], action == BRANCHES[1], *key_bits, *deck, *query]
        rows.append([pad_features([*direct, *deck, *query]),
                     pad_features([*pairs, *ready, *deck, *query]), pad_features(context)])
    return rows


def raw_features(features):
    direct = features[..., 0, :7]
    pairs, attainable = features[..., 1, :7], features[..., 1, 7:14]
    context = features[..., 2, :23]
    deck, query = context[..., 9:16], context[..., 16:23]
    spend = context[..., 0:1]
    branch_sign = context[..., 4:5] - context[..., 5:6]
    query_sign = ((direct + (direct == 0) * pairs) * query).sum(-1, keepdim=True)
    current = (direct != 0) | (pairs != 0)
    coverage = (attainable * deck).sum(-1, keepdim=True)
    known = (current * deck).sum(-1, keepdim=True)
    return torch.cat((torch.ones_like(spend), context, direct, pairs, attainable,
                      branch_sign * query_sign, known, coverage, spend * coverage), dim=-1)
