"""Public arithmetic rewrites and probabilistic transfer across Lean guards."""

from difflib import SequenceMatcher
from itertools import permutations
import re

import torch

from .data import digest
from .failed_prefix import CHARGE, WORLDS
from .live_quote_policy import pad_features
from .paired_recovery import target_prior


SCHEMA = "ppt.rewritten-recovery.v1"
ALIASES = tuple(permutations(range(3)))
NUMBER = re.compile(r"\d+")
TOKENS = re.compile(r"[a-zA-Z_.]+|\d+|[^\s]")


def unfold_guard(left, steps):
    operator, modulus = left.split(" % ", 1)
    callee, *args = operator.split()
    depth = int(args[-1]) - steps
    term = " ".join((callee, *args[:-1], str(depth)))
    if callee.endswith(".geom"):
        for _ in range(steps):
            term = f"1 + {args[0]} * ({term})"
    elif callee.endswith(".maskAt"):
        namespace = callee.rsplit(".", 1)[0]
        for _ in range(steps):
            term = f"{namespace}.nextMask ({term})"
    else:
        raise ValueError("the guard is outside the registered Lean recurrence families")
    return f"({term}) % {modulus}"


def rewrite(case, partition, *, unfold=False):
    routes = []
    statement = case["statement"]
    for arm, route in enumerate(case["routes"]):
        left, right = route["guard"].split(" = ", 1)
        seed = int(digest([SCHEMA, case["module"], arm])[:12], 16)
        offset, scale = 2 + seed % 11, 2 + (seed // 11) % 4
        if unfold:
            left = unfold_guard(left, 1 + (seed // 44) % 3)
        if partition == "development":
            a, b = f"(({left}) + {offset}) * {scale}", f"({right} + {offset}) * {scale}"
            if seed & 1:
                a, b = b, a
        else:
            style = seed % 3
            if style == 0:
                a, b = f"({left}) + {offset}", f"{right} + {offset}"
            elif style == 1:
                a, b = f"{offset} + {right}", f"{offset} + ({left})"
            else:
                a, b = f"{scale} * {right}", f"{scale} * ({left})"
        transformed = f"{a} = {b}"
        pattern = rf"(\(hr{arm} : (?:True → )*)\({re.escape(route['guard'])}\)"
        statement, replacements = re.subn(pattern, lambda m: f"{m[1]}({transformed})", statement)
        if replacements != 1:
            raise ValueError("the rewrite did not target exactly one Lean recovery binder")
        routes.append({**route, "guard": transformed})
    marker = [SCHEMA, case["module"], "unfold"] if unfold else [SCHEMA, case["module"]]
    return {**case, "module": digest(marker), "statement": statement, "routes": routes,
            "expected": case["truth"]}


def profile(guard):
    numbers = [int(value) for value in NUMBER.findall(guard)]
    tokens = TOKENS.findall(guard)
    histogram = [0.] * 9
    for token in tokens:
        histogram[int(digest(token)[:8], 16) % len(histogram)] += 1. / len(tokens)
    fixed = ([min(val, 64) / 64 for val in numbers[:8]] + [0.] * 8)[:8]
    tail = ([min(val, 64) / 64 for val in numbers[::-1][:6]] + [0.] * 6)[:6]
    return pad_features([*fixed, *tail, *histogram, guard.count("(") / 8, len(tokens) / 40,
                         float("geom" in guard), float("maskAt" in guard)])


def relation(source, target):
    first, second = TOKENS.findall(source), TOKENS.findall(target)
    token_similarity = SequenceMatcher(a=first, b=second, autojunk=False).ratio()
    text_similarity = SequenceMatcher(a=source, b=target, autojunk=False).ratio()
    a, b = set(NUMBER.findall(source)), set(NUMBER.findall(target))
    numeric_overlap = len(a & b) / max(1, len(a | b))
    return token_similarity, text_similarity, numeric_overlap


def pair_tokens(source, target, choice, observed, warm_a, warm_b):
    """Expose public texts and the checked bit without exposing guard correspondence."""
    rows, lexical = [], []
    for destination, route in enumerate(target["routes"]):
        for source_arm, earlier in enumerate(source["routes"]):
            first, second = earlier["guard"], route["guard"]
            matching = relation(first, second)
            public = [*matching, source_arm == choice, 0. if observed is None else float(observed) * 2 - 1,
                      earlier["pass_cost"] / 7, route["pass_cost"] / 7, *warm_a[source_arm], *warm_b[destination]]
            rows.append([profile(first), profile(second), pad_features(public)])
            lexical.append(sum(matching) / len(matching))
    return rows, lexical


def mix_table(source_weights, choice, observed):
    event = (choice, observed) if observed is not None else None
    rows = [target_prior(source_weights, alias, event) for alias in ALIASES]
    return [[weight / sum(row) for weight in row] for row in rows]


def assignment_logits(edge):
    edge = edge.reshape(-1, 3, 3)
    return torch.stack([sum(edge[:, dest, origin] for dest, origin in enumerate(alias))
                        for alias in ALIASES], dim=-1)


def planned_batch(next_state, menu, pass_cost, fail_cost, weights):
    """Differentiable exact continuation under a six-world target belief."""
    worlds = torch.tensor(WORLDS, device=weights.device)
    failed = torch.arange(8, device=weights.device)[:, None].bitwise_and(2 ** torch.arange(3, device=weights.device)).bool()
    support = ~(failed[:, None, :] & worlds[None]).any(dim=-1)
    mass = weights[:, None, :] * support
    posterior = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
    success = posterior @ worlds.to(weights.dtype)
    possible = mass.sum(-1) > 0.
    value = weights.new_zeros(len(weights), 8)
    scores = [weights.new_full((len(weights), 8, 3), -torch.inf)]
    for remaining in range(1, menu.shape[1] // 8):
        span = slice(remaining * 8, (remaining + 1) * 8)
        child = next_state[:, span]
        future = value.gather(1, child.flatten(1)).reshape_as(child)
        q = success * (1. - CHARGE * pass_cost[:, None]) + (1. - success) * (future - CHARGE * fail_cost[:, None])
        q = q.masked_fill(~(menu[:, span] & possible.unsqueeze(-1)), -torch.inf)
        scores.append(q)
        value = torch.cat((value, q.amax(dim=-1).clamp_min(0.)), dim=1)
    return torch.cat(scores, dim=1)
