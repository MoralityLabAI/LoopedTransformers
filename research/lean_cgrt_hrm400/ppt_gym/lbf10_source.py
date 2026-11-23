"""LBF-10 opt-in structural alternatives. The legacy generator is unchanged."""
from __future__ import annotations

import random
from lcgrt import fragment as F

FAMILY = "lbf10_independent_equal_cost_diamonds.v1"


def generate(seed: int, depth: int, family: str, *, tid: str) -> F.Theorem:
    thm = F.generate(seed, depth, family, arity3=False, tid=tid)
    old_n = thm.n_atoms
    old_size = F.reference_sizes(thm)
    heads = sorted({head for (_, head), kind in
                    zip(thm.rules, thm.gen["rule_kind"])
                    if kind == "sol" and 2 <= old_size[head] <= 8})
    if not heads:
        raise ValueError("structural augmentation requires an eligible solution head")
    kinds = list(thm.gen["rule_kind"])
    groups = []

    def pair():
        a = thm.n_atoms
        thm.n_atoms += 2
        return [a, a + 1]

    def add(pre, head, kind):
        thm.rules.append([list(pre), head])
        kinds.append(kind)

    for head in heads:
        d = old_size[head]
        group = {"head": head, "cost": d, "live": [], "dead": []}
        for live in (True, False):
            levels = [pair() for _ in range(d - 1)]
            group["live" if live else "dead"] = levels
            if live:
                thm.facts.extend(levels[0])
            for lower, upper in zip(levels, levels[1:]):
                for a in lower:
                    for b in upper:
                        add([a], b, "lbf10_live" if live else "lbf10_dead")
            for a in levels[-1]:
                add([a], head, "lbf10_live" if live else "lbf10_dead")
        groups.append(group)
    sizes = F.reference_sizes(thm)
    assert all(sizes[a] == old_size[a] for a in range(old_n))
    assert len(thm.facts) == len(set(thm.facts))
    for group in groups:
        for i, level in enumerate(group["live"]):
            assert all(sizes[a] == i + 1 for a in level)
        assert all(sizes[a] == F.INF for level in group["dead"] for a in level)

    # Randomize all public binder names, including original and added rules.
    rng = random.Random(f"lbf10-diamonds-v1|{seed}|{depth}|{family}")
    perm = list(range(thm.n_atoms)); rng.shuffle(perm)
    order = list(range(len(thm.rules))); rng.shuffle(order)
    facts = [perm[a] for a in thm.facts]; rng.shuffle(facts)
    thm.facts = facts
    thm.rules = [[[perm[p] for p in thm.rules[i][0]], perm[thm.rules[i][1]]]
                 for i in order]
    thm.goal = [perm[a] for a in thm.goal]
    thm.gen["rule_kind"] = [kinds[i] for i in order]
    thm.gen["solution_atoms"] = sorted(perm[a] for a in thm.gen["solution_atoms"])
    thm.gen["lbf10"] = {
        "family": FAMILY,
        "groups": [{"head": perm[g["head"]], "cost": g["cost"],
                    "live": [[perm[a] for a in lev] for lev in g["live"]],
                    "dead": [[perm[a] for a in lev] for lev in g["dead"]]}
                   for g in groups],
    }
    return thm
