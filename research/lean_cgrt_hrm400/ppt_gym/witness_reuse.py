"""Paired Lean proof plans with a choice to prove and reuse one local fact."""

from dataclasses import dataclass
from itertools import combinations, product
import random

from .data import digest
from .dependent_witness import FORK, PREFIX


FAMILY = "ppt-witness-reuse"
CACHE = "h999"
OPTIONS = ("direct", "cache_first", "cache_second")


@dataclass(frozen=True, order=True)
class Route:
    chains: tuple[int, int]
    tails: tuple[int, int]
    uses: tuple[int, int]

    def costs(self):
        direct = len(PREFIX) + 3 + sum(
            tail + self.chains[chain] for tail, chain in zip(self.tails, self.uses, strict=True)
        )
        return (direct, *(direct + 1 + length - self.uses.count(i) * (length - 1)
                          for i, length in enumerate(self.chains)))

    def swap_cache_names(self):
        return Route(self.chains[::-1], self.tails, tuple(1 - use for use in self.uses))


def layouts():
    """Balance single and competing reuse opportunities across disjoint sets."""
    by_cost = {}
    routes = [Route(chains, tails, uses)
              for chains in product(range(2, 7), repeat=2)
              for tails in product(range(1, 4), repeat=2)
              for uses in product(range(2), repeat=2)]
    for route in routes:
        costs = route.costs()
        if 14 <= costs[0] <= 19 and route < route.swap_cache_names():
            by_cost.setdefault(costs[0], []).append((route, min(costs)))
    groups = {False: [], True: []}
    for routes_at_cost in by_cost.values():
        for (left, left_cost), (right, right_cost) in combinations(routes_at_cost, 2):
            if left_cost != right_cost:
                groups[left.uses[0] == left.uses[1] and right.uses[0] == right.uses[1]].append((left, right))

    rng = random.Random(831)
    for group in groups.values():
        rng.shuffle(group)
    for split, start, stop in (("train", 0, 8), ("dev", 8, 20)):
        for group in groups.values():
            for sides in group[start:stop]:
                yield split, sides


def statement_and_plans(sides, names):
    """Use shared proof names only through the type-directed tactic menu."""
    hypotheses = [f"{names['P']} 0", f"{names['P']} 1"]
    plans = []
    for witness, side in enumerate(sides):
        def fact(ty, verb="apply"):
            hypotheses.append(ty)
            return f"{verb} h{len(hypotheses) - 1}"

        def atom(symbol):
            return f"{names[symbol]} {witness}"

        def chain(nodes):
            seed = fact(nodes[0], "exact")
            arrows = [fact(f"{left} → {right}") for left, right in zip(nodes, nodes[1:])]
            return (*reversed(arrows), seed)

        endpoints = [atom(f"C{i}") for i in range(2)]
        builders = [chain([*(atom(f"C{i}N{j}") for j in range(size - 1)), endpoints[i]])
                    for i, size in enumerate(side.chains)]
        obligations = [atom("U"), atom("V")]
        tails = []
        for i, (goal, length, use) in enumerate(zip(obligations, side.tails, side.uses, strict=True)):
            nodes = [endpoints[use], *(atom(f"T{i}N{j}") for j in range(length - 1)), goal]
            steps = [fact(f"{left} → {right}") for left, right in zip(nodes, nodes[1:])]
            tails.append(tuple(reversed(steps)))
        expand = fact(f"({' ∧ '.join(obligations)}) → {atom('Q')}")
        root = (*PREFIX, FORK[witness], expand)

        options = []
        for cache in (None, 0, 1):
            have = () if cache is None else (f"have {CACHE} : {endpoints[cache]} := ?_",)
            build = () if cache is None else builders[cache]
            close = [(*tail, *((f"exact {CACHE}",) if use == cache else builders[use]))
                     for tail, use in zip(tails, side.uses, strict=True)]
            options.append((*root, *have, "constructor", *close[0], *close[1], *build))
        plans.append(tuple(options))

    predicates = " ".join(sorted(names.values()))
    binders = tuple(f"h{i}" for i in range(len(hypotheses)))
    local_types = " ".join(f"({name} : {ty})" for name, ty in zip(binders, hypotheses, strict=True))
    statement = (f"∀ ({predicates} : Nat → Prop) {local_types}, "
                 f"∃ n, {names['P']} n ∧ {names['Q']} n")
    return statement, (*predicates.split(), *binders), tuple(plans)


def cohort():
    symbols = ["P", "Q", "U", "V", "C0", "C1"]
    symbols.extend(f"C{i}N{j}" for i in range(2) for j in range(5))
    symbols.extend(f"T{i}N{j}" for i in range(2) for j in range(2))
    for split, sides in layouts():
        cluster = digest([FAMILY, [vars(side) for side in sides]])
        rng = random.Random(cluster)
        names = dict(zip(symbols, rng.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), len(symbols)), strict=True))
        checks = min(min(side.costs()) for side in sides)
        for world, routes in enumerate((sides, sides[::-1])):
            # Independently vary option order without changing the paired reachability.
            routes = tuple(side.swap_cache_names() if rng.getrandbits(1) else side for side in routes)
            statement, binders, paths = statement_and_plans(routes, names)
            yield {"split": split, "cluster": cluster, "world": world, "checks": checks,
                   "statement": statement, "binders": binders, "paths": paths,
                   "costs": tuple(route.costs() for route in routes)}
