"""Recovery choices that retain or abandon an already discharged Lean goal."""

from functools import lru_cache
from itertools import product
from math import fsum
import random

from .computational_guards import FAMILIES, GUARD_CHECK, PREFIX, equation
from .data import digest


SCHEMA = "ppt.failed-prefix-recovery.v1"
SETUP = (*PREFIX, "apply hp0", GUARD_CHECK)
REJECTION = ("apply hbad", GUARD_CHECK)
PREFIX_COST = len(SETUP) + len(REJECTION)
ALLOWANCES = (3, 5, 7)
CHARGE = .025
WORLDS = tuple(bits for bits in product((False, True), repeat=3) if 0 < sum(bits) < 3)


def cases(partition, count):
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        rng = random.Random(int(digest([SCHEMA, partition, index])[:15], 16))
        known = equation(family, rng, partition)
        rejected = equation(family, rng, partition)
        forced = (known.claim(), rejected.claim((rejected.value + 1) % 7 if rejected.value < 7 else 0))
        bits = [True, False, bool(rng.getrandbits(1))]
        rng.shuffle(bits)
        claims = []
        for bit in bits:
            guard = equation(family, rng, partition)
            claims.append(guard.claim() if bit else guard.claim(guard.value + 1))
        before = rng.sample((0, 0, 1, 2), 3)
        after = [rng.randrange(3) for _ in bits]
        witnesses = rng.sample(range(2, 64), 3)
        pair = digest([SCHEMA, family, forced, sorted(claims), before, after, witnesses])
        for world in range(2):
            offset = 0 if world == 0 else rng.choice((1, 2))
            order = [(j + offset) % 3 for j in range(3)]
            bound = ["(P Q : Nat → Prop)", f"(hp0 : ({forced[0]}) → P {witnesses[0]})",
                     f"(hbad : ({forced[1]}) → Q {witnesses[0]})"]
            routes = []
            for arm, item in enumerate(order):
                target = f"{'P' if arm else 'Q'} {witnesses[arm]}"
                name = f"hr{arm}"
                binders = ["True"] * before[arm] + [f"({claims[item]})"] + ["True"] * after[arm]
                bound.append(f"({name} : {' → '.join([*binders, target])})")
                tail = []
                if arm:
                    bound.append(f"(hq{arm} : Q {witnesses[arm]})")
                    tail.append(f"exact hq{arm}")
                attempt = [f"apply {name}", *(["trivial"] * before[arm]), GUARD_CHECK]
                suffix = [*(["trivial"] * after[arm]), *tail]
                routes.append({"guard": claims[item], "attempt": attempt, "suffix": suffix,
                               "pass_cost": len(attempt + suffix), "fail_cost": len(attempt)})
            statement = f"∀ {' '.join(bound)}, ∃ n : Nat, P n ∧ Q n"
            yield {"pair": pair, "module": digest([pair, world]), "family": family, "world": world,
                   "statement": statement, "routes": routes, "expected": [bits[idx] for idx in order]}


def legal(routes, remaining, discarded):
    return tuple(index for index, route in enumerate(routes)
                 if not discarded & (1 << index) and route["pass_cost"] <= remaining)


def trace_case(case, allowance, select):
    remaining, discarded, events, proved = allowance, 0, [], False
    while menu := legal(case["routes"], remaining, discarded):
        action = select(remaining, discarded, menu)
        if action is None:
            break
        if action not in menu:
            raise ValueError("the recovery policy chose an unavailable Lean continuation")
        route = case["routes"][action]
        proved = case["truth"][action]
        charge = route["pass_cost" if proved else "fail_cost"]
        events.append({"arm": action, "before": remaining, "passed": proved, "charge": charge})
        remaining -= charge
        if proved:
            break
        discarded |= 1 << action
    checks = PREFIX_COST + allowance - remaining
    return {"allowance": allowance, "trace": events, "checks": checks, "proved": proved,
            "utility": float(proved) - CHARGE * checks}


def planned_values(routes, weights):
    """Plan exact utility given a frozen prior over the seven truth patterns."""
    @lru_cache(maxsize=None)
    def scores(remaining, failed):
        members = [(bits, weight) for bits, weight in zip(WORLDS, weights, strict=True)
                   if all(not (failed & (1 << index)) or not ok for index, ok in enumerate(bits))]
        total = fsum(weight for _, weight in members)
        if not total:
            return {}
        result = {}
        for arm in legal(routes, remaining, failed):
            route = routes[arm]
            following = route["fail_cost"]
            update = failed | (1 << arm)
            continuation = max((0., *scores(remaining - following, update).values()))
            result[arm] = fsum(weight * (1 - CHARGE * route["pass_cost"] if bits[arm]
                                         else continuation - CHARGE * following)
                               for bits, weight in members) / total
        return result

    return scores


def probability_weights(probability):
    return [product_probability(p if bit else 1 - p for p, bit in zip(probability, world, strict=True))
            for world in WORLDS]


def product_probability(values):
    weight = 1.
    for factor in values:
        weight *= factor
    return weight
