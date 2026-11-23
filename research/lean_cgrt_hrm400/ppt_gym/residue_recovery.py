"""Native recovery across distinct partitions of one arithmetic observation."""

import random

from .computational_guards import FAMILIES, GUARD_CHECK, geom, mask_at
from .data import digest


SCHEMA = "ppt.residue-recovery.v1"
MODULUS = 9
FIRST_BUDGET = 5


def split_residues(rng, widths):
    labels = list(range(MODULUS))
    rng.shuffle(labels)
    left, centre, _ = widths
    return [sorted(labels[:left]), sorted(labels[left:left + centre]), sorted(labels[left + centre:])]


def cross_partition(rng, source, widths):
    while True:
        target = split_residues(rng, widths)
        if all(sum(bool(set(group) & set(other)) for other in target) >= 2 for group in source):
            return target


def make_case(pair, family, world, widths, groups, carrier, residue, rng):
    witnesses = rng.sample(range(2, 64), 3)
    binders = ["(P Q : Nat → Prop)", f"(hp0 : (1 = 1) → P {witnesses[0]})",
               f"(hbad : (0 = 1) → Q {witnesses[0]})"]
    routes = []
    for arm, group in enumerate(groups):
        guard = " ∨ ".join(f"({carrier}) % {MODULUS} = {item}" for item in group)
        before, after = rng.randrange(2), rng.randrange(3)
        result = f"{'P' if arm else 'Q'} {witnesses[arm]}"
        premises = ["True"] * before + [f"({guard})"] + ["True"] * after
        binders.append(f"(hr{arm} : {' → '.join([*premises, result])})")
        tail = []
        if arm:
            binders.append(f"(hq{arm} : Q {witnesses[arm]})")
            tail.append(f"exact hq{arm}")
        attempt = [f"apply hr{arm}", *(["trivial"] * before), GUARD_CHECK]
        suffix = [*(["trivial"] * after), *tail]
        routes.append({"guard": guard, "residues": group, "attempt": attempt, "suffix": suffix,
                       "pass_cost": len(attempt + suffix), "fail_cost": len(attempt)})
    return {"pair": pair, "module": digest([pair, world]), "family": family, "world": world,
            "statement": f"∀ {' '.join(binders)}, ∃ n : Nat, P n ∧ Q n", "routes": routes,
            "expected": [residue in group for group in groups], "shape": widths}


def cases(partition, count):
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        rng = random.Random(int(digest([SCHEMA, partition, index])[:15], 16))
        if family == FAMILIES[0]:
            base = rng.randrange(2, 9)
            depth = rng.randrange(9, 14) if partition == "development" else rng.randrange(2, 9)
            carrier = f"BPD.OptimalBasis.geom {base} {depth}"
            residue = geom(base, depth) % MODULUS
        else:
            base = 0
            depth = rng.randrange(32, 63) if partition == "development" else rng.randrange(1, 32)
            carrier = f"MathCan.FiniteHorizonGap.maskAt {depth}"
            residue = mask_at(depth) % MODULUS
        source = split_residues(rng, (3, 3, 3))
        widths = rng.sample((2, 3, 4), 3) if partition == "development" else (3, 3, 3)
        target = cross_partition(rng, source, widths)
        pair = digest([SCHEMA, partition, index, carrier, source, target])
        first = make_case(pair, family, 0, (3, 3, 3), source, carrier, residue, rng)
        second = make_case(pair, family, 1, widths, target, carrier, residue, rng)
        yield {"pair": pair, "family": family, "base": base, "depth": depth, "residue": residue,
               "certificate": f"({carrier}) % {MODULUS} = {residue}", "first": first, "second": second}


def route_of(case):
    return tuple(next(arm for arm, route in enumerate(case["routes"]) if value in route["residues"])
                 for value in range(MODULUS))


def compatibility(first, arm, observed):
    if observed is None:
        return (True,) * MODULUS
    return tuple((route == arm) == observed for route in route_of(first))


def world_weights(second, probabilities, worlds):
    out = [0.] * len(worlds)
    for arm, value in zip(route_of(second), probabilities, strict=True):
        key = tuple(index == arm for index in range(3))
        out[worlds.index(key)] += value
    return out
