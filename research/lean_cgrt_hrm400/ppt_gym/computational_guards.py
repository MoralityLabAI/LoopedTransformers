"""Paired existential witness decisions with executable Lean side conditions."""

import random
import re
from dataclasses import dataclass

from .data import digest
SCHEMA = "ppt.native-computational-guards.v1"
FAMILIES = ("BPD.geom", "MathCan.maskAt")
PREFIX = ("apply Exists.intro", "constructor")
OPEN = ("apply h0", "apply h1")
COMPLETION = (("apply h2", "exact h4"), ("apply h3", "exact h5"))


@dataclass(frozen=True)
class Equation:
    family: str
    expression: str
    value: int

    def claim(self, right=None):
        return f"{self.expression} = {self.value if right is None else right}"


def geom(base, steps):
    value = 0
    for _ in range(steps):
        value = 1 + base * value
    return value


def mask_at(steps):
    value = 1
    for _ in range(steps):
        value = ((2 * value) % 64) ^ (3 if (value // 32) % 2 else 0)
    return value


def equation(family, rng, partition):
    modulus = rng.choice((7, 11, 13, 19))
    held = partition == "development"
    if family == FAMILIES[0]:
        base = rng.randrange(2, 9)
        steps = rng.randrange(9, 14) if held else rng.randrange(2, 9)
        return Equation(family, f"BPD.OptimalBasis.geom {base} {steps} % {modulus}", geom(base, steps) % modulus)
    steps = rng.randrange(32, 63) if held else rng.randrange(1, 32)
    return Equation(family, f"MathCan.FiniteHorizonGap.maskAt {steps} % {modulus}", mask_at(steps) % modulus)


def cases(partition, pairs):
    seen = set()
    for index in range(pairs):
        family = FAMILIES[index % len(FAMILIES)]
        guards = 2 + (index // len(FAMILIES) % 2)
        rng = random.Random(int(digest([SCHEMA, partition, index])[:16], 16))
        right = equation(family, rng, partition)
        wrong = equation(family, rng, partition)
        while wrong.value == right.value:
            wrong = equation(family, rng, partition)
        number = rng.sample(range(2, 40), 2)
        warm = [[equation(family, rng, partition).claim() for _ in range(guards - 1)] for _ in range(2)]
        tails = [right.claim(), wrong.claim(right.value)]
        if rng.getrandbits(1):
            tails.reverse()
        pair = digest([SCHEMA, family, warm, sorted(tails), number])
        if pair in seen:
            raise ValueError("duplicate computational guard pair")
        seen.add(pair)
        for world in range(2):
            claims = tails if world == 0 else tails[::-1]
            winner = claims.index(right.claim())
            binders = ["(P Q R : Nat → Prop)"]
            for side in range(2):
                leave = " → ".join(f"({claim})" for claim in (*warm[side], claims[side]))
                binders.append(f"(h{side} : {leave} → P {number[side]})")
            binders.extend(f"(h{2 + i} : R {n} → Q {n})" for i, n in enumerate(number))
            binders.extend(f"(h{4 + i} : R {n})" for i, n in enumerate(number))
            statement = f"∀ {' '.join(binders)}, ∃ n : Nat, P n ∧ Q n"
            script = (*PREFIX, OPEN[winner], *(("decide",) * guards), *COMPLETION[winner])
            yield {"module": digest([pair, world]), "pair": pair, "world": world, "family": family,
                   "statement": statement, "guards": guards, "test_claims": claims, "kernel_winner": winner,
                   "reference": script, "partition": partition}


def menu(public):
    if not public["goals"]:
        return ()
    goal = public["goals"][0]
    target = goal["target"]
    if "and" in target:
        return (PREFIX[1],)
    raw = target.get("raw", "")
    if raw.startswith("∃ "):
        return (PREFIX[0],)
    if raw.startswith("P "):
        return OPEN
    if raw.startswith("Q "):
        return ("apply h2", "apply h3")
    if raw.startswith("R "):
        return ("exact h4", "exact h5")
    if " = " in raw and re.search(r"(?:geom|maskAt)", raw):
        return ("decide",)
    raise ValueError(f"an unregistered native guard state appeared: {target}")
