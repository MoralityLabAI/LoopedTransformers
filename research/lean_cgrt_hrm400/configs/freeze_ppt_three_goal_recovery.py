"""Freeze a transfer family with distinct pending-goal failure patterns."""

import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path
import random

from configs.freeze_ppt_pending_recovery import TREES, proof_tree
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import RECOVERY_FAMILY, SCHEMA, SearchCase, load


GENERATOR = "ppt.three-goal-recovery-transfer.v1"
LAYOUT_SEED = 684139
MIXED_ARITY_PAIRS = 48


def shapes(arity):
    for alternatives, extra, mode in product(range(2, 6), range(4 - arity), ("fragmented", "missing")):
        stable = 5 - arity - extra
        for left_size in range(1, stable):
            middle_size = stable - left_size
            left = () if left_size == 1 else TREES[left_size][0]
            middle = () if middle_size == 1 else TREES[middle_size][0]
            slots = (0,) * (arity - 1) + (extra,)
            yield arity, alternatives, slots, left, middle, mode


def atom_count(shape):
    arity, options, slots, _, _, mode = shape
    return 6 - arity - sum(slots) + options * (arity + sum(slots)) + int(mode == "missing")


def worlds(layout, index, rng):
    cluster = digest([GENERATOR, layout])
    atoms = [f"P{i}" for i in range(1 + sum(map(atom_count, layout)))]
    if len(atoms) > 48:
        raise ValueError("three-goal layout exceeds the graph input cap")
    order = atoms.copy()
    rng.shuffle(order)
    names = iter(order)
    goal = next(names)
    rules, proof_facts = [], []

    for arity, alternatives, slots, left, middle, mode in layout:
        first, first_facts = proof_tree(left, names, rules)
        second, second_facts = proof_tree(middle, names, rules)
        pending = next(names)
        rules.append(f"{first} → {second} → {pending} → {goal}")
        options = []
        for _ in range(alternatives):
            obligations, available = [], []
            for depth in slots:
                target, facts = proof_tree(((),) if depth else (), names, rules)
                obligations.append(target)
                available.extend(facts)
            rules.append(" → ".join([*obligations, pending]))
            options.append(available)
        supported = rng.choice(options)
        live = [*first_facts, *second_facts, *supported]
        if mode == "missing":
            # Keep the joint proof viable and replace one fact required by the second goal.
            decoy = next(names)
            rules.append(f"{decoy} → {decoy}")
            dead = [*first_facts, *second_facts[1:], decoy, *supported]
        else:
            first_option, other_option = rng.sample(options, 2)
            split = set(rng.sample(range(arity), rng.randrange(1, arity)))
            fragmented = [(first_option if slot in split else other_option)[slot] for slot in range(arity)]
            dead = [*first_facts, *second_facts, *fragmented]
        proof_facts.append((live, dead))
    if next(names, None) is not None:
        raise ValueError("unused atoms in the three-goal layout")
    rng.shuffle(rules)
    binders = [f"({' '.join(atoms)} : Prop)", *[f"(r{i} : {rule})" for i, rule in enumerate(rules)]]
    for side in (0, 1):
        facts = [*proof_facts[0][side], *proof_facts[1][1 - side]]
        local = [f"(h{i} : {fact})" for i, fact in enumerate(facts)]
        yield SearchCase("loopt_horn", "dev", RECOVERY_FAMILY, digest([cluster, side]), cluster,
                         f"three_goal_recovery_{index}_{side}", " ".join([*binders, *local]), goal)


def freeze(source, destination):
    base = json.loads(source.read_bytes())
    reg = base["registration"]
    rng = random.Random(LAYOUT_SEED)
    two, three = tuple(shapes(2)), tuple(shapes(3))
    mixed = [pair for pair in product(two, three) if pair[0][-1] != pair[1][-1]]
    wide = [pair for pair in combinations(three, 2) if pair[0][-1] != pair[1][-1]]
    layouts = [*rng.sample(mixed, MIXED_ARITY_PAIRS), *wide]
    transfer = [case for index, layout in enumerate(layouts) for case in worlds(layout, index, rng)]
    previous = {raw["lineage"] for raw in base["cases"]}
    if any(case.lineage in previous for case in transfer):
        raise ValueError("transfer retained an earlier theorem lineage")
    plan = {
        "schema": SCHEMA, "seed": LAYOUT_SEED,
        "source_sha256": {**base["source_sha256"], "pending_alias_training": sha256_file(source),
                          "three_goal_generator": sha256_file(Path(__file__))},
        "registration": {
            **reg, "scope": "three-obligation transfer development with a missing second goal or fragmented join",
            "development_layout_namespace": GENERATOR,
            "unit": "two worlds sharing root rules and exchanging the viable three-goal branch",
            "structural_pairs": {"train": reg["structural_pairs"]["train"], "dev": layouts},
            "counts": {**reg["counts"], "development_pairs": len(layouts),
                       "mixed_arity_development_pairs": MIXED_ARITY_PAIRS},
        },
        "cases": [asdict(case) for case in [*load(source)["train"], *transfer]],
    }
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(canonical(plan) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    freeze(args.source, args.destination)


if __name__ == "__main__":
    main()
