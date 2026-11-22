"""Freeze matched Horn worlds with equal linear evidence and different rule support."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import random

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SCHEMA, SearchCase, load


SEED = 469141
MODEL_SEED = 103


def shapes(arities):
    for arity in arities:
        for options in (2, 3, 4):
            for slots in itertools.combinations_with_replacement(range(7 - arity), arity):
                extras = sum(slots)
                stem = 7 - arity - extras
                atoms = 1 + 2 * (stem + options * (arity + extras))
                if stem < 1 or atoms > 48:
                    continue
                for split in itertools.combinations_with_replacement(range(1, arity), 2):
                    if sum(split) == arity:
                        yield arity, options, slots, split


def worlds(shape, split, number, rng):
    group = digest(["equal-linear-evidence-v1", shape])
    return pair_worlds((shape, shape), group, split, number, rng)


def branch_atoms(shape, proof_checks=8):
    arity, options, slots, _ = shape
    stem = proof_checks - 1 - arity - sum(slots)
    return stem + options * (arity + sum(slots))


def pair_worlds(branches, group, split, number, rng, *, proof_checks=8, family="first-choice-conjunction",
                name="conjunction"):
    atoms = 1 + sum(branch_atoms(shape, proof_checks) for shape in branches)
    names = [f"P{i}" for i in range(atoms)]
    order = names.copy()
    rng.shuffle(order)
    symbols = iter(order)
    goal = next(symbols)
    rules, both = [], []

    for arity, options, slots, fragmented in branches:
        stem = proof_checks - 1 - arity - sum(slots)
        head = goal
        for _ in range(stem):
            premise = next(symbols)
            rules.append(f"{premise} → {head}")
            head = premise
        ends = []
        for _ in range(options):
            tops = [next(symbols) for _ in slots]
            rules.append(" → ".join([*tops, head]))
            end = []
            for top, depth in zip(tops, slots, strict=True):
                for _ in range(depth):
                    premise = next(symbols)
                    rules.append(f"{premise} → {top}")
                    top = premise
                end.append(top)
            ends.append(end)
        complete = rng.randrange(options)
        partial = rng.sample(range(options), 2)
        assigned = [partial[0]] * fragmented[0] + [partial[1]] * fragmented[1]
        rng.shuffle(assigned)
        both.append((ends[complete], [ends[index][slot] for slot, index in enumerate(assigned)]))
    rng.shuffle(rules)
    variables = "(" + " ".join(sorted(names)) + " : Prop)"
    declarations = [f"(r{i} : {rule})" for i, rule in enumerate(rules)]
    for side in (0, 1):
        live = sorted([*both[0][side], *both[1][1 - side]])
        binders = " ".join([variables, *[f"(h{i} : {leaf})" for i, leaf in enumerate(live)], *declarations])
        yield SearchCase("loopt_horn", split, family, digest([group, side]), group,
                         f"{name}_{split}_{number}_{side}", binders, goal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_bytes())
    rng = random.Random(SEED)
    seen_arity = list(shapes((2, 3)))
    rng.shuffle(seen_arity)
    unseen_arity = rng.sample(list(shapes((4,))), 8)
    pools = {"train": seen_arity[:24], "dev": [*seen_arity[24:40], *unseen_arity]}
    data = [case for split, pool in pools.items() for i, shape in enumerate(pool)
            for case in worlds(shape, split, i, rng)]
    data.extend(case for case in load(args.source)["train"] if case.source == "bpd_lg02")
    doc = {
        "schema": SCHEMA, "seed": SEED,
        "source_sha256": {**source["source_sha256"], "conjunction_generator": sha256_file(Path(__file__)),
                         "bpd_train_source": sha256_file(args.source)},
        "registration": {
            "scope": "fresh development mechanism pilot", "tactic_checks": 8,
            "unit": "one two-world pair with equal linear evidence and opposite provable branches",
            "model_seeds": [MODEL_SEED], "dev_alias_clusters": len(pools["dev"]),
            "updates": 48, "rollouts": 2, "lr": 0.001,
            "shapes_by_cluster": {digest(["equal-linear-evidence-v1", shape]): shape
                                  for pool in pools.values() for shape in pool},
            "primary": "both worlds independently certified against matched head-only and TRM",
            "controls": ["hybrid_mean", "hybrid_frozen", "hybrid_imitation", "trm", "trm_frozen"],
            "activation": "development only; old branched and ARG-03-N holdouts remain sealed",
        },
        "cases": [asdict(case) for case in data],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(doc) + "\n")


if __name__ == "__main__":
    main()
