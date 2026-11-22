"""Freeze equality pairs where accumulated support can favor a late proof."""

import argparse
from dataclasses import asdict
from itertools import combinations_with_replacement, product
import json
import math
from pathlib import Path
import random

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SCHEMA, SearchCase, load
from ppt_gym.rewrite import BPD_REWRITE_FAMILY


GENERATOR = "ppt.rewrite-deadline-layout.v1"
SEED = 408571
TRAIN_PAIRS_PER_STRATUM = 16
DEV_PAIRS_PER_STRATUM = 12
RETENTION = 1 / (1 + math.exp(-3.4))
MINIMUM_MARGIN = 0.00005
START, TARGET = "start", "target"
CHAINS = [tuple(f"{side}{depth}" for depth in range(10)) for side in ("left", "right")]


def edges(layout, winner):
    short, long = layout
    root = [(START, chain[0]) for chain in CHAINS]
    rules = [*root]
    for side, chain in enumerate(CHAINS):
        early = side == winner
        rules.extend(zip(chain, chain[1:]))
        rules.append((chain[6 if early else 7], TARGET))
        rules.extend((chain[depth], f"leaf{side}_{index}") for index, depth in enumerate(short if early else long))
    return rules


def diffusion_margin(rules, winner):
    neighbors = {}
    for left, right in rules:
        neighbors.setdefault(left, []).append(right)
        neighbors.setdefault(right, []).append(left)
    mass = dict.fromkeys(neighbors, 0.0)
    mass[TARGET] = 1.0
    for _ in range(8):
        mass = {node: float(node == TARGET) + RETENTION * sum(mass[j] for j in connected) / len(connected)
                for node, connected in neighbors.items()}
    return mass[CHAINS[winner][0]] - mass[CHAINS[1 - winner][0]]


def layouts():
    aligned, misleading = [], []
    for count in (2, 3):
        placements = tuple(combinations_with_replacement((3, 4, 5, 6, 8, 9), count))
        for layout in product(placements, repeat=2):
            margin = diffusion_margin(edges(layout, winner=0), winner=0)
            if margin >= MINIMUM_MARGIN:
                aligned.append(layout)
            elif margin <= -MINIMUM_MARGIN:
                misleading.append(layout)
    return {"aligned": aligned, "misleading": misleading}


def pair(layout, split, number, rng):
    cluster = digest([GENERATOR, layout])
    graph = edges(layout, winner=0)
    roles = sorted({node for rule in graph for node in rule})
    atoms = [f"x{i}" for i in range(len(roles))]
    rng.shuffle(atoms)
    names = dict(zip(roles, atoms, strict=True))
    order = list(range(2, len(graph)))
    rng.shuffle(order)
    order = [0, 1, *order]
    for winner in (0, 1):
        rules = edges(layout, winner)
        variables = f"({' '.join(sorted(atoms))} : Nat)"
        equalities = [f"(h{i} : {names[rules[j][0]]} = {names[rules[j][1]]})"
                      for i, j in enumerate(order)]
        yield SearchCase(
            "loopt_horn", split, BPD_REWRITE_FAMILY, digest([cluster, winner]), cluster,
            f"rewrite_deadline_{number}_{winner}", " ".join([variables, *equalities]),
            f"{names[START]} = {names[TARGET]}",
        )


def freeze(source, output):
    original = json.loads(source.read_bytes())
    rng = random.Random(SEED)
    splits = {"train": {}, "dev": {}}
    cases, index = [], 0
    for stratum, shapes in layouts().items():
        selected = rng.sample(shapes, TRAIN_PAIRS_PER_STRATUM + DEV_PAIRS_PER_STRATUM)
        for split, group in (("train", selected[:TRAIN_PAIRS_PER_STRATUM]),
                             ("dev", selected[TRAIN_PAIRS_PER_STRATUM:])):
            splits[split][stratum] = group
            for layout in group:
                cases.extend(pair(layout, split, index, rng))
                index += 1
    plan = {
        "schema": SCHEMA, "seed": SEED,
        "source_sha256": {**original["source_sha256"], "bpd_mixed_training": sha256_file(source),
                          "rewrite_deadline_generator": sha256_file(Path(__file__))},
        "registration": {
            "scope": "eight-check equality deadline development with matched opposite root choices",
            "layout_namespace": GENERATOR, "layouts": splits,
            "fixed_retention": RETENTION, "minimum_absolute_margin": MINIMUM_MARGIN,
            "primary": "native certification in both worlds of a paired layout",
            "early_path_checks": 8, "late_path_checks": 9, "confirmation_scored": False,
        },
        "cases": [asdict(case) for case in [*load(source)["train"], *cases]],
    }
    with output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(canonical(plan) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze(args.source, args.output)


if __name__ == "__main__":
    main()
