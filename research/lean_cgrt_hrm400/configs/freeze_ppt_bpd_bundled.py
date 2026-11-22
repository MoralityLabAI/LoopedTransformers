"""Freeze matched bundle openings alongside the audited mixed rewrite training sources."""

import argparse
from dataclasses import asdict
from itertools import combinations_with_replacement
import json
from pathlib import Path
import random

from configs.check_ppt_bpd_bundled_opening import SHAPES
from configs.freeze_ppt_bpd_rewrite import ARMS
from ppt_gym.bundled_rewrite import BPD_BUNDLE_FAMILY
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SCHEMA, SearchCase, load


SEED = 183456873
TRAIN_PAIRS = 32
DEV_PAIRS = 24
GENERATOR = "ppt.bpd-bundled-opening.v1"


def paired_cases(shape, ordinal, split, rng):
    start, target = "start", "target"
    roles = [start, target]
    bundles = []
    for side in (0, 1):
        spine = [start, *(f"{side}_{depth}" for depth in range(1, 7))]
        dead = f"{side}_dead"
        roles.extend((*spine[1:], dead))
        edges = [*zip(spine, spine[1:]), (spine[-1], target)]
        for j, depth in enumerate(shape):
            spur = f"{side}_spur{j}"
            roles.append(spur)
            edges.append((spine[depth], spur))
        rng.shuffle(edges)
        bundles.append((dead, edges))
    names = {role: f"v{rng.randrange(1 << 44):011x}" for role in roles}
    if len(set(names.values())) != len(roles):
        raise ValueError("fresh equality bundle atoms collided")
    cluster = digest([GENERATOR, shape])
    variables = f"({' '.join(sorted(names.values()))} : Nat)"
    for winner in (0, 1):
        hypotheses = []
        for side, (dead, edges) in enumerate(bundles):
            premises = [f"{names[a]} = {names[dead if b == target and side != winner else b]}" for a, b in edges]
            nested = premises[-1]
            for premise in reversed(premises[:-1]):
                nested = f"({premise}) ∧ ({nested})"
            hypotheses.append(f"(h{side} : {nested})")
        yield SearchCase("loopt_bundle", split, BPD_BUNDLE_FAMILY, digest([cluster, winner]),
                         cluster, f"bpd_bundle_{ordinal}_{winner}", " ".join([variables, *hypotheses]),
                         f"{names[start]} = {names[target]}")


def freeze(mixed, destination):
    original = json.loads(mixed.read_bytes())
    initial_training = load(mixed)["train"]
    rng = random.Random(SEED)
    candidates = [shape for count in (2, 3, 4)
                  for shape in combinations_with_replacement(range(1, 7), count) if shape not in SHAPES]
    shapes = rng.sample(candidates, TRAIN_PAIRS + DEV_PAIRS)
    cases, layouts = [], {}
    for ordinal, shape in enumerate(shapes):
        split = "train" if ordinal < TRAIN_PAIRS else "dev"
        cluster = digest([GENERATOR, shape])
        layouts[cluster] = {"split": split, "shape": shape}
        cases.extend(paired_cases(shape, ordinal, split, rng))
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(canonical({
            "schema": SCHEMA, "seed": SEED,
            "source_sha256": {**original["source_sha256"], "mixed_rewrite_cohort": sha256_file(mixed),
                              "paired_bundle_generator": sha256_file(Path(__file__)),
                              "bundle_layout_source": sha256_file(Path(__file__).with_name("check_ppt_bpd_bundled_opening.py"))},
            "registration": {
                "scope": "paired bundle-opening development with a shared native continuation",
                "arms": ARMS, "model_seeds": [151], "tactic_checks": 8, "cpu_seconds": 6,
                "layouts": layouts, "reference_certificates": 2 * DEV_PAIRS,
                "bundle_continuation": True, "rewrite_exploration": True,
                "primary": "independent certificates in both worlds of each paired opening layout",
                "minimum_meaningful_effect": 0.05, "confirmation_scored": False,
                "initialization_cohort_sha256": original["registration"]["initialization_cohort_sha256"],
                "updates": (len(initial_training) + 2 * TRAIN_PAIRS + 1) // 2, "rollouts": 2, "lr": 0.001,
            },
            "cases": [asdict(case) for case in [*initial_training, *cases]],
        }) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze(args.mixed, args.output)


if __name__ == "__main__":
    main()
