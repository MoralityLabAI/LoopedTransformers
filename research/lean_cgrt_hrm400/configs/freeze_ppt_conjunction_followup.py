"""Separate structural calibration pairs from an unopened conjunction confirmation."""
import argparse
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import random

from configs.freeze_ppt_conjunction_pilot import branch_atoms, pair_worlds, shapes
from configs.run_ppt_conjunction_pilot import ARMS
from ppt_gym.data import canonical, digest, sha256_file


SCHEMA = "ppt.conjunction-followup.v1"
DOMAIN_SEED = 732709
CALIBRATION_WORLD_SEED = 912419
CONFIRMATION_WORLD_SEED = 954661
CALIBRATION_PAIRS = 96
CONFIRMATION_PAIRS = 480
INTERVENTIONS = ("ablate-interaction", "control-common-shift")


def domain():
    return [pair for arity in (2, 3, 4) for pair in combinations(shapes((arity,)), 2)
            if 1 + sum(branch_atoms(shape) for shape in pair) <= 48]


def encode_worlds(pairs, split, seed):
    rng = random.Random(seed)
    for i, branches in enumerate(pairs):
        cluster = digest(["asymmetric-balanced-Horn-v1", branches])
        yield from pair_worlds(branches, cluster, split, i, rng)


def freeze(pilot, registry, calibration):
    pilot_sha = sha256_file(pilot)
    admissible = domain()
    selected = random.Random(DOMAIN_SEED).sample(admissible, CALIBRATION_PAIRS + CONFIRMATION_PAIRS)
    calibration_pairs = selected[:CALIBRATION_PAIRS]
    confirmation_pairs = selected[CALIBRATION_PAIRS:]
    payload = {
        "schema": SCHEMA, "scope": "fixed seed-103 trained policies on new asymmetric Horn layouts",
        "pilot_sha256": pilot_sha, "domain_seed": DOMAIN_SEED, "domain_size": len(admissible),
        "arms": list(ARMS), "root_interventions": INTERVENTIONS, "minimum_effect": 0.05, "alpha": 0.05,
        "planned_comparisons": len(ARMS), "unit": "one two-world pair of distinct same-arity branch layouts",
        "primary": "candidate versus each training control on both-world certified closure",
        "causal": "matched first-action shift versus first-action removal on both-world certified closure",
        "calibration": {"seed": CALIBRATION_WORLD_SEED, "pairs": calibration_pairs},
        "confirmation": {"seed": CONFIRMATION_WORLD_SEED, "pairs": confirmation_pairs},
        "activation": "calibration must exceed control and sham floors with adequate prospective resolution",
    }
    data = {"schema": SCHEMA, "registry_digest": digest(payload), "scope": "calibration",
            "cases": [asdict(case) for case in encode_worlds(calibration_pairs, "calibration", CALIBRATION_WORLD_SEED)]}
    for path, document in ((registry, payload), (calibration, data)):
        with path.open("x", encoding="utf-8", newline="\n") as out:
            out.write(canonical(document) + "\n")
    print(f"froze {len(calibration_pairs)} calibration and {len(confirmation_pairs)} confirmation pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("calibration", type=Path)
    args = parser.parse_args()
    freeze(args.pilot, args.registry, args.calibration)


if __name__ == "__main__":
    main()
