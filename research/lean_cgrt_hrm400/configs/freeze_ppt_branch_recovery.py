"""Freeze contrasting Lean worlds for one early checkpoint decision."""

import argparse
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import random

from configs.freeze_ppt_conjunction_pilot import branch_atoms, pair_worlds, shapes
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import RECOVERY_FAMILY, SCHEMA, load


SEED = 151
LAYOUT_SEED = 571912
ARMS = ("hybrid_recovery_rl", "hybrid_calibration", "hybrid_imitation", "trm_recovery_rl", "trm_calibration")
PROOF_CHECKS = 7


def layouts(arities):
    return [shape for shape in shapes(arities) if PROOF_CHECKS - 1 - shape[0] - sum(shape[2]) >= 2]


def valid_pairs(shapes_):
    return [pair for pair in combinations(shapes_, 2) if 1 + sum(branch_atoms(s, PROOF_CHECKS) for s in pair) <= 48]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_training", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = json.loads(args.mixed_training.read_bytes())
    rng = random.Random(LAYOUT_SEED)
    seen, wider = layouts((2, 3)), layouts((4,))
    base, shifted = valid_pairs(seen), valid_pairs(wider)
    rng.shuffle(base)
    rng.shuffle(shifted)
    train, dev = base[:32], [*base[32:44], *shifted[:12]]
    exclusions = set(train + dev)
    reserved = rng.sample([pair for pair in valid_pairs(seen + wider) if pair not in exclusions], 192)
    cases = []
    for split, pairs in (("train", train), ("dev", dev)):
        for i, pair in enumerate(pairs):
            cluster = digest(["early-branch-recovery-v1", pair])
            cases.extend(pair_worlds(pair, cluster, split, i, rng, proof_checks=PROOF_CHECKS,
                                     family=RECOVERY_FAMILY, name="recovery"))
    cases.extend(case for case in load(args.mixed_training)["train"] if case.source != "loopt_horn")
    doc = {
        "schema": SCHEMA, "seed": LAYOUT_SEED,
        "source_sha256": {**original["source_sha256"], "mixed_source": sha256_file(args.mixed_training),
                         "recovery_generator": sha256_file(Path(__file__)),
                         "paired_horn_generator": sha256_file(Path(__file__).with_name("freeze_ppt_conjunction_pilot.py"))},
        "registration": {
            "scope": "development-only Lean early branch recovery",
            "model_seeds": [SEED], "controls": ARMS[1:], "tactic_checks": 8, "updates": 51, "rollouts": 2, "lr": .001,
            "recovery_rollouts_per_world": 1, "initialization_cohort_sha256": (
                original["registration"]["initialization_cohort_sha256"]),
            "unit": "two worlds sharing one attempted tactic and swapping the productive continuation",
            "primary": "both worlds independently certified within eight checks",
            "minimum_meaningful_effect": 0.05,
            "structural_pairs": {"train": train, "dev": dev}, "reserved_pairs": reserved,
            "confirmation_scored": False,
        },
        "cases": [asdict(case) for case in cases],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(doc) + "\n")


if __name__ == "__main__":
    main()
