"""Freeze two-world proof deadlines whose paired choice is an interior tactic."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

from configs.freeze_ppt_deadline import DEADLINE, MAX_CHECKS, domain, make_worlds
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import CONTINUATION_DEADLINE_FAMILY, SCHEMA, load


SEED = 151
LAYOUT_SEED = 338297
ARMS = ("continuation_rl", "continuation_calibration", "deadline_imitation", "return_weighted_imitation",
        "all_proofs_imitation", "trm_rl", "trm_calibration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_training", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = json.loads(args.mixed_training.read_bytes())
    rng = random.Random(LAYOUT_SEED)
    binary, ternary = domain(2, prefix=1), domain(3, prefix=1)
    rng.shuffle(binary)
    rng.shuffle(ternary)
    train = binary[:24]
    development = [*binary[24:36], *ternary[:12]]
    reserved_pool = [*binary[36:], *ternary[12:]]
    reserved = rng.sample(reserved_pool, 480)
    cases = [case for split, pool in (("train", train), ("dev", development))
             for number, branches in enumerate(pool)
             for case in make_worlds(branches, split, number, rng, prefix=1, family=CONTINUATION_DEADLINE_FAMILY)]
    cases.extend(case for case in load(args.mixed_training)["train"] if case.source != "loopt_horn")
    doc = {
        "schema": SCHEMA, "seed": LAYOUT_SEED,
        "source_sha256": {**original["source_sha256"], "mixed_source": sha256_file(args.mixed_training),
                         "continuation_deadline_generator": sha256_file(Path(__file__)),
                         "deadline_layout_generator": sha256_file(Path(__file__).with_name("freeze_ppt_deadline.py"))},
        "registration": {
            "scope": "development-only interior-choice proof deadlines",
            "model_seeds": [SEED], "controls": ARMS[1:], "tactic_checks": MAX_CHECKS, "proof_deadline": DEADLINE,
            "proposal_attempt": 2, "prefer_direct_fact": True, "updates": 43, "rollouts": 2, "lr": 0.001,
            "initialization_cohort_sha256": original["registration"]["initialization_cohort_sha256"],
            "unit": "two worlds with the same first tactic and opposite on-time second tactics",
            "primary": "both worlds independently certified within six checks",
            "minimum_meaningful_effect": 0.05,
            "structural_pairs": {"train": train, "dev": development}, "reserved_pairs": reserved,
            "confirmation_scored": False,
        },
        "cases": [asdict(case) for case in cases],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(doc) + "\n")


if __name__ == "__main__":
    main()
