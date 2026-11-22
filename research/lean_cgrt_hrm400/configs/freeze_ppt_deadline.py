"""Freeze paired Lean layouts in which both first tactics prove at different costs."""
import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path
import random

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import DEADLINE_FAMILY, SCHEMA, SearchCase, load


SEED = 151
ALLOCATION_SEED = 734901
DEADLINE = 6
MAX_CHECKS = 8
ARMS = ("arrival_rl", "readout_calibration", "deadline_imitation", "all_proofs_imitation",
        "terminal_readout_rl", "trm_rl", "trm_calibration")


def layouts(arity, prefix=0):
    candidates = []
    for stem in (1, 2):
        fast_depth = DEADLINE - prefix - stem - 1 - arity
        slow_depth = MAX_CHECKS - prefix - stem - 1 - arity
        if fast_depth < 0:
            continue
        fast = (v for v in product(range(fast_depth + 1), repeat=arity) if sum(v) == fast_depth)
        slow = [v for v in product(range(slow_depth + 1), repeat=arity) if sum(v) == slow_depth]
        paired_slots = sorted({tuple(sorted(zip(a, b, strict=True))) for a in fast for b in slow})
        for slots, tail, decoys in product(paired_slots, range(2 + prefix), range(1, 4 + prefix)):
            candidates.append((stem, slots, tail, decoys))
    return candidates


def atoms_in(layout):
    stem, slots, tail, decoys = layout
    return stem + sum(1 + max(fast, slow) + tail for fast, slow in slots) + len(slots) * decoys


def domain(arity, prefix=0):
    return [branches for branches in combinations(layouts(arity, prefix), 2)
            if sum(map(atoms_in, branches)) + 1 + prefix <= 48]


def make_worlds(branches, split, number, rng, *, prefix=0, family=DEADLINE_FAMILY):
    identifier = "continuation-deadline-v1" if prefix else "proof-deadline-v1"
    cluster = digest([identifier, branches])
    names = [f"P{i}" for i in range(1 + prefix + sum(map(atoms_in, branches)))]
    order = names.copy()
    rng.shuffle(order)
    symbols = iter(order)
    goal = next(symbols)
    rules, alternatives = [], []
    frontier = goal
    for _ in range(prefix):
        next_goal = next(symbols)
        rules.append(f"{next_goal} → {frontier}")
        frontier = next_goal
    for stem, slots, tail, decoys in branches:
        head = frontier
        for _ in range(stem):
            child = next(symbols)
            rules.append(f"{child} → {head}")
            head = child
        fronts = []
        choices = [[], []]
        for depths in slots:
            chain = [next(symbols) for _ in range(1 + max(depths) + tail)]
            fronts.append(chain[0])
            rules.extend(f"{child} → {parent}" for parent, child in zip(chain, chain[1:]))
            for world, depth in enumerate(depths):
                choices[world].append(chain[depth])
        rules.append(" → ".join([*fronts, head]))
        for _ in range(decoys):
            decoy_fronts = [next(symbols) for _ in slots]
            rules.append(" → ".join([*decoy_fronts, head]))
        alternatives.append(choices)
    rng.shuffle(rules)
    variables = "(" + " ".join(sorted(names)) + " : Prop)"
    assumptions = [f"(r{i} : {rule})" for i, rule in enumerate(rules)]
    for world in (0, 1):
        proven = sorted([*alternatives[0][world], *alternatives[1][1 - world]])
        facts = [f"(h{i} : {fact})" for i, fact in enumerate(proven)]
        binders = " ".join([variables, *facts, *assumptions])
        yield SearchCase("loopt_horn", split, family, digest([cluster, world]), cluster,
                         f"{'continuation' if prefix else 'proof'}_deadline_{split}_{number}_{world}", binders, goal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_training", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = json.loads(args.mixed_training.read_bytes())
    rng = random.Random(ALLOCATION_SEED)
    binary, ternary = domain(2), domain(3)
    rng.shuffle(binary)
    rng.shuffle(ternary)
    train, dev = binary[:32], [*binary[32:44], *ternary[:12]]
    reserved = [*binary[44:284], *ternary[12:252]]
    if len(reserved) != 480:
        raise ValueError("could not reserve 480 unseen structural pairs")
    cases = [case for split, pool in (("train", train), ("dev", dev))
             for i, layouts_ in enumerate(pool) for case in make_worlds(layouts_, split, i, rng)]
    cases.extend(case for case in load(args.mixed_training)["train"] if case.source != "loopt_horn")
    if not {"bpd_lg02", "evalmath_lg01"} <= {case.source for case in cases}:
        raise ValueError("mixed native roots are missing from deadline training")
    doc = {
        "schema": SCHEMA, "seed": ALLOCATION_SEED,
        "source_sha256": {**original["source_sha256"], "deadline_generator": sha256_file(Path(__file__)),
                         "mixed_source": sha256_file(args.mixed_training)},
        "registration": {
            "scope": "development on a new proof-cost layout family", "model_seeds": [SEED], "controls": ARMS[1:],
            "tactic_checks": MAX_CHECKS, "proof_deadline": DEADLINE, "updates": 51, "rollouts": 2, "lr": 0.001,
            "initialization_cohort_sha256": original["registration"]["initialization_cohort_sha256"],
            "unit": "two worlds with the same rules and swapped minimum proof costs at their two initial tactics",
            "primary": "both worlds independently certified within six checks",
            "reference": "independent native certificates forced through each first tactic in six and eight checks",
            "all_arms_keep_late_certificates": True, "minimum_meaningful_effect": 0.05,
            "structural_pairs": {"train": train, "dev": dev}, "reserved_pairs": reserved,
            "confirmation_scored": False,
        },
        "cases": [asdict(case) for case in cases],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(doc) + "\n")


if __name__ == "__main__":
    main()
