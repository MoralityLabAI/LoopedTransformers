"""Freeze recovery training and development pairs with balanced pending evidence."""

import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path
import random

from configs.freeze_ppt_branch_recovery import ARMS, SEED
from configs.freeze_ppt_pending_recovery import TREES, proof_tree
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import RECOVERY_FAMILY, SCHEMA, SearchCase, load


GENERATOR = "ppt.pending-alternative-recovery.v1"
LAYOUT_SEED = 417229
TRAIN_PAIRS = 32
MIXED_ARITY_PAIRS = 26


def shapes(arity):
    for alternatives in range(2, 6):
        for extra in range(4 - arity):
            # After the shared first tactic, the active and pending proofs cost six checks.
            for active in TREES[5 - arity - extra]:
                slots = (0,) * (arity - 1) + (extra,)
                yield arity, alternatives, slots, active


def branch_atoms(shape):
    arity, alternatives, slots, _ = shape
    return 6 - arity - sum(slots) + alternatives * (arity + sum(slots))


def branches(layout, split, index, rng):
    cluster = digest([GENERATOR, layout])
    atoms = [f"P{i}" for i in range(1 + sum(map(branch_atoms, layout)))]
    if len(atoms) > 48:
        raise ValueError("pending alternatives exceed the graph input cap")
    order = atoms.copy()
    rng.shuffle(order)
    names = iter(order)
    goal = next(names)
    rules, stable, proof_facts = [], [], []
    for arity, alternatives, slots, active in layout:
        first, first_facts = proof_tree(active, names, rules)
        stable.extend(first_facts)
        pending = next(names)
        rules.append(f"{first} → {pending} → {goal}")
        leaves = []
        for _ in range(alternatives):
            obligations, available = [], []
            for depth in slots:
                target, facts = proof_tree(((),) if depth else (), names, rules)
                obligations.append(target)
                available.extend(facts)
            rules.append(" → ".join([*obligations, pending]))
            leaves.append(available)
        complete = leaves[rng.randrange(alternatives)]
        first_alternative, second_alternative = rng.sample(leaves, 2)
        mask = set(rng.sample(range(arity), rng.randrange(1, arity)))
        split_facts = [(first_alternative if slot in mask else second_alternative)[slot] for slot in range(arity)]
        proof_facts.append((complete, split_facts))
    if next(names, None) is not None:
        raise ValueError("unconsumed atoms in the pending proof template")
    rng.shuffle(rules)
    signatures = [f"({' '.join(atoms)} : Prop)", *[f"(r{i} : {rule})" for i, rule in enumerate(rules)]]
    for side in (0, 1):
        available = [*stable, *proof_facts[0][side], *proof_facts[1][1 - side]]
        facts = [f"(h{i} : {fact})" for i, fact in enumerate(available)]
        yield SearchCase("loopt_horn", split, RECOVERY_FAMILY, digest([cluster, side]), cluster,
                         f"pending_alternatives_{split}_{index}_{side}", " ".join([*signatures, *facts]), goal)


def freeze(source, output):
    original = json.loads(source.read_bytes())
    rng = random.Random(LAYOUT_SEED)
    two, three = tuple(shapes(2)), tuple(shapes(3))
    train = rng.sample(list(combinations(two, 2)), TRAIN_PAIRS)
    dev = [*rng.sample(list(product(two, three)), MIXED_ARITY_PAIRS),
           *list(combinations(three, 2))]
    if not dev or set(train) & set(dev):
        raise ValueError("training and development pending layouts overlap")
    cases = [case for split, layouts in (("train", train), ("dev", dev))
             for index, layout in enumerate(layouts) for case in branches(layout, split, index, rng)]
    mixed = load(source)["train"]
    old_pairs = [case for case in mixed if case.family == RECOVERY_FAMILY]
    prior_clusters = {row.cluster for row in mixed}
    if any(case.cluster in prior_clusters for case in cases):
        raise ValueError("the new layouts collide with earlier training")
    # Recovery pairs are one update each. The other corpora are paired into two-case updates.
    updates = TRAIN_PAIRS + len(old_pairs) // 2 + (len(mixed) - len(old_pairs) + 1) // 2
    result = {
        "schema": SCHEMA,
        "seed": LAYOUT_SEED,
        "registration": {
            "scope": "development on balanced alternative-rule support in a pending obligation",
            "initialization_cohort_sha256": original["registration"]["initialization_cohort_sha256"],
            "model_seeds": [SEED], "controls": ARMS[1:],
            "tactic_checks": 8, "updates": updates, "rollouts": 2, "lr": .001,
            "recovery_rollouts_per_world": 1, "confirmation_scored": False,
            "pending_arrival_ablation": "hybrid_unrouted_rl",
            "unit": "two worlds sharing public root rules and exchanging complete and fragmented pending support",
            "primary": "both worlds independently certified within eight tactic checks",
            "minimum_meaningful_effect": .05, "structural_pairs": {"train": train, "dev": dev},
            "counts": {"new_training_pairs": len(train), "development_pairs": len(dev),
                       "mixed_arity_development_pairs": MIXED_ARITY_PAIRS},
        },
        "source_sha256": {**original["source_sha256"], "branch_recovery_source": sha256_file(source),
                          "pending_alias_generator": sha256_file(Path(__file__)),
                          "pending_base_generator": sha256_file(Path(__file__).with_name("freeze_ppt_pending_recovery.py"))},
        "cases": [asdict(case) for case in [*mixed, *cases]],
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze(args.source, args.output)


if __name__ == "__main__":
    main()
