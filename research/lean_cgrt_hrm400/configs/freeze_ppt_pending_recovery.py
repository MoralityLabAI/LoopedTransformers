"""Freeze recovery pairs whose competing branches have pending Lean goals."""

import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path
import random

from configs.freeze_ppt_branch_recovery import ARMS
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import RECOVERY_FAMILY, SearchCase


SCHEMA = "ppt.pending-recovery-transfer.v1"
LAYOUT_SEED = 867217
SEEDS = (103, 151, 197)
CHECKS = 8
PAIR_COUNT = 64
EDITS = ("pristine", "ablate-history", "common-shift", "signed-margin-sham")

# Each tree node costs exactly one Lean check, including the leaf facts.
LEAF = ()
CHAIN = (LEAF,)
TREES = {
    2: (CHAIN,),
    3: ((CHAIN,), (LEAF, LEAF)),
    4: (((CHAIN,),), ((LEAF, LEAF),), (LEAF, CHAIN), (LEAF, LEAF, LEAF)),
}


def proof_tree(tree, names, rules):
    target = next(names)
    leaves = []
    premises = []
    for child in tree:
        premise, facts = proof_tree(child, names, rules)
        premises.append(premise)
        leaves.extend(facts)
    if premises:
        rules.append(" → ".join([*premises, target]))
    else:
        leaves.append(target)
    return target, leaves


def pair(layout, number, rng):
    group = digest([SCHEMA, layout])
    names = [f"P{i}" for i in range(15)]
    shuffled = names.copy()
    rng.shuffle(shuffled)
    symbols = iter(shuffled)
    goal = next(symbols)
    rules, stable_facts, swapped = [], [], []
    for active, pending in layout:
        first, first_facts = proof_tree(active, symbols, rules)
        second, second_facts = proof_tree(pending, symbols, rules)
        rules.append(f"{first} → {second} → {goal}")
        stable_facts.extend(first_facts)
        stable_facts.extend(second_facts[1:])
        swapped.append((second_facts[0], next(symbols)))
    if next(symbols, None) is not None:
        raise ValueError("a branch must spend six checks after the speculative tactic")
    # Keep both decoy atoms in the public graph without adding a path to the goal.
    decoy, other_decoy = [unsupported for _, unsupported in swapped]
    rules.extend((f"{decoy} → {other_decoy}", f"{other_decoy} → {decoy}"))
    rng.shuffle(rules)
    declarations = [f"(r{i} : {rule})" for i, rule in enumerate(rules)]
    variables = "(" + " ".join(names) + " : Prop)"
    for side in (0, 1):
        live = [*stable_facts, swapped[0][side], swapped[1][1 - side]]
        binders = " ".join([variables, *[f"(h{i} : {fact})" for i, fact in enumerate(live)], *declarations])
        yield SearchCase("loopt_horn", "dev", RECOVERY_FAMILY, digest([group, side]), group,
                         f"pending_recovery_{number}_{side}", binders, goal)


def freeze(previous, sessions, output):
    original = json.loads(previous.read_bytes())
    rng = random.Random(LAYOUT_SEED)
    branches = [shape for active_size, pending_size in ((2, 4), (3, 3), (4, 2))
                for shape in product(TREES[active_size], TREES[pending_size])]
    all_layouts = list(combinations(branches, 2))
    layouts = rng.sample(all_layouts, PAIR_COUNT)
    cases = [asdict(case) for number, shape in enumerate(layouts) for case in pair(shape, number, rng)]
    old_roots = {case["lineage"] for case in original["cases"]}
    if any(case["lineage"] in old_roots for case in cases):
        raise ValueError("a transfer proof collides with the original cohort")
    if len(sessions) != len(SEEDS):
        raise ValueError("provide one frozen session directory per registered seed")
    checkpoints = {str(seed): {arm: sha256_file(directory / f"{arm}.sqlite") for arm in ARMS}
                   for seed, directory in zip(SEEDS, sessions, strict=True)}
    registration = {
        "scope": "frozen-policy development transfer to two obligations after the first tactic",
        "unit": "one active/pending topology pair, with two worlds exchanging a pending fact",
        "shared_templates": True,
        "primary": "both worlds independently certified within eight Lean tactic checks",
        "minimum_meaningful_effect": 0.05,
        "pairs": PAIR_COUNT, "worlds": len(cases), "seeds": SEEDS, "arms": ARMS, "edits": EDITS,
        "planned_comparisons": 7,
        "train_updates": 0, "confirmatory": False,
        "previous_cohort_sha256": sha256_file(previous),
        "generator_sha256": sha256_file(Path(__file__)),
        "source_session_sha256": checkpoints,
    }
    with output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical({"schema": SCHEMA, "seed": LAYOUT_SEED, "registration": registration,
                             "layouts": layouts, "cases": cases}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sessions", type=Path, nargs=len(SEEDS))
    args = parser.parse_args()
    freeze(args.previous, args.sessions, args.output)


if __name__ == "__main__":
    main()
