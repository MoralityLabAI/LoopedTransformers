"""Pre-register novel branched eight-check Horn pairs for frozen policies."""
import argparse
from dataclasses import asdict
from functools import cache
import itertools
from pathlib import Path
import random

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase

SCHEMA = "ppt.residual_fact_confirmation.v1"
SEED = 591427
PAIR_COUNT = 80
ARMS = ("transport", "transport-head", "trm", "head-only")


def compositions(total, parts):
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in compositions(total - first, parts - 1):
            yield (first, *rest)


@cache
def tree_shapes(nodes):
    if nodes == 1:
        return ((),)
    return tuple(sorted({
        tuple(sorted(children))
        for degree in range(1, 4)
        for sizes in compositions(nodes - 1, degree)
        for children in itertools.product(*(tree_shapes(size) for size in sizes))
    }))


def public_facts(shape):
    if not shape:
        return "public"
    return tuple(public_facts(child) for child in shape)


def mark_switch(shape, depth=0):
    if not shape:
        if depth >= 2:
            yield "switch"
        return
    for i, child in enumerate(shape):
        for marked in mark_switch(child, depth + 1):
            others = [public_facts(other) for j, other in enumerate(shape) if i != j]
            yield tuple(sorted([*others, marked], key=repr))


def has_branch(tree):
    return len(tree) > 1 or any(has_branch(child) for child in tree)


def build_pair(branches, rng, number):
    # Two disjoint seven-node proof trees make exactly eight tactics through
    # either successful branch, counting the first tactic at the shared goal.
    symbols = [f"P{i}" for i in range(15)]
    rng.shuffle(symbols)
    goal = symbols[0]
    symbols_left = iter(symbols[1:])
    hypotheses, facts, switches = [], [], []

    def encode(tree):
        name = next(symbols_left)
        if tree == "public":
            facts.append(name)
        elif tree == "switch":
            switches.append(name)
        else:
            children = [encode(child) for child in tree]
            hypotheses.append(" → ".join([*children, name]))
        return name

    for branch in branches:
        hypotheses.append(f"{encode(branch)} → {goal}")
    if next(symbols_left, None) is not None or len(switches) != 2:
        raise ValueError("the pair does not contain two seven-node proof trees")
    rng.shuffle(hypotheses)
    cluster = digest(["branched-transport-v1", branches])
    variables = "(" + " ".join(sorted(symbols)) + " : Prop)"
    rules = [f"(r{i} : {rule})" for i, rule in enumerate(hypotheses)]
    for world, fact in enumerate(switches):
        live = [*sorted(facts), fact]
        binders = " ".join([variables, *[f"(h{i} : {value})" for i, value in enumerate(live)], *rules])
        yield SearchCase("loopt_horn", "confirm", "branched-first-choice-alias", digest([cluster, world]),
                         cluster, f"transport_confirm_{number}_{world}", binders, goal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rng = random.Random(SEED)
    marked = sorted({tree for shape in tree_shapes(7) if has_branch(shape)
                     for tree in mark_switch(shape)}, key=repr)
    combinations = list(itertools.combinations(marked, 2))
    if len(combinations) < PAIR_COUNT:
        raise ValueError("insufficient structurally distinct branched pairs")
    pairs = rng.sample(combinations, PAIR_COUNT)
    cases = [asdict(case) for i, pair in enumerate(pairs) for case in build_pair(pair, rng, i)]
    doc = {
        "schema": SCHEMA, "seed": SEED, "development_sha256": sha256_file(args.development),
        "generator_sha256": sha256_file(Path(__file__)),
        "registration": {"arms": ARMS, "checkpoint_seed": 103, "independent_pairs": PAIR_COUNT,
                         "tactic_checks": 8, "proof_cpu_seconds": 6, "max_macs": 1_200_000_000,
                         "primary": "transport versus full graph TRM and both frozen-backbone controls",
                         "minimum_meaningful_gain": "0.05", "alpha": "0.05", "planned_comparisons": 3,
                         "scope": "frozen seed-103 policies on sampled, structurally distinct branched pairs",
                         "unit": "the two premise worlds from one unordered pair of marked proof trees",
                         "activation_gate": "development transport explains both worlds in more pairs than full graph TRM"},
        "cases": cases,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(doc) + "\n")
    print(f"sealed {PAIR_COUNT} paired roots without invoking a policy")


if __name__ == "__main__":
    main()
