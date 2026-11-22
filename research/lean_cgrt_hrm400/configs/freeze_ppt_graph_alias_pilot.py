"""Freeze paired Horn contexts in which the first root choice needs a distant premise."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import random

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SCHEMA, SearchCase, load


def nonsense_shape(edges, vertices):
    return min(tuple(sorted((order[left], order[right]) for left, right in edges))
               for order in itertools.permutations(range(vertices)))


def worlds(split, count, seed, seen):
    rng = random.Random(seed)
    for index in range(count):
        for _ in range(128):
            lengths = sorted([rng.randint(4, 6), rng.randint(4, 6)]) if split == "train" else [rng.randint(6, 7), 7]
            nuisance = [(s, t) for s in range(5) for t in range(s + 1, 5) if rng.random() < 0.45]
            shape = (tuple(lengths), nonsense_shape(nuisance, 5))
            if shape not in seen:
                seen.add(shape)
                break
        else:
            raise ValueError("could not freeze enough distinct alias graph clusters")
        cluster = digest(["first-choice-graph", shape])
        end_a, end_b = lengths[0], sum(lengths)
        first, second = list(range(1, end_a + 1)), list(range(end_a + 1, end_b + 1))
        edges = [(src, dst) for chain in (first, second) for dst, src in zip([0, *chain], chain)]
        edges.extend((end_b + 1 + a, end_b + 1 + b) for a, b in nuisance)
        rng.shuffle(edges)
        order = list(range(end_b + 6))
        rng.shuffle(order)
        names = [f"P{i}" for i in order]
        props = "(" + " ".join(sorted(names)) + " : Prop)"
        hypotheses = [f"(r{i} : {names[src]} → {names[dst]})" for i, (src, dst) in enumerate(edges)]
        for side, leaf in enumerate((first[-1], second[-1])):
            binders = " ".join([props, f"(h0 : {names[leaf]})", *hypotheses])
            tid = f"alias_{split}_{seed}_{index}_{side}"
            yield SearchCase("loopt_horn", split, "first-choice-alias", digest([cluster, side]),
                             cluster, tid, binders, names[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bpd_cohort", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = json.loads(args.bpd_cohort.read_bytes())
    bpd_train = [case for case in load(args.bpd_cohort)["train"] if case.source == "bpd_lg02"]
    used = set()
    cases = [*worlds("train", 24, 7311, used), *bpd_train, *worlds("dev", 12, 9013, used)]
    document = {
        "schema": SCHEMA, "source_sha256": {
            "generator": sha256_file(Path(__file__)), "bpd_training_cohort": sha256_file(args.bpd_cohort),
            **original["source_sha256"],
        },
        "registration": {
            "scope": "development mechanism pilot", "tactic_checks": 8,
            "unit": "two alternative-source worlds matched on public lexical inputs",
            "train_alias_clusters": 24, "dev_alias_clusters": 12, "model_seeds": [103],
            "primary": "graph hybrid RL versus equally informed graph TRM RL",
            "calibration": "equally informed frozen-body graph TRM with trained heads",
            "old_observation_ceiling": 0.5,
        },
        "cases": [asdict(case) for case in cases],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(document) + "\n")


if __name__ == "__main__":
    main()
