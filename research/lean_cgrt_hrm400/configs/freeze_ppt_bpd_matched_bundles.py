"""Freeze bundle opening pairs with matched public endpoints and degrees."""

import argparse
from dataclasses import asdict
from itertools import combinations_with_replacement
import json
from pathlib import Path
import random

from configs.check_ppt_bpd_bundled_opening import SHAPES
from configs.freeze_ppt_bpd_bundled import DEV_PAIRS, TRAIN_PAIRS
from ppt_gym.bundled_rewrite import BPD_BUNDLE_FAMILY
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SCHEMA, SearchCase, load


SEED = 737191804
GENERATOR = "ppt.bpd-matched-bundle-opening.v2"


def paired_cases(shape, ordinal, split, rng):
    roles = {"start", "target"}
    bundles = []
    for side in (0, 1):
        auxiliary, dead = f"aux_{side}", f"dead_{side}"
        roles.add(dead)
        edges = []
        for branch, root in enumerate(("start", auxiliary)):
            chain = [root, *(f"chain_{side}_{branch}_{hop}" for hop in range(1, 7))]
            roles.update(chain)
            edges.extend(zip(chain, chain[1:]))
            edges.append((chain[-1], f"finish_{branch}"))
            for index, depth in enumerate(shape):
                leaf = f"leaf_{side}_{branch}_{index}"
                roles.add(leaf)
                edges.append((chain[depth], leaf))
        rng.shuffle(edges)
        bundles.append((dead, edges))

    names = {role: f"v{rng.randrange(1 << 44):011x}" for role in sorted(roles)}
    if len(set(names.values())) != len(roles):
        raise ValueError("fresh equality bundle atoms collided")
    variables = f"({' '.join(sorted(names.values()))} : Nat)"
    cluster = digest([GENERATOR, shape])
    for winner in (0, 1):
        hypotheses = []
        for side, (dead, edges) in enumerate(bundles):
            endpoints = ("target", dead) if side == winner else (dead, "target")
            resolve = dict(zip(("finish_0", "finish_1"), endpoints, strict=True))
            premises = [f"{names[left]} = {names[resolve.get(right, right)]}" for left, right in edges]
            body = premises[-1]
            for premise in reversed(premises[:-1]):
                body = f"({premise}) ∧ ({body})"
            hypotheses.append(f"(h{side} : {body})")
        yield SearchCase("loopt_bundle", split, BPD_BUNDLE_FAMILY, digest([cluster, winner]), cluster,
                         f"bpd_matched_bundle_{ordinal}_{winner}", " ".join([variables, *hypotheses]),
                         f"{names['start']} = {names['target']}")


def freeze(previous, destination):
    original = json.loads(previous.read_bytes())
    registration = original["registration"]
    if not registration["bundle_continuation"]:
        raise ValueError("expected the verified first bundle cohort")
    training = [case for case in load(previous)["train"] if case.family != BPD_BUNDLE_FAMILY]
    seen = set(SHAPES) | {tuple(layout["shape"]) for layout in registration["layouts"].values()}
    candidates = [shape for count in (2, 3, 4)
                  for shape in combinations_with_replacement(range(1, 7), count) if shape not in seen]
    rng = random.Random(SEED)
    layouts, cases = {}, []
    for ordinal, shape in enumerate(rng.sample(candidates, TRAIN_PAIRS + DEV_PAIRS)):
        split = "train" if ordinal < TRAIN_PAIRS else "dev"
        cluster = digest([GENERATOR, shape])
        layouts[cluster] = {"split": split, "shape": shape}
        cases.extend(paired_cases(shape, ordinal, split, rng))
    registration = {**registration, "layouts": layouts, "balanced_endpoint_degrees": True,
                    "scope": "paired bundle-opening development with balanced public endpoints and degrees"}
    doc = {"schema": SCHEMA, "seed": SEED, "registration": registration,
           "source_sha256": {**original["source_sha256"], "first_bundle_cohort": sha256_file(previous),
                             "matched_bundle_generator": sha256_file(Path(__file__))},
           "cases": [asdict(case) for case in [*training, *cases]]}
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(canonical(doc) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze(args.previous, args.output)


if __name__ == "__main__":
    main()
