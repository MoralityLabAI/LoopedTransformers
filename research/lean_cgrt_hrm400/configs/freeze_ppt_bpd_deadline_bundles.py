"""Freeze matched bundle choices with reachable paths on both sides of the tactic deadline."""

import argparse
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import random

from configs.freeze_ppt_bpd_bundled import DEV_PAIRS, TRAIN_PAIRS
from ppt_gym.bundled_rewrite import BPD_BUNDLE_FAMILY
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SCHEMA, SearchCase, load


GENERATOR = "ppt.bpd-deadline-bundle-opening.v3"
SEED = 1978613417


def paired_cases(spread, concentrated, ordinal, split, rng):
    bundles, roles = [], {"start", "target"}
    concentration_side = rng.randrange(2)
    for side in (0, 1):
        bridge, merge = f"bridge_{side}", f"merge_{side}"
        edges = [(bridge, merge), (merge, "target")]
        roles.update((bridge, merge))
        spur_depths = (concentrated, spread) if side == concentration_side else (spread, concentrated)
        for branch, (root, depths) in enumerate(zip(("start", f"auxiliary_{side}"), spur_depths, strict=True)):
            chain = [root, *(f"chain_{side}_{branch}_{hop}" for hop in range(1, 6))]
            roles.update(chain)
            edges.extend(zip(chain, chain[1:]))
            edges.append((chain[-1], f"finish_{branch}"))
            for index, depth in enumerate(depths):
                leaf = f"leaf_{side}_{branch}_{index}"
                roles.add(leaf)
                edges.append((chain[depth], leaf))
        rng.shuffle(edges)
        bundles.append((bridge, merge, edges))
    names = {role: f"v{rng.randrange(1 << 44):011x}" for role in sorted(roles)}
    if len(set(names.values())) != len(roles):
        raise ValueError("fresh deadline bundle atoms collided")
    variables = f"({' '.join(sorted(names.values()))} : Nat)"
    cluster = digest([GENERATOR, spread, concentrated])
    for winner in (0, 1):
        hypotheses = []
        for side, (bridge, merge, edges) in enumerate(bundles):
            endpoints = (merge, bridge) if side == winner else (bridge, merge)
            resolve = dict(zip(("finish_0", "finish_1"), endpoints, strict=True))
            premises = [f"{names[left]} = {names[resolve.get(right, right)]}" for left, right in edges]
            body = premises[-1]
            for premise in reversed(premises[:-1]):
                body = f"({premise}) ∧ ({body})"
            hypotheses.append(f"(h{side} : {body})")
        yield SearchCase("loopt_bundle", split, BPD_BUNDLE_FAMILY, digest([cluster, winner]), cluster,
                         f"bpd_deadline_bundle_{ordinal}_{winner}", " ".join([variables, *hypotheses]),
                         f"{names['start']} = {names['target']}")


def freeze(previous, destination):
    original = json.loads(previous.read_bytes())
    registration = original["registration"]
    if not registration.get("balanced_endpoint_degrees") or registration.get("public_rewrite_distances"):
        raise ValueError("expected the matched-endpoint bundle cohort")
    training = [case for case in load(previous)["train"] if case.family != BPD_BUNDLE_FAMILY]
    rng = random.Random(SEED)
    candidates = [(spread, (depth,) * count) for count in (3, 4, 5)
                  for spread in combinations(range(1, 6), count) for depth in range(1, 6)]
    layouts, cases = {}, []
    for ordinal, (spread, concentrated) in enumerate(rng.sample(candidates, TRAIN_PAIRS + DEV_PAIRS)):
        split = "train" if ordinal < TRAIN_PAIRS else "dev"
        cluster = digest([GENERATOR, spread, concentrated])
        layouts[cluster] = {"split": split, "spread": spread, "concentrated": concentrated}
        cases.extend(paired_cases(spread, concentrated, ordinal, split, rng))
    registration = {**registration, "layouts": layouts, "public_rewrite_distances": [7, 8],
                    "scope": "paired bundle-opening development with both targets reachable and one route within budget"}
    doc = {"schema": SCHEMA, "seed": SEED, "registration": registration,
           "source_sha256": {**original["source_sha256"], "matched_bundle_cohort": sha256_file(previous),
                             "deadline_bundle_generator": sha256_file(Path(__file__))},
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
