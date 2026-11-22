"""Freeze fresh BPD-style equality layouts against the audited seed-151 policies."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

from configs.freeze_ppt_bpd_rewrite import ARMS
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.rewrite import BPD_REWRITE_FAMILY


SCHEMA = "ppt.bpd-rewrite-transfer.v1"
SEED = 97284061
LAYOUTS = 96


def tree(size, rng):
    if size == 1:
        return ()
    if size == 2 or rng.randrange(2) == 0:
        return (tree(size - 1, rng),)
    left = rng.randrange(1, size - 1)
    return tuple(sorted((tree(left, rng), tree(size - 1 - left, rng)), key=repr))


def draw_depth_and_periphery(rng):
    depth = rng.randrange(3, 8)
    root_sizes = [rng.randrange(1, 6), rng.randrange(1, 6)]
    decorations = [tuple(sorted((tree(size, rng) for size in root_sizes), key=repr))]
    maximum = 3 if depth <= 5 else 2
    for _ in range(depth):
        size = rng.randrange(maximum + 1)
        decorations.append((tree(size, rng),) if size else ())
    return depth, tuple(decorations)


def paired_cases(depth, decorations, ordinal, rng):
    nodes = [f"p{i}" for i in range(depth + 1)]
    base = list(zip(nodes, nodes[1:]))
    cut = rng.randrange(1, depth - 1)
    cut_edge = base[cut]
    base.append(("isolated_left", "isolated_right"))
    roles = [*nodes, "isolated_left", "isolated_right"]

    def attach(source, shape):
        target = f"spur{len(roles)}"
        roles.append(target)
        base.append((source, target))
        for branch in shape:
            attach(target, branch)

    for spine, branches in zip(nodes, decorations, strict=True):
        for branch in branches:
            attach(spine, branch)
    if len(base) > 32 or len(roles) > 48:
        raise ValueError("the frozen scaffold exceeds the shared graph budget")
    names = {role: f"v{rng.randrange(1 << 44):011x}" for role in roles}
    if len(set(names.values())) != len(names):
        raise ValueError("fresh equality binder names collided")
    order = list(range(len(base)))
    rng.shuffle(order)
    reversed_edges = {i for i in order if not rng.randrange(3)}
    variables = f"({' '.join(sorted(names.values()))} : Nat)"
    cluster = digest([SCHEMA, depth, decorations])
    variants = []
    for valid in (True, False):
        rules = base.copy()
        if not valid:
            # The two-edge switch preserves every atom degree and disconnects the target.
            rules[cut] = (cut_edge[0], "isolated_left")
            rules[depth] = (cut_edge[1], "isolated_right")
        bindings = []
        for j, index in enumerate(order):
            left, right = rules[index]
            if index in reversed_edges:
                left, right = right, left
            bindings.append(f"(h{j} : {names[left]} = {names[right]})")
        case = SearchCase("bpd_lg02", "confirm", BPD_REWRITE_FAMILY, digest([cluster, valid]), cluster,
                          f"bpd_rewrite_transfer_{ordinal}_{int(valid)}", " ".join([variables, *bindings]),
                          f"{names[nodes[0]]} = {names[nodes[-1]]}")
        variants.append({"case": asdict(case), "valid": valid})
    rng.shuffle(variants)
    return {"layout_id": cluster, "proof_checks": depth, "worlds": variants}


def freeze(development, audit_file, output):
    audit = json.loads(audit_file.read_bytes())
    dev_sha = sha256_file(development)
    if (audit["cohort_sha256"] != dev_sha or not audit["exploration_study_sha256"] or set(audit["trials"]) != set(ARMS)
            or {trial["manifest"]["seed"] for trial in audit["trials"].values()} != {151}):
        raise ValueError("the fresh readout requires the audited seed-151 exploration policies")
    rng = random.Random(SEED)
    cases = []
    for ordinal in range(LAYOUTS):
        depth, decorations = draw_depth_and_periphery(rng)
        cases.append(paired_cases(depth, decorations, ordinal, rng))
    if len({layout["layout_id"] for layout in cases}) != LAYOUTS:
        raise ValueError("the independent scaffold draw contains a structural collision")
    document = {
        "schema": SCHEMA, "seed": SEED, "generator_sha256": sha256_file(Path(__file__)),
        "source_cohort_sha256": dev_sha, "source_audit_sha256": sha256_file(audit_file),
        "reference_sha256": audit["reference_sha256"],
        "registration": {
            "seed": 151, "arms": ARMS, "layout_count": LAYOUTS, "tactic_checks": 8,
            "max_macs": 1_200_000_000, "cpu_seconds": 6, "alpha": "0.05", "minimum_effect": "0.05",
            "unit": "one independently drawn anchored equality scaffold",
            "primary": "native proof completion on one valid world per layout with the shared revisit rule",
            "secondary": "native proof completion with ordinary inference",
            "domain": "fresh BPD-style local Nat equality forests with a three-to-seven rewrite spine",
            "negative_control": "degree-matched disconnected world from the same equality scaffold",
            "source_test_partitions_opened": False,
        },
        "checkpoints": {
            arm: {key: trial[key] for key in ("session_sha256", "final_weights_sha256")}
            for arm, trial in audit["trials"].items()
        },
        "layouts": cases,
    }
    with output.open("x", encoding="utf-8", newline="\n") as file:
        file.write(canonical(document) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "audit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    freeze(args.development, args.audit, args.output)


if __name__ == "__main__":
    main()
