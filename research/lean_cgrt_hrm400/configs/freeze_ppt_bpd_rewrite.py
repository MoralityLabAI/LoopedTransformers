"""Freeze original BPD-LG-02 rewrites alongside mixed public training declarations."""

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import BPD_GENERATOR, SCHEMA, SearchCase, bpd_generator, load
from ppt_gym.rewrite import BPD_REWRITE_FAMILY


ARMS = ("hybrid_rl", "hybrid_calibration", "hybrid_imitation", "trm_rl", "trm_calibration")


def registered_exploration(cohort, study):
    if study is None:
        return None
    registration = json.loads(study.read_bytes())
    rule = Path(__file__).resolve().parents[1] / "ppt_gym" / "rewrite.py"
    if (registration["schema"] != "ppt.bpd-rewrite-exploration.v1"
            or registration["cohort_sha256"] != sha256_file(cohort)
            or tuple(registration["arms"]) != ARMS or registration["seed"] != 151
            or registration["rewrite_rule_sha256"] != sha256_file(rule)):
        raise ValueError("the rewrite exploration study differs from its registered inputs")
    return registration["rewrite_rule_sha256"]


def rewrites():
    generator = bpd_generator()
    for key, split in sorted(generator.assign_groups().items()):
        if key[0] != "rw" or split not in ("train", "dev"):
            continue
        cluster = generator.digest(generator.group_signature(*key))
        for nuisance, variant in product((0, 1), repeat=2):
            spec = generator.graph_spec(*key, nuisance, variant)
            tid = "thm_bpd2_" + generator.digest((*key, nuisance, variant))[:16]
            declaration, _ = generator.source(spec, tid)
            signature = declaration.removeprefix(f"theorem {tid} ").removesuffix(" := by")
            binders, target = signature.rsplit(" : ", 1)
            case = SearchCase("bpd_lg02", split, BPD_REWRITE_FAMILY, spec["alpha_hash"],
                              cluster, tid, binders, target)
            yield case, generator.semantic_label(spec)


def freeze(mixed, destination):
    original = json.loads(mixed.read_bytes())
    cases = [case for case, _ in rewrites()]
    if [sum(case.split == split for case in cases) for split in ("train", "dev")] != [64, 16]:
        raise ValueError("the original BPD rewrite partitions changed")
    training = load(mixed)["train"]
    sources = {case.source for case in training}
    if sources != {"loopt_horn", "bpd_lg02", "evalmath_lg01"}:
        raise ValueError("expected the mixed LoopedTransformers, BPD, and EvalMath source cohort")
    cohort = {
        "schema": SCHEMA, "seed": 151,
        "source_sha256": {**original["source_sha256"], "mixed_public_training": sha256_file(mixed),
                          "bpd_lg02_rewrite": sha256_file(BPD_GENERATOR)},
        "registration": {
            "scope": "original BPD-LG-02 rewrite development",
            "arms": ARMS, "model_seeds": [151], "tactic_checks": 8, "cpu_seconds": 6,
            "minimum_meaningful_effect": 0.05, "unit": "BPD structural group",
            "primary": "independently certified proofs on the native rewrite development declarations",
            "symbolic_reference": "public equality shortest path under the same rewrite budget",
            "initialization_cohort_sha256": original["registration"]["initialization_cohort_sha256"],
            "source_cases": {"bpd_rewrite_train": 64, "bpd_rewrite_dev": 16},
            "updates": (len(training) + 64 + 1) // 2, "rollouts": 2, "lr": 0.001,
            "confirmation_scored": False,
        },
        "cases": [asdict(case) for case in [*training, *cases]],
    }
    with destination.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(cohort) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    freeze(args.mixed, args.destination)


if __name__ == "__main__":
    main()
