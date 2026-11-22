"""Prepare train-only EvalMath declarations and a matched warm-start RL study."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase


ARMS = ("root_rl", "root_calibration", "root_imitation", "warm_full_rl")
SEEDS = (103, 151, 197)
EXACT_DECLARATION = re.compile(r"theorem ([A-Za-z_][A-Za-z0-9_]*) (\([^\n]+\)) : ([^\n]+)")
BINDER = re.compile(r"\(([A-Za-z][A-Za-z0-9_ ]*) : ([^()]+)\)")


def evalmath_cases(training_file, manifest_file):
    manifest = json.loads(manifest_file.read_bytes())
    if (manifest["schema"] != "BPD-LG-01-v2" or training_file.name != "train.jsonl"
            or sha256_file(training_file) != manifest["sha256"]["train.jsonl"]):
        raise ValueError("EvalMath input must be the manifest-verified original training file")
    rows = [json.loads(line) for line in training_file.read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["split_counts"]["train"] or any(row["split"] != "train" for row in rows):
        raise ValueError("EvalMath source includes an unexpected split")
    for row in rows:
        if row["family"] != "exact" or row["host_label"] != "positive_kernel_checked":
            continue
        declaration = EXACT_DECLARATION.fullmatch(row["public_goal_source"])
        if declaration is None:
            continue
        tid, context, target = declaration.groups()
        groups = BINDER.findall(context)
        if not groups or " ".join(f"({n} : {t})" for n, t in groups) != context or groups[0][1] != "Prop":
            continue
        atoms = set(groups[0][0].split()) | {"True"}

        def supported(proposition):
            return all(formula.removesuffix(" ∧ True") in atoms for formula in proposition.split(" → "))

        if " → " in target or not supported(target) or not all(supported(ty) for _, ty in groups[1:]):
            continue
        binders = [f"({groups[0][0]} : Prop)"]
        for index, (_, ty) in enumerate(groups[1:]):
            role = "r" if " → " in ty else "h"
            binders.append(f"({role}{index} : {ty})")
        yield SearchCase("evalmath_lg01", "train", "evalmath_exact", row["statement_normalized_sha256"],
                         row["root_normalized_sha256"], tid, " ".join(binders), target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "evalmath_train", "evalmath_manifest", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    original = json.loads(args.pilot.read_bytes())
    additions = list(evalmath_cases(args.evalmath_train, args.evalmath_manifest))
    if not additions:
        raise ValueError("missing EvalMath training examples supported by the public tactic grammar")
    doc = {
        **original,
        "cases": original["cases"] + [asdict(case) for case in additions],
        "source_sha256": {**original["source_sha256"], "evalmath_lg01_train": sha256_file(args.evalmath_train),
                         "evalmath_lg01_manifest": sha256_file(args.evalmath_manifest)},
        "registration": {
            **original["registration"], "scope": "reused conjunction development with new warm-start training",
            "initialization_cohort_sha256": sha256_file(args.pilot), "model_seeds": SEEDS, "controls": ARMS[1:],
            "updates": 48, "primary": "root recurrence RL versus matched readout calibration and imitation",
            "baseline": "identically initialized whole-model RL and the pre-adaptation imitation checkpoint",
            "evalmath_train_roots": [case.root for case in additions],
            "confirmation_scored": False,
        },
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(doc) + "\n")
    print(f"added {len(additions)} EvalMath training declarations")


if __name__ == "__main__":
    main()
