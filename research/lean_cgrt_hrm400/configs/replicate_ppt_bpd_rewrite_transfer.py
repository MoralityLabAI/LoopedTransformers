"""Replicate the native BPD transfer comparison on two frozen model seeds."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from configs.evaluate_ppt_bpd_rewrite_transfer import MODES, one_arm, prepare, read_arm
from configs.freeze_ppt_bpd_rewrite import ARMS
from ppt_gym.data import canonical, sha256_file
from ppt_gym.ram_limits import TreeBudget


SEEDS = (103, 197)
SCHEMA = "ppt.bpd-rewrite-transfer-replication.v1"


def audit_file(audits, seed):
    return audits / f"bpd-rewrite-exploration-audit-{seed}.json"


def register_transfer(args):
    original = audit_file(args.audits, 151)
    doc, _, cases = prepare(args.cohort, args.witness, original)
    seeds = {}
    for seed in SEEDS:
        path = audit_file(args.audits, seed)
        audit = json.loads(path.read_bytes())
        if (audit["seed"] != seed or not audit["exploration_study_sha256"]
                or audit["cohort_sha256"] != doc["source_cohort_sha256"]
                or audit["reference_sha256"] != doc["reference_sha256"]
                or set(audit["trials"]) != set(ARMS)):
            raise ValueError("expected the registered mixed-source exploration audit for both seeds")
        seeds[str(seed)] = {"audit_sha256": sha256_file(path), "checkpoints": {
            arm: {key: trial[key] for key in ("session_sha256", "final_weights_sha256")}
            for arm, trial in audit["trials"].items()
        }}
    reg = {"schema": SCHEMA, "cohort_sha256": sha256_file(args.cohort), "witness_sha256": sha256_file(args.witness),
           "original_audit_sha256": sha256_file(original), "seeds": seeds, "arms": ARMS, "layout_count": len(cases),
           "scope": "frozen seeds 103 and 197 on the previously scored BPD-style transfer layouts",
           "primary": "paired native certification with the shared revisit rule",
           "secondary": "paired native certification with ordinary inference",
           "minimum_effect": doc["registration"]["minimum_effect"], "alpha": doc["registration"]["alpha"],
           "experimental_unit": "layout, grouping both model seeds"}
    with args.registry.open("x", encoding="utf-8", newline="\n") as output:
        output.write(canonical(reg) + "\n")
    print(f"Registered {len(cases)} layouts and {len(seeds)} frozen model seeds.")


def inputs(args):
    reg = json.loads(args.registry.read_bytes())
    original = audit_file(args.audits, 151)
    if (reg["schema"] != SCHEMA or reg["cohort_sha256"] != sha256_file(args.cohort)
            or reg["witness_sha256"] != sha256_file(args.witness) or reg["original_audit_sha256"] != sha256_file(original)
            or tuple(reg["arms"]) != ARMS or set(reg["seeds"]) != set(map(str, SEEDS))
            or any(reg["seeds"][str(seed)]["audit_sha256"] != sha256_file(audit_file(args.audits, seed))
                   for seed in SEEDS)):
        raise ValueError("the transfer inputs differ from the registered replication")
    doc, witness, cases = prepare(args.cohort, args.witness, original)
    if reg["layout_count"] != len(cases):
        raise ValueError("the replicate proof population changed")
    return reg, doc, witness, cases


def seed_document(reg, doc, seed):
    return {**doc, "checkpoints": reg["seeds"][str(seed)]["checkpoints"],
            "registration": {**doc["registration"], "seed": seed}}


def evidence_path(args, seed, arm):
    return args.output / f"{seed}-{arm}.json.gz"


def worker(args, reg, doc, witness, cases):
    tree = TreeBudget(2, 1)
    try:
        options = SimpleNamespace(arm=args.arm, cohort=args.cohort, witness=args.witness,
                                  sessions=args.sessions / f"hybrid-bpd-rewrite-exploration-{args.seed}",
                                  source_audit=audit_file(args.audits, args.seed))
        one_arm(options, tree, seed_document(reg, doc, args.seed), witness, cases, evidence_path(args, args.seed, args.arm))
    finally:
        tree.close()


def summarize(reg, cases, outcomes):
    from spiral_ln.evaluation_finite_sample import evaluate_manifest

    comparisons = {}
    for inference in MODES:
        comparisons[inference] = {}
        for control in ARMS[1:]:
            rows = [{"case_id": f"{seed}:{root}", "cluster_id": case.cluster,
                     "candidate": int(outcomes[(seed, "hybrid_rl")]["rows"][root][inference]["certified"]),
                     "baseline": int(outcomes[(seed, control)]["rows"][root][inference]["certified"])}
                    for seed in SEEDS for root, case in cases.items()]
            comparisons[inference][control] = evaluate_manifest({
                "schema_version": 1, "experiment_id": f"bpd-rewrite-seed-replication-{inference}-{control}",
                "target_population": reg["scope"], "evaluation_mode": "sampled", "alpha": reg["alpha"],
                "planned_comparisons": 2 * (len(ARMS) - 1), "minimum_effect": reg["minimum_effect"],
                "hypothesis_preregistered": True, "sealed_readout": False,
                "clusters_independent": False, "exchangeable_pair_labels": False, "cases": rows,
            })
    return comparisons


def campaign(args, reg, doc, cases):
    args.output.mkdir(exist_ok=True)
    tree = TreeBudget(6, 2)

    def run(task):
        seed, arm = task
        receipt = evidence_path(args, seed, arm)
        if not receipt.exists():
            command = [sys.executable, "-u", "-m", "configs.replicate_ppt_bpd_rewrite_transfer", "run",
                       *(str(getattr(args, field)) for field in ("cohort", "witness", "audits", "registry", "sessions", "output")),
                       "--seed", str(seed), "--arm", arm]
            with (args.output / f"{seed}-{arm}.log").open("ab") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = read_arm(receipt, arm, seed_document(reg, doc, seed), cases, reg["cohort_sha256"], reg["witness_sha256"])
        counts = {mode: sum(row[mode]["certified"] for row in record["rows"].values()) for mode in MODES}
        print(seed, arm, counts, flush=True)
        return task, record

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = dict(pool.map(run, ((seed, arm) for seed in SEEDS for arm in ARMS)))
        report = {"registry_sha256": sha256_file(args.registry), "comparisons": summarize(reg, cases, outcomes),
                  "receipts": {f"{seed}-{arm}": sha256_file(evidence_path(args, seed, arm)) for seed, arm in outcomes},
                  "memory": tree.stats()}
        with (args.output / "summary.json").open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    finally:
        tree.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    evaluate = commands.add_parser("run")
    for command in (register, evaluate):
        for name in ("cohort", "witness", "audits", "registry"):
            command.add_argument(name, type=Path)
    for name in ("sessions", "output"):
        evaluate.add_argument(name, type=Path)
    evaluate.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    evaluate.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "register":
        register_transfer(args)
        return
    if bool(args.seed) != bool(args.arm):
        parser.error("supply both --seed and --arm for an individual replicate trial")
    reg, doc, witness, cases = inputs(args)
    if args.arm:
        worker(args, reg, doc, witness, cases)
    else:
        campaign(args, reg, doc, cases)


if __name__ == "__main__":
    main()
