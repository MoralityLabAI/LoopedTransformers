"""Run the matched CPU trials on the frozen original BPD rewrite declarations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from configs.freeze_ppt_bpd_rewrite import ARMS, registered_exploration
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "reference", "archive", "initialization", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--exploration-study", type=Path)
    args = parser.parse_args()
    exploration = registered_exploration(args.cohort, args.exploration_study)
    cohort_sha = sha256_file(args.cohort)
    registration = json.loads(args.cohort.read_bytes())["registration"]
    reference = json.loads(args.reference.read_bytes())
    if (tuple(registration["arms"]) != ARMS or registration["model_seeds"] != [151]
            or reference["cohort_sha256"] != cohort_sha or reference["lean"] != "4.19.0"
            or set(reference["rows"]) != {case.root for case in load(args.cohort)["dev"]}
            or not all(row["replay"]["replay_passed"] for row in reference["rows"].values())
            or sum(row["certified"] for row in reference["rows"].values()) != 8):
        raise ValueError("expected the registered rewrite cohort and its complete native reference")
    args.output.mkdir(exist_ok=True)

    def run(arm):
        architecture, objective = arm.split("_", 1)
        source = "hybrid_imitation" if architecture == "hybrid" else "trm"
        output = args.output / arm
        command = [
            sys.executable, "-u", "-m", "ppt_gym.hybrid_rl", "run", str(args.cohort),
            str(output.with_suffix(".sqlite")), "--report", str(output.with_suffix(".json")),
            "--architecture", architecture, "--graph-context",
            "--initialize-from", str(args.initialization / f"151-{source}.sqlite"),
            "--alias-training", "contrast-search", "--seed", "151", "--ram", "2",
            "--updates", str(registration["updates"]), "--rollouts", str(registration["rollouts"]),
            "--lr", str(registration["lr"]), "--cpu-seconds", str(registration["cpu_seconds"]),
            "--mathcan-archive", str(args.archive),
        ]
        if architecture == "hybrid":
            command.extend(("--residual-fact-transport", "--premise-interaction"))
        if objective == "imitation":
            command.extend(("--objective", "imitation"))
        if objective == "calibration":
            command.append("--head-only")
        if exploration:
            command.append("--rewrite-exploration")
        with output.with_suffix(".log").open("ab") as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        print(arm, status, flush=True)
        return arm, status

    tree = TreeBudget(6, 2)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            statuses = dict(pool.map(run, ARMS))
        receipt = {
            "statuses": statuses, "memory": tree.stats(), "cohort": cohort_sha,
            "reference_sha256": sha256_file(args.reference),
        }
        if exploration:
            receipt["exploration_study_sha256"] = sha256_file(args.exploration_study)
        destination = args.output / "campaign-resource-audit.json"
        partial = destination.with_suffix(".partial")
        partial.write_text(canonical(receipt) + "\n", encoding="utf-8")
        partial.replace(destination)
    finally:
        tree.close()
    if any(statuses.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
