"""Train root-only and whole-model controls from the same verified imitation checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from configs.freeze_ppt_root_adaptation import ARMS, SEEDS
from ppt_gym.data import canonical, sha256_file
from ppt_gym.ram_limits import TreeBudget


OPTIONS = {
    "root_rl": ("--root-fact-adapter",),
    "root_calibration": ("--root-fact-adapter", "--head-only"),
    "root_imitation": ("--root-fact-adapter", "--objective", "imitation"),
    "warm_full_rl": (),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "archive", "initialization_work", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    reg = json.loads(args.cohort.read_bytes())["registration"]
    if (tuple(reg["model_seeds"]) != SEEDS or tuple(reg["controls"]) != ARMS[1:]
            or not reg["evalmath_train_roots"]):
        raise ValueError("warm-start cohort or registered arms changed")
    args.output.mkdir(exist_ok=True)
    tasks = [(seed, arm) for seed in SEEDS for arm in ARMS]

    def run(seed, arm):
        original = "hybrid-conjunction-pilot" if seed == 103 else f"hybrid-conjunction-seed-{seed}"
        source = args.initialization_work / original / f"{seed}-hybrid_imitation.sqlite"
        label = f"{seed}-{arm}"
        output = args.output / label
        cmd = [
            sys.executable, "-u", "-m", "ppt_gym.hybrid_rl", "run", str(args.cohort), str(output.with_suffix(".sqlite")),
            "--report", str(output.with_suffix(".json")), "--architecture", "hybrid", "--graph-context",
            "--residual-fact-transport", "--premise-interaction", "--initialize-from", str(source),
            "--alias-training", "contrast-search", "--seed", str(seed), "--updates", str(reg["updates"]),
            "--rollouts", str(reg["rollouts"]), "--lr", str(reg["lr"]), "--ram", "2", "--mathcan-archive",
            str(args.archive), *OPTIONS[arm],
        ]
        with output.with_suffix(".log").open("ab") as log:
            status = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(label, status, flush=True)
        return label, status

    tree = TreeBudget(6, 2)
    statuses = {}
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(run, *task) for task in tasks]
            for future in futures:
                arm, status = future.result()
                statuses[arm] = status
    finally:
        receipt = {"statuses": statuses, "memory": tree.stats(), "cohort": sha256_file(args.cohort)}
        (args.output / "campaign-resource-audit.json").write_text(canonical(receipt) + "\n", encoding="utf-8")
        tree.close()
    if len(statuses) != len(tasks) or any(statuses.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
