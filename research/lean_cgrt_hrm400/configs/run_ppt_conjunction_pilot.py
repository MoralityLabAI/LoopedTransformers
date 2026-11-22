"""Run all matched conjunction-pilot policies under one CPU process-tree budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from ppt_gym.data import sha256_file
from ppt_gym.ram_limits import TreeBudget


ARMS = {
    "hybrid": ("hybrid", "rl", False, True, True),
    "hybrid_mean": ("hybrid", "rl", False, True, False),
    "hybrid_frozen": ("hybrid", "rl", True, True, True),
    "hybrid_imitation": ("hybrid", "imitation", False, True, True),
    "trm": ("trm", "rl", False, False, False),
    "trm_frozen": ("trm", "rl", True, False, False),
}
FLAGS = ("--head-only", "--residual-fact-transport", "--premise-interaction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_bytes())
    reg = cohort["registration"]
    seed, = reg["model_seeds"]
    if reg["controls"] != list(ARMS)[1:] or not args.archive.is_file():
        raise ValueError("missing train archive or unregistered policy controls")
    args.output.mkdir(exist_ok=True)
    budget = TreeBudget(6, 2)
    statuses = {}

    def run(arm, spec):
        architecture, objective, *switches = spec
        prefix = args.output / f"{seed}-{arm}"
        command = [
            sys.executable, "-u", "-m", "ppt_gym.hybrid_rl", "run", str(args.cohort),
            str(prefix.with_suffix(".sqlite")), "--report", str(prefix.with_suffix(".json")),
            "--architecture", architecture, "--objective", objective, "--graph-context", "--ram", "2",
            "--seed", str(seed), "--alias-training", "contrast-search", "--updates", str(reg["updates"]),
            "--rollouts", str(reg["rollouts"]), "--lr", str(reg["lr"]), "--mathcan-archive", str(args.archive),
            *(flag for flag, enabled in zip(FLAGS, switches, strict=True) if enabled),
        ]
        with prefix.with_suffix(".log").open("ab") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        print(arm, code, flush=True)
        return arm, code

    try:
        with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
            futures = [pool.submit(run, arm, spec) for arm, spec in ARMS.items()]
            for future in futures:
                arm, code = future.result()
                statuses[arm] = code
    finally:
        (args.output / "campaign-resource-audit.json").write_text(
            json.dumps({"statuses": statuses, "memory": budget.stats(), "cohort": sha256_file(args.cohort)},
                       indent=2) + "\n", encoding="utf-8")
        budget.close()
    if len(statuses) != len(ARMS) or any(statuses.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
