"""Run matched Lean policy arms on the same on-time proof objective."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from configs.freeze_ppt_deadline import ARMS, DEADLINE, MAX_CHECKS, SEED
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


HYBRID = ("--residual-fact-transport", "--premise-interaction", "--root-fact-adapter")
OPTIONS = {
    "arrival_rl": (*HYBRID, "--arrival-transport"),
    "readout_calibration": (*HYBRID, "--arrival-transport", "--head-only"),
    "deadline_imitation": (*HYBRID, "--arrival-transport", "--objective", "imitation"),
    "all_proofs_imitation": (*HYBRID, "--arrival-transport", "--objective", "imitation"),
    "terminal_readout_rl": HYBRID,
    "trm_rl": (),
    "trm_calibration": ("--head-only",),
}


def native_witness(cohort, folder):
    from configs.check_ppt_conjunction_pilot import ProofLengthWitness
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    by_pair = defaultdict(list)
    configure(2, 0)
    params = SimpleNamespace(cpu_seconds=6, max_macs=1_200_000_000)
    tree = TreeBudget(2, 2)
    try:
        with LeanRepl(timeout=15) as repl:
            if repl.version != "4.19.0":
                raise ValueError("expected pinned native Lean")
            for case in load(cohort)["dev"]:
                trials = [rollout(repl, tree, case, ProofLengthWitness(), params, first_choice=action)
                          for action in (0, 1)]
                checks = [episode["tactic_checks"] for episode in trials]
                replay = [replay_search(repl, case, episode) for episode in trials]
                if (sorted(checks) != [DEADLINE, MAX_CHECKS]
                        or not all(ep["verified_closed"] and ep["within_cpu_budget"] for ep in trials)
                        or not all(episode["replay_passed"] for episode in replay)):
                    raise ValueError(f"native proof-cost reference failed for {case.tid}")
                by_pair[case.cluster].append({
                    "root": case.root, "menu": trials[0]["events"][0]["candidate_menu"], "checks": checks,
                    "early_first": checks.index(DEADLINE), "replay": replay,
                    "certificates": [episode["certificate"] for episode in trials],
                })
    finally:
        tree.close()
    for a, b in by_pair.values():
        if a["menu"] != b["menu"] or a["early_first"] == b["early_first"]:
            raise ValueError("paired worlds must swap the on-time tactic without changing the root menu")
    result = {"cohort_sha256": sha256_file(cohort), "pairs": by_pair}
    (folder / "native-witness.json").write_text(canonical(result) + "\n", encoding="utf-8")
    print("native witness verified", len(by_pair), "pairs", flush=True)


def train(args, reg):
    def run(arm):
        arch = "trm" if arm.startswith("trm_") else "hybrid"
        source = args.initialization / f"{SEED}-{'trm' if arch == 'trm' else 'hybrid_imitation'}.sqlite"
        root = args.output / arm
        cmd = [
            sys.executable, "-u", "-m", "ppt_gym.hybrid_rl", "run", str(args.cohort),
            str(root.with_suffix(".sqlite")), "--report", str(root.with_suffix(".json")),
            "--architecture", arch, "--graph-context", "--initialize-from", str(source),
            "--alias-training", "contrast-search", "--ram", "2", "--seed", str(SEED), "--updates",
            str(reg["updates"]), "--rollouts", str(reg["rollouts"]), "--lr", str(reg["lr"]),
            "--mathcan-archive", str(args.archive), *OPTIONS[arm],
        ]
        if arm != "all_proofs_imitation":
            cmd.extend(("--proof-deadline", str(reg["proof_deadline"])))
        with root.with_suffix(".log").open("ab") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(arm, code, flush=True)
        return arm, code

    tree = TreeBudget(6, 2)
    statuses = {}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            statuses = dict(pool.map(run, ARMS))
    finally:
        receipt = {"statuses": statuses, "memory": tree.stats(), "cohort": sha256_file(args.cohort)}
        (args.output / "campaign-resource-audit.json").write_text(canonical(receipt) + "\n", encoding="utf-8")
        tree.close()
    if statuses != dict.fromkeys(ARMS, 0):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "archive", "initialization", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--witness-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    reg = json.loads(args.cohort.read_bytes())["registration"]
    if tuple(reg["controls"]) != ARMS[1:] or reg["model_seeds"] != [SEED] or reg["proof_deadline"] != DEADLINE:
        raise ValueError("wrong registered deadline study")
    args.output.mkdir(exist_ok=True)
    if args.witness_only:
        native_witness(args.cohort, args.output)
        return
    witness = args.output / "native-witness.json"
    if not witness.is_file():
        # Windows retains each assigned Job for the lifetime of its process.
        # Keep the smaller verifier Job out of the parallel training parent.
        subprocess.run([sys.executable, "-u", "-m", "configs.run_ppt_deadline",
                        str(args.cohort), str(args.archive), str(args.initialization), str(args.output),
                        "--witness-only"], check=True)
    report = json.loads(witness.read_bytes())
    if report["cohort_sha256"] != sha256_file(args.cohort) or len(report["pairs"]) != len(reg["structural_pairs"]["dev"]):
        raise ValueError("incomplete or changed native deadline witness")
    train(args, reg)


if __name__ == "__main__":
    main()
