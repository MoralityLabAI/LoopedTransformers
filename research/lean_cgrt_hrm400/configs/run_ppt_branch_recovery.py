"""Validate both checkpoint choices and train matched recovery policies."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from configs.freeze_ppt_branch_recovery import ARMS, PROOF_CHECKS, SEED
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


HYBRID = ("--residual-fact-transport", "--premise-interaction")
UNROUTED = "hybrid_unrouted_rl"
OPTIONS = {
    "hybrid_recovery_rl": HYBRID,
    "hybrid_calibration": (*HYBRID, "--head-only"),
    "hybrid_imitation": (*HYBRID, "--objective", "imitation"),
    "trm_recovery_rl": (),
    "trm_calibration": ("--head-only",),
    UNROUTED: HYBRID,
}


def registered_seed(cohort, study, seed):
    if study is None and seed is None:
        return SEED
    if study is None or seed is None:
        raise ValueError("recovery replication requires both a seed and its registration")
    document = json.loads(study.read_bytes())
    if (document["schema"] != "ppt.branch-recovery-seed-replication.v1"
            or document["cohort_sha256"] != sha256_file(cohort)
            or tuple(document["arms"]) != ARMS or seed not in document["seeds"]):
        raise ValueError("recovery seed is outside the registered comparison")
    return seed


def native_witness(cohort, output):
    from configs.check_ppt_conjunction_pilot import ProofLengthWitness
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(2, 0)
    options = SimpleNamespace(cpu_seconds=6, max_macs=1_200_000_000)
    witness = ProofLengthWitness()
    pairs = defaultdict(list)
    tree = TreeBudget(2, 2)
    try:
        with LeanRepl(timeout=15) as repl:
            if repl.version != "4.19.0":
                raise ValueError("unexpected native Lean toolchain")
            for case in load(cohort)["dev"]:
                outcomes = []
                for decision in (0, 1):
                    ep = rollout(repl, tree, case, witness, options, recovery_choice=decision)
                    native = replay_search(repl, case, ep)
                    event, = (e for e in ep["events"] if e["kind"] == "recovery")
                    if (event["checks_used"] != 1 or event["input"]["explored"] != 0
                            or event["selected_index"] != decision or not native["replay_passed"]):
                        raise ValueError("expected the registered keep-or-abandon checkpoint")
                    outcomes.append({"decision": decision, "observation": event["input"],
                                     "checks": ep["tactic_checks"], "native": native,
                                     "success": ep["verified_closed"] and ep["within_cpu_budget"],
                                     "certificate": ep["certificate"]})
                winners = [row for row in outcomes if row["success"]]
                if len(winners) != 1 or winners[0]["checks"] != PROOF_CHECKS + winners[0]["decision"]:
                    raise ValueError(f"recovery reference failed for {case.tid}")
                pairs[case.cluster].append({"root": case.root, "winner": winners[0]["decision"], "outcomes": outcomes})
                if len(pairs[case.cluster]) == 2 and len(pairs) % 8 == 0:
                    print("checked both recovery choices on", len(pairs), "pairs", flush=True)
    finally:
        tree.close()
    for left, right in pairs.values():
        if (left["winner"] == right["winner"]
                or left["outcomes"][0]["observation"]["parent_menu"] != right["outcomes"][0]["observation"]["parent_menu"]):
            raise ValueError("paired worlds must swap the successful recovery choice")
    output.write_text(canonical({"cohort_sha256": sha256_file(cohort), "pairs": pairs}) + "\n", encoding="utf-8")
    print("native recovery witness verified", len(pairs), "pairs", flush=True)


def train(args, reg, seed):
    arms = (*ARMS, UNROUTED) if reg.get("pending_arrival_ablation") else ARMS
    cohort_sha = sha256_file(args.cohort)
    receipt = args.output / "campaign-resource-audit.json"
    previous = json.loads(receipt.read_bytes()) if receipt.exists() else None
    if previous is not None and (previous["cohort"] != cohort_sha
                                 or not set(previous["statuses"]).issubset(arms)):
        raise ValueError("resource receipt belongs to a different campaign")

    def run(arm):
        architecture = "trm" if arm.startswith("trm_") else "hybrid"
        parent_arm = "trm" if architecture == "trm" else "hybrid_imitation"
        source = args.initialization / f"{seed}-{parent_arm}.sqlite"
        path = args.output / arm
        router = ("--pending-arrival-router",) if (
            args.pending_arrival_router and architecture == "hybrid" and arm != UNROUTED
        ) else ()
        command = [
            sys.executable, "-u", "-m", "ppt_gym.hybrid_rl", "run", str(args.cohort), str(path.with_suffix(".sqlite")),
            "--report", str(path.with_suffix(".json")), "--architecture", architecture, "--graph-context",
            "--initialize-from", str(source), "--branch-recovery", "--prefer-direct-fact",
            "--alias-training", "contrast-search", "--seed", str(seed), "--updates", str(reg["updates"]),
            "--rollouts", str(reg["rollouts"]), "--lr", str(reg["lr"]), "--ram", "2", "--mathcan-archive",
            str(args.archive), *OPTIONS[arm], *router,
        ]
        log_path = path.with_suffix(".log")
        with log_path.open("ab") as log:
            start = log.tell()
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            with log_path.open("rb") as log:
                log.seek(start)
                locked = log.read().rstrip().endswith(b"sqlite3.OperationalError: database is locked")
            if locked:
                print(arm, "resuming after a SQLite lock", flush=True)
                with log_path.open("ab") as log:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        print(arm, code, flush=True)
        return arm, code

    tree = TreeBudget(6, 2)
    statuses = {}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            statuses = dict(pool.map(run, arms))
    finally:
        try:
            attempt = {"statuses": statuses, "memory": tree.stats()}
            result = {"cohort": cohort_sha, **attempt}
            if previous is not None:
                previous_attempts = previous.get("attempts", [
                    {"statuses": previous["statuses"], "memory": previous["memory"]}
                ])
                result["attempts"] = [*previous_attempts, attempt]
            temporary = receipt.with_suffix(".partial")
            temporary.write_text(canonical(result) + "\n", encoding="utf-8")
            temporary.replace(receipt)
        finally:
            tree.close()
    if statuses != dict.fromkeys(arms, 0):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "archive", "initialization", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--witness-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, len(ARMS) + 1), default=len(ARMS))
    parser.add_argument("--seed-study", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pending-arrival-router", action="store_true")
    args = parser.parse_args()
    reg = json.loads(args.cohort.read_bytes())["registration"]
    if tuple(reg["controls"]) != ARMS[1:] or reg["model_seeds"] != [SEED] or reg["recovery_rollouts_per_world"] != 1:
        raise ValueError("wrong registered recovery experiment")
    if bool(reg.get("pending_arrival_ablation")) != args.pending_arrival_router:
        raise ValueError("the router flag must match the frozen pending-arrival comparison")
    seed = registered_seed(args.cohort, args.seed_study, args.seed)
    args.output.mkdir(exist_ok=True)
    witness = args.output / "native-witness.json"
    if args.witness_only:
        native_witness(args.cohort, witness)
        return
    if not witness.is_file():
        router = ("--pending-arrival-router",) if args.pending_arrival_router else ()
        subprocess.run([sys.executable, "-u", "-m", "configs.run_ppt_branch_recovery",
                        str(args.cohort), str(args.archive), str(args.initialization), str(args.output),
                        "--witness-only", *router], check=True)
    reference = json.loads(witness.read_bytes())
    if (reference["cohort_sha256"] != sha256_file(args.cohort)
            or len(reference["pairs"]) != len(reg["structural_pairs"]["dev"])):
        raise ValueError("invalid native recovery witness")
    train(args, reg, seed)


if __name__ == "__main__":
    main()
