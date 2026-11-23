"""Command line for the CPU-only Lean reasoning and online learning gym."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ram_limits import TreeBudget


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--ram", type=int, choices=(2, 4, 6), default=6, help="process-tree cap in GiB")
    root.add_argument("--threads", type=int, default=2)
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("curriculum", type=Path)
    freeze.add_argument("--seed", type=int, default=904)
    freeze.add_argument("--families", type=int, default=2)
    freeze.add_argument("--queries", type=int, default=2, help="query roots per source family")
    freeze.add_argument("--rounds", type=int, default=1)
    for command in ("run", "replay", "report", "export"):
        cli = commands.add_parser(command)
        cli.add_argument("curriculum", type=Path)
        cli.add_argument("session", type=Path)
        if command != "run":
            cli.add_argument("output", type=Path)
        if command == "export":
            cli.add_argument("--replay", type=Path, required=True)
    run = commands.choices["run"]
    run.add_argument("--plan", choices=("a", "b"), default="a")
    run.add_argument("--preset", choices=("micro", "tiny", "small", "medium"), default="micro")
    run.add_argument("--seed", type=int, default=904)
    run.add_argument("--support", type=int, choices=(0, 1, 2, 4, 8, 16), default=16)
    run.add_argument("--rounds", type=int, default=1)
    run.add_argument("--persistent", action="store_true")
    run.add_argument("--meta-steps", type=int, default=0)
    run.add_argument("--backbone-every", type=int, default=0)
    run.add_argument("--collector", choices=("sgd", "rls", "rank8", "horn"), default="sgd")
    run.add_argument("--depths", type=int, choices=(1, 2, 4, 8, 16), nargs="+", default=[2])
    run.add_argument("--arms", nargs="+", default=["sgd", "rls", "rank8", "retrieval", "frozen", "shuffled",
                                                  "extra_compute", "static", "auto8", "horn"])
    run.add_argument("--final-only", action="store_true")
    run.add_argument("--cpu-seconds", type=float, default=2.0)
    run.add_argument("--max-macs", type=int, default=0, help="ranker MAC allowance per proof; 0 disables")
    run.add_argument("--timeout", type=int, default=30, help="native Lean RPC/certification timeout")
    run.add_argument("--steps", type=int, default=0, help="ledger entries per invocation; 0 runs to completion")
    run.add_argument("--audit", type=Path, help="write memory and completion metrics as JSON")
    return root


def main():
    args = parser().parse_args()
    if not 1 <= args.threads <= 64:
        raise SystemExit("--threads must be in 1..64")
    if args.command == "freeze":
        # The curriculum generator and its provenance helpers do not use model weights.
        from .ram_curriculum import freeze

        freeze(args.curriculum, args.seed, args.families, args.queries, args.rounds)
        return
    budget = TreeBudget(args.ram, args.threads)
    try:
        from . import ram_campaign, ram_store
        from .ram_learning import ARMS

        if args.command == "run":
            if (not set(args.arms) <= set(ARMS) or len(args.arms) != len(set(args.arms))
                    or len(args.depths) != len(set(args.depths))
                    or not 1 <= args.rounds <= 16 or not 0 < args.cpu_seconds <= 300
                    or min(args.meta_steps, args.backbone_every, args.steps, args.max_macs) < 0
                    or not 1 <= args.timeout <= 300
                    or (args.plan == "a" and set(args.arms) & {"zero_messages", "rolled_messages"})):
                raise SystemExit("invalid experiment arms or limits")
            options = {key: value for key, value in vars(args).items() if key not in
                       {"command", "session", "curriculum", "audit", "ram", "threads", "steps"}}
            ram_campaign.configure(args.threads, args.seed)
            result = ram_campaign.run(args.session, args.curriculum, budget, options, args.steps)
            if args.audit:
                args.audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        else:
            from .ram_results import report, export

            with ram_store.open_store(args.session) as db:
                seed = ram_store.read_manifest(db)["options"]["seed"]
                ram_campaign.configure(args.threads, seed)
                if args.command == "replay":
                    replay = ram_campaign.replay(db, args.curriculum, budget, args.output)
                    result = {k: v for k, v in replay.items() if k != "verified_episodes"}
                    result["verified_episodes"] = len(replay["verified_episodes"])
                elif args.command == "report":
                    result = report(db, args.curriculum, args.output)
                else:
                    result = export(db, args.curriculum, args.replay, args.output)
        print(json.dumps(result, indent=2))
    finally:
        budget.close()


if __name__ == "__main__":
    main()
