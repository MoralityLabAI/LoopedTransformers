"""Freeze and run capped, verifier-driven hybrid versus TRM development trials."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from ..data import canonical, digest, sha256_file
from ..ram_limits import TreeBudget
from . import corpus


def certify(case, episode, timeout=15):
    from lcgrt import config
    from lcgrt.certify import TACTIC_RE

    script = episode["accepted_script"]
    if not episode["live_kernel"] or not script or any(not TACTIC_RE.fullmatch(tactic) for tactic in script):
        return None
    source = (f"theorem {case.tid} : {case.statement} := by\n  intro "
              + " ".join(name for field in case.binders.split(") (")
                         for name in field.lstrip("(").split(" : ", 1)[0].split())
              + "\n" + "".join(f"  {tactic}\n" for tactic in script)
              + f"#print axioms {case.tid}\n")
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "Certificate.lean"
        path.write_text(source, encoding="utf-8")
        proc = subprocess.run([str(config.LEAN_EXE), str(path)], timeout=timeout,
                              capture_output=True, text=True, encoding="utf-8", env=config.lean_env())
    passed = proc.returncode == 0 and f"'{case.tid}' does not depend on any axioms" in proc.stdout.splitlines()
    return {"source": source, "source_digest": digest(source), "passed": passed,
            "axioms": [] if passed else None, "returncode": proc.returncode}


def rollout(repl, tree, case, policy, args, rng=None, intervention=None):
    import torch

    from ..live_search import search_episode

    passes = 4 if policy.architecture == "hybrid" else 1
    macs = 0
    start = tree.cpu_seconds()

    def within():
        return tree.cpu_seconds() - start < args.cpu_seconds

    def rank(packet, menu, _):
        nonlocal macs
        if len(menu) == 1:
            return 0, False
        need = policy.macs(len(menu), passes)
        if macs + need > args.max_macs:
            return None, False
        macs += need
        logits, _ = policy(packet, menu, passes, intervention=intervention)
        if rng is None:
            return int(logits.argmax()), True
        return int(torch.multinomial(logits.softmax(-1), 1, generator=rng)), True

    ep = search_episode(repl, case, None, policy=policy.architecture, depth=passes, check_budget=8,
                        rollback_budget=8, ranker=rank, within_budget=within)
    cpu = tree.cpu_seconds() - start
    cert = certify(case, ep)
    ep.update(certificate=cert, verified_closed=bool(cert and cert["passed"]), model_macs=macs,
              search_cpu_seconds=cpu, within_cpu_budget=cpu <= args.cpu_seconds)
    return ep


def reinforce(policy, episodes, optimizer):
    """Actor-critic credit from the realized Lean proof return and tactic cost."""
    import torch
    from torch.nn import functional as F

    terms, returns = [], []
    for ep in episodes:
        if ep.get("reward_censored"):
            continue
        completion = float((ep["verified_closed"] or ep.get("archived_closure", False)) and ep["within_cpu_budget"])
        returns.append(completion - 0.025 * ep["tactic_checks"])
        for event in ep["events"]:
            if event["kind"] != "tactic" or len(event["candidate_menu"]) == 1:
                continue
            logits, value = policy(canonical(event["input"]), event["candidate_menu"], ep["depth"])
            dist = torch.distributions.Categorical(logits=logits)
            target = value.new_tensor(completion - 0.025 * (ep["tactic_checks"] - event["attempt"] + 1))
            selected = logits.new_tensor(event["selected_index"], dtype=torch.long)
            terms.append(-dist.log_prob(selected) * (target - value.detach())
                         + 0.5 * F.smooth_l1_loss(value, target) - 0.01 * dist.entropy())
    if not terms:
        return {"mean_return": sum(returns) / len(returns) if returns else None,
                "decisions": 0, "loss": None}
    loss = torch.stack(terms).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite proof policy loss")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"mean_return": sum(returns) / len(returns), "decisions": len(terms), "loss": float(loss.detach())}


def imitate(policy, episodes, optimizer):
    """Compute-matched supervision on the surviving paths of compiled rollouts."""
    import torch
    from torch.nn import functional as F

    from ..ram_learning import credit

    terms = []
    for ep in episodes:
        if not (ep["verified_closed"] and ep["within_cpu_budget"]):
            continue
        for index, (label, _) in credit(ep):
            event = ep["events"][index]
            if label == 1 and len(event["candidate_menu"]) > 1:
                logits, _ = policy(canonical(event["input"]), event["candidate_menu"], ep["depth"])
                target = logits.new_tensor([event["selected_index"]], dtype=torch.long)
                terms.append(F.cross_entropy(logits[None], target))
    if not terms:
        return {"decisions": 0, "loss": None}
    loss = torch.stack(terms).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"decisions": len(terms), "loss": float(loss.detach())}


def _training_order(cases, seed):
    by_source = {}
    for case in cases:
        by_source.setdefault(case.source, []).append(case)
    rng = random.Random(seed)
    for group in by_source.values():
        rng.shuffle(group)
    return [group[i] for i in range(max(map(len, by_source.values())))
            for group in by_source.values() if i < len(group)]


def trial(args, tree):
    import torch

    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl

    from .. import ram_store
    from ..live_lean import fingerprint
    from ..live_search import replay_search
    from ..ram_campaign import configure
    from ..ram_learning import tensor_hash
    from .mathcan import ARCHIVE_SHA, MathCanReplay
    from .model import ProofPolicy

    configure(tree.threads, args.seed)
    cases = corpus.load(args.cohort)
    train = _training_order(cases["train"], args.seed)
    train_count = min(args.updates * args.batch, len(train))
    start = [("initial", case, None) for case in cases["dev"]]
    updates = [("train", train[i:i + args.batch], None) for i in range(0, train_count, args.batch)]
    mathcan = MathCanReplay(args.mathcan_archive) if args.mathcan_archive else None
    if mathcan is not None:
        if args.objective != "rl":
            raise ValueError("archived effect replay is reserved for reinforcement trials")
        order = random.Random(args.seed).sample(mathcan.starts, len(mathcan.starts))
        for index, key in enumerate(order[:len(updates) // 4]):
            updates.insert(5 * index + 4, ("mathcan", key, None))
    finish = [("final", case, None) for case in cases["dev"]]
    interventions = [(kind, case, (kind, 2)) for kind in ("zero", "sham") for case in cases["dev"]
                     if args.architecture == "hybrid" and args.interventions]
    tasks = start + updates + finish + interventions
    if args.graph_context:
        from .graph_model import GraphProofPolicy

        model = GraphProofPolicy(args.architecture, args.width)
    else:
        model = ProofPolicy(args.architecture, args.width)
    if args.head_only:
        model.requires_grad_(False)
        model.actor.requires_grad_(True)
        model.critic.requires_grad_(True)
        if args.graph_context:
            model.graph.action.requires_grad_(True)
            model.graph.value.requires_grad_(True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    train_step = reinforce if args.objective == "rl" else imitate
    rng = torch.Generator().manual_seed(args.seed)
    with LeanRepl(timeout=15) as repl, ram_store.open_store(args.session, write=True) as db:
        if repl.version != "4.19.0":
            raise ValueError("this cohort uses the pinned native Lean 4.19 verifier")
        manifest = {"schema": "ppt.hybrid_rl.trial.v1", "architecture": args.architecture,
                    "parameters": model.parameter_count(), "width": args.width, "seed": args.seed,
                    "optimizer_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "head_only": args.head_only, "objective": args.objective, "graph_context": args.graph_context,
                    "mathcan_archive_sha256": ARCHIVE_SHA if mathcan else None,
                    "cohort": sha256_file(args.cohort), "entry_count": len(tasks), "rollouts": args.rollouts,
                    "batch": args.batch, "lr": args.lr, "max_macs": args.max_macs, "cpu_seconds": args.cpu_seconds,
                    "train_count": train_count, "interventions": args.interventions,
                    "threads": tree.threads, "cap": tree.limit, "torch": str(torch.__version__),
                    "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    "verifier": fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)}
        ram_store.initialize(db, manifest)
        with db:
            db.execute("BEGIN IMMEDIATE")
            position = db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
            state = ram_store.checkpoint(db)
            if state is not None:
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                rng.set_state(state["rng"])
            end = min(len(tasks), position + args.steps) if args.steps else len(tasks)
            before = tensor_hash(model.state_dict())
            for phase, group, edit in tasks[position:end]:
                ep = []
                if phase in ("train", "mathcan"):
                    if phase == "mathcan":
                        ep = [mathcan.rollout(group, model, rng, args) for _ in range(args.rollouts)]
                    else:
                        for case in group:
                            for _ in range(args.rollouts):
                                ep.append(rollout(repl, tree, case, model, args, rng=rng))
                    update = train_step(model, ep, optimizer)
                    after = tensor_hash(model.state_dict())
                else:
                    ep.append(rollout(repl, tree, group, model, args, intervention=edit))
                    # Replay all development trajectories, including failures and rollback.
                    ep[0]["native_replay"] = replay_search(repl, group, ep[0])
                    update, after = None, before
                row = {"phase": phase, "episodes": ep, "update": update,
                       "weights_before": before, "weights_after": after}
                saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng.get_state()}
                ram_store.append(db, row, saved if update is not None or position == 0 else None)
                db.commit()
                before = after
                position += 1
                if position < end:
                    db.execute("BEGIN IMMEDIATE")
                    if db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0] != position:
                        raise RuntimeError("session was advanced by another process")
        metrics = {}
        for row in ram_store.ledger(db):
            for ep in row["episodes"]:
                name = f"{row['phase']}/{ep['family']}"
                counts = metrics.setdefault(name, Counter())
                counts["episodes"] += 1
                counts["certified"] += int(ep["verified_closed"] and ep["within_cpu_budget"])
                if ep.get("source") == "mathcan_ram08_audited_train":
                    counts["archived_closures"] += int(ep["archived_closure"])
                    counts["censored_reward"] += int(ep["reward_censored"])
                counts["tactic_checks"] += ep["tactic_checks"]
            if row["update"]:
                metrics.setdefault("policy_updates", Counter()).update(
                    decisions=row["update"]["decisions"],
                    changed_parameters=row["weights_before"] != row["weights_after"])
    return {"manifest": manifest, "complete": position == len(tasks), "metrics": metrics, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="command", required=True)
    freeze = cmds.add_parser("freeze")
    freeze.add_argument("cohort", type=Path)
    freeze.add_argument("--seed", type=int, default=24011)
    freeze.add_argument("--train", type=int, default=48)
    freeze.add_argument("--dev", type=int, default=12)
    run = cmds.add_parser("run")
    run.add_argument("cohort", type=Path)
    run.add_argument("session", type=Path)
    run.add_argument("--architecture", choices=("hybrid", "trm"), required=True)
    run.add_argument("--objective", choices=("rl", "imitation"), default="rl")
    run.add_argument("--head-only", action="store_true")
    run.add_argument("--graph-context", action="store_true",
                     help="supply the same visible premise graph to the hybrid or TRM baseline")
    run.add_argument("--mathcan-archive", type=Path)
    run.add_argument("--report", type=Path)
    run.add_argument("--width", type=int, default=256)
    run.add_argument("--seed", type=int, default=21)
    run.add_argument("--updates", type=int, default=16)
    run.add_argument("--batch", type=int, default=2)
    run.add_argument("--rollouts", type=int, default=2)
    run.add_argument("--steps", type=int, default=0)
    run.add_argument("--lr", type=float, default=0.0003)
    run.add_argument("--max-macs", type=int, default=1_200_000_000)
    run.add_argument("--cpu-seconds", type=float, default=6)
    run.add_argument("--ram", type=int, choices=(2, 4, 6), default=6)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--interventions", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        if not 2 <= args.train <= 2048 or not 2 <= args.dev <= 512:
            parser.error("expected a bounded development cohort")
        corpus.freeze(args.cohort, args.seed, args.train, args.dev)
        return
    if not (1 <= args.threads <= 16 and 1 <= args.batch <= 16 and 1 <= args.rollouts <= 8
            and 1 <= args.updates <= 2048 and args.steps >= 0 and 0 < args.lr <= 0.1
            and args.max_macs > 0 and 0 < args.cpu_seconds <= 60):
        parser.error("invalid training or resource budget")
    budget = TreeBudget(args.ram, args.threads)
    try:
        result = trial(args, budget)
        if args.report:
            args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        budget.close()


if __name__ == "__main__":
    main()
