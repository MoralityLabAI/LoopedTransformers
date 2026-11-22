"""Freeze and run capped, verifier-driven hybrid versus TRM development trials."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from collections import Counter
from functools import partial
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


def rollout(repl, tree, case, policy, args, rng=None, intervention=None, first_choice=None):
    import torch

    from ..live_search import search_episode

    passes = 4 if policy.architecture == "hybrid" else 1
    macs = 0
    start = tree.cpu_seconds()

    def within():
        return tree.cpu_seconds() - start < args.cpu_seconds

    def rank(packet, menu, remaining):
        nonlocal macs
        if len(menu) == 1:
            return 0, False
        need = policy.macs(len(menu), passes)
        if macs + need > args.max_macs:
            return None, False
        macs += need
        logits, _ = policy(packet, menu, passes, intervention=intervention)
        if first_choice is not None and remaining == 8:
            if len(menu) != 2:
                raise ValueError("paired proposal requires exactly two root actions")
            return first_choice, True
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


def proof_return(episode):
    completion = float((episode["verified_closed"] or episode.get("archived_closure", False))
                       and episode["within_cpu_budget"])
    return completion - 0.025 * episode["tactic_checks"]


def actor_critic_terms(policy, ep, terminal_return, *, after_root=False):
    import torch
    from torch.nn import functional as F

    for event in ep["events"]:
        if (event["kind"] != "tactic" or len(event["candidate_menu"]) == 1
                or (after_root and event["attempt"] == 1)):
            continue
        logits, value = policy(canonical(event["input"]), event["candidate_menu"], ep["depth"])
        dist = torch.distributions.Categorical(logits=logits)
        target = value.new_tensor(terminal_return + 0.025 * (event["attempt"] - 1))
        selected = logits.new_tensor(event["selected_index"], dtype=torch.long)
        log_prob = dist.log_prob(selected)
        proposal = ep.get("root_proposal_probability") if event["attempt"] == 1 else None
        importance = log_prob.detach().exp() / proposal if proposal is not None else 1.0
        yield (-importance * log_prob * (target - value.detach())
               + 0.5 * F.smooth_l1_loss(value, target) - 0.01 * dist.entropy())


def policy_step(policy, optimizer, loss, returns, decisions):
    import torch

    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite proof policy loss")
    if not loss.requires_grad:
        return {"mean_return": sum(returns) / len(returns), "decisions": decisions, "loss": float(loss)}
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"mean_return": sum(returns) / len(returns), "decisions": decisions, "loss": float(loss.detach())}


def reinforce(policy, episodes, optimizer):
    """Actor-critic credit from the realized Lean proof return and tactic cost."""
    import torch

    terms, returns = [], []
    for ep in episodes:
        if ep.get("reward_censored"):
            continue
        terminal_return = proof_return(ep)
        returns.append(terminal_return)
        terms.extend(actor_critic_terms(policy, ep, terminal_return))
    if not terms:
        return {"mean_return": sum(returns) / len(returns) if returns else None,
                "decisions": 0, "loss": None}
    return policy_step(policy, optimizer, torch.stack(terms).mean(), returns, len(terms))


def paired_credit(policy, episodes, optimizer, *, continue_search=False):
    """Credit the response to a verified fact switch, holding the proposed action fixed."""
    import torch

    terms, continuations, returns = [], [], []
    for left, right in zip(episodes[::2], episodes[1::2], strict=True):
        if left.get("reward_censored") or right.get("reward_censored"):
            continue
        return_a, return_b = proof_return(left), proof_return(right)
        returns.extend((return_a, return_b))
        a, b = left["events"][0], right["events"][0]
        q = left["root_proposal_probability"]
        if (a["candidate_menu"] != b["candidate_menu"] or a["selected_index"] != b["selected_index"]
                or q != right["root_proposal_probability"]):
            raise ValueError("paired fact credit requires the same sampled action in both worlds")
        probabilities = []
        for ep, event in ((left, a), (right, b)):
            logits, _ = policy(canonical(event["input"]), event["candidate_menu"], ep["depth"])
            probabilities.append(logits.softmax(-1)[event["selected_index"]])
        # This is the fact-sensitive part of the paired return, with the
        # uniform proposal corrected explicitly.
        terms.append(-(return_a - return_b) * (probabilities[0] - probabilities[1]) / (4 * q))
        if continue_search:
            continuations.extend(actor_critic_terms(policy, left, return_a, after_root=True))
            continuations.extend(actor_critic_terms(policy, right, return_b, after_root=True))
    if not terms:
        return {"mean_return": None, "decisions": 0, "loss": None}
    loss = torch.stack(terms).mean()
    if continuations:
        loss = loss + torch.stack(continuations).mean()
    return policy_step(policy, optimizer, loss, returns, len(returns) + len(continuations))


def paired_alias_rollouts(repl, tree, group, model, args, rng):
    import torch

    episodes = []
    balanced = args.alias_training == "contrast-search"
    offset = int(digest([args.seed, group[0].cluster, "root-proposal"])[0], 16) % 2 if balanced else 0
    for attempt in range(args.rollouts):
        choice = (offset + attempt) % 2 if balanced else int(torch.randint(2, (1,), generator=rng))
        pair = [rollout(repl, tree, case, model, args, rng=rng, first_choice=choice) for case in group]
        valid = all(ep["events"] and ep["within_cpu_budget"] for ep in pair)
        if valid:
            a, b = (ep["events"][0] for ep in pair)
            if (len(a["candidate_menu"]) != 2 or a["candidate_menu"] != b["candidate_menu"]
                    or a["selected_index"] != choice or b["selected_index"] != choice):
                raise ValueError("paired worlds must execute the same uniformly proposed root action")
        for ep in pair:
            ep["root_proposal_probability"] = 0.5
            if not valid:
                ep["reward_censored"] = True
        episodes.extend(pair)
    return episodes


def imitate(policy, episodes, optimizer):
    """Compute-matched supervision on the surviving paths of compiled rollouts."""
    import torch
    from torch.nn import functional as F

    from ..ram_learning import credit

    terms = []
    for ep in episodes:
        if ep.get("reward_censored") or not ep["within_cpu_budget"]:
            continue
        if ep.get("archived_closure"):
            # The archive has a linear receipt path with tactic rejections and no rollback.
            labels = [(i, 1) for i, event in enumerate(ep["events"])
                      if event["status"] in ("accepted_nonterminal", "archived_closure")]
        else:
            labels = [(i, label) for i, (label, _) in credit(ep)]
        for index, label in labels:
            event = ep["events"][index]
            if label == 1 and len(event["candidate_menu"]) > 1:
                logits, _ = policy(canonical(event["input"]), event["candidate_menu"], ep["depth"])
                target = logits.new_tensor([event["selected_index"]], dtype=torch.long)
                terms.append(F.cross_entropy(logits[None], target))
    if not terms:
        return {"decisions": 0, "loss": None}
    loss = torch.stack(terms).mean()
    if loss.requires_grad:
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


def _paired_training_batches(cases, seed):
    clusters, other = {}, []
    for case in cases:
        if case.family in corpus.PAIRED_FAMILIES:
            clusters.setdefault(case.cluster, []).append(case)
        else:
            other.append(case)
    if not clusters or any(len(pair) != 2 for pair in clusters.values()):
        raise ValueError("paired training needs both alias worlds in each cluster")
    pairs = list(clusters.values())
    rng = random.Random(seed)
    rng.shuffle(pairs)
    rng.shuffle(other)
    sources = pairs, [other[i:i + 2] for i in range(0, len(other), 2)]
    return [group[i] for i in range(max(map(len, sources))) for group in sources if i < len(group)]


def initialize_from_imitation(model, args, initialization_cohort):
    from .. import ram_store
    from ..ram_learning import tensor_hash

    with ram_store.open_store(args.initialize_from) as source:
        manifest = ram_store.read_manifest(source)
        if (manifest["architecture"] != "hybrid" or manifest["objective"] != "imitation"
                or manifest["seed"] != args.seed or manifest["width"] != args.width
                or manifest["cohort"] != initialization_cohort or not manifest["premise_interaction"]
                or manifest.get("root_fact_adapter", False) or manifest["head_only"]):
            raise ValueError("expected the matched full imitation initialization checkpoint")
        rows = ram_store.ledger(source)
        count, last = 0, None
        for last in rows:
            count += 1
        weights = ram_store.checkpoint(source)["model"]
    weight_hash = tensor_hash(weights)
    if last is None or count != manifest["entry_count"] or last["phase"] != "final" or last["weights_after"] != weight_hash:
        raise ValueError("imitation initialization did not complete its frozen evaluation")
    if args.root_fact_adapter:
        copied = {"graph.root_transport." + name: weights["graph.transport." + name]
                  for name in model.graph.transport.state_dict()}
        copied["graph.root_gain"] = model.graph.root_gain.detach()
        weights = {**weights, **copied}
    model.load_state_dict(weights)
    return {"session_sha256": sha256_file(args.initialize_from), "imitation_weights_sha256": weight_hash}


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
    if args.alias_training == "standard":
        train = _training_order(cases["train"], args.seed)
        batches = [train[i:i + args.batch] for i in range(0, len(train), args.batch)]
    else:
        batches = _paired_training_batches(cases["train"], args.seed)
    batches = batches[:args.updates]
    train_count = sum(map(len, batches))
    start = [("initial", case, None) for case in cases["dev"]]
    updates = [("train", batch, None) for batch in batches]
    mathcan = MathCanReplay(args.mathcan_archive) if args.mathcan_archive else None
    if mathcan is not None:
        if args.objective != "rl" and args.alias_training != "contrast-search":
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

        model = GraphProofPolicy(args.architecture, args.width, residual_transport=args.residual_fact_transport,
                                 premise_interaction=args.premise_interaction, root_fact_adapter=args.root_fact_adapter)
    else:
        model = ProofPolicy(args.architecture, args.width)
    initialization = None
    if args.initialize_from:
        cohort_reg = json.loads(args.cohort.read_bytes())["registration"]
        initialization = initialize_from_imitation(
            model, args, cohort_reg.get("initialization_cohort_sha256", sha256_file(args.cohort))
        )
    if args.root_fact_adapter:
        model.requires_grad_(False)
        model.graph.root_gain.requires_grad_(True)
        trainable_route = model.graph.root_transport.action if args.head_only else model.graph.root_transport
        trainable_route.requires_grad_(True)
    elif args.head_only:
        model.requires_grad_(False)
        model.actor.requires_grad_(True)
        model.critic.requires_grad_(True)
        if args.graph_context:
            model.graph.action.requires_grad_(True)
            model.graph.value.requires_grad_(True)
            if model.graph.transport is not None:
                model.graph.transport.action.requires_grad_(True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    train_step = reinforce if args.objective == "rl" else imitate
    paired_step = (partial(paired_credit, continue_search=args.alias_training == "contrast-search")
                   if args.objective == "rl" and args.alias_training in ("contrast", "contrast-search")
                   else train_step)
    rng = torch.Generator().manual_seed(args.seed)
    with LeanRepl(timeout=15) as repl, ram_store.open_store(args.session, write=True) as db:
        if repl.version != "4.19.0":
            raise ValueError("this cohort uses the pinned native Lean 4.19 verifier")
        manifest = {"schema": "ppt.hybrid_rl.trial.v1", "architecture": args.architecture,
                    "parameters": model.parameter_count(), "width": args.width, "seed": args.seed,
                    "optimizer_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "head_only": args.head_only, "objective": args.objective, "graph_context": args.graph_context,
                    "residual_fact_transport": args.residual_fact_transport,
                    "premise_interaction": args.premise_interaction,
                    "alias_training": args.alias_training,
                    "mathcan_archive_sha256": ARCHIVE_SHA if mathcan else None,
                    "cohort": sha256_file(args.cohort), "entry_count": len(tasks), "rollouts": args.rollouts,
                    "batch": args.batch, "lr": args.lr, "max_macs": args.max_macs, "cpu_seconds": args.cpu_seconds,
                    "train_count": train_count, "interventions": args.interventions,
                    "threads": tree.threads, "cap": tree.limit, "torch": str(torch.__version__),
                    "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    "verifier": fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)}
        if initialization is not None:
            manifest.update(initialization=initialization, root_fact_adapter=args.root_fact_adapter)
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
                    step = train_step
                    if phase == "mathcan":
                        ep = [mathcan.rollout(group, model, rng, args) for _ in range(args.rollouts)]
                    elif args.alias_training != "standard" and group[0].family in corpus.PAIRED_FAMILIES:
                        ep = paired_alias_rollouts(repl, tree, group, model, args, rng)
                        step = paired_step
                    else:
                        for case in group:
                            for _ in range(args.rollouts):
                                ep.append(rollout(repl, tree, case, model, args, rng=rng))
                    update = step(model, ep, optimizer)
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
    run.add_argument("--residual-fact-transport", action="store_true",
                     help="add the hybrid's long-range evidence channels to the visible premise graph")
    run.add_argument("--premise-interaction", action="store_true",
                     help="learn to route evidence conditioned on which premises share a rule")
    run.add_argument("--root-fact-adapter", action="store_true",
                     help="adapt independent first-action fact transport while preserving continuation weights")
    run.add_argument("--initialize-from", type=Path, help="start from a completed matched imitation checkpoint")
    run.add_argument("--alias-training", choices=("standard", "paired", "contrast", "contrast-search"),
                     default="standard")
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
    if args.residual_fact_transport and (not args.graph_context or args.architecture != "hybrid"):
        parser.error("--residual-fact-transport requires --graph-context and --architecture hybrid")
    if args.premise_interaction and not args.residual_fact_transport:
        parser.error("--premise-interaction requires --residual-fact-transport")
    if args.initialize_from and not args.premise_interaction:
        parser.error("--initialize-from requires --premise-interaction")
    if args.root_fact_adapter and not args.initialize_from:
        parser.error("--root-fact-adapter requires --initialize-from")
    if args.alias_training != "standard" and args.batch != 2:
        parser.error("paired alias variants require a batch of two worlds")
    if args.alias_training in ("paired", "contrast") and args.objective != "rl":
        parser.error("paired and contrast training require reinforcement learning")
    if args.alias_training == "contrast-search" and args.rollouts != 2:
        parser.error("contrast-search gives each world exactly two opposing root proposals")
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
