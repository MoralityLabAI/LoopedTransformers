"""Fit a collective rewrite bank against native feedback and matched controls."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from configs.check_ppt_bpd_shared_conjuncts import BUDGET, SCHEMA, STRESS_SCHEMA, layouts, stress_layouts, world
from configs.probe_ppt_bpd_dependent_controls import fit_calibrated_margin
from configs.probe_ppt_bpd_dependent_policy import eligible_rewrites, trajectory
from configs.probe_ppt_bpd_dependent_routing import load_routed_checkpoint, score_native
from configs.probe_ppt_bpd_mixed_policy import fit as fit_matched
from configs.probe_ppt_bpd_rewrite_revisits import audited_policies
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


def read_cohort(path, split):
    from ppt_gym.conjunctive_rewrite import proposal_menu

    witness = json.loads(gzip.decompress(path.read_bytes()))
    schema = witness["schema"]
    stress = schema == STRESS_SCHEMA
    if witness["split"] != split or witness["tactic_checks"] != BUDGET or schema not in (SCHEMA, STRESS_SCHEMA):
        raise ValueError("unexpected shared-rewrite witness identity or budget")
    if stress and split != "dev":
        raise ValueError("the decoy layouts are reserved for evaluation")
    expected = set(stress_layouts() if stress else layouts()[split])
    groups = defaultdict(set)
    for root, row in witness["rows"].items():
        shape = tuple(row["shape"])
        winner = row["winner"]
        case, _, _, _ = world(split, shape, winner, stress=stress)
        if case != SearchCase(**row["case"]) or case.root != root or winner not in (0, 1) or len(row["branches"]) != 2:
            raise ValueError("the shared-rewrite declaration differs from the registered layout")
        groups[shape].add(winner)
        first = row["branches"][0]["events"][0]
        row["packet"], row["menu"] = first["packet"], first["menu"]
        for choice, branch in enumerate(row["branches"]):
            events, cert = branch["events"], branch["certificate"]
            if (branch["choice"] != choice or branch["sum_if_split_first"] != 9 or len(events) != branch["checks"]
                    or len(events) != BUDGET + int(choice != winner) or bool(events[BUDGET - 1]["targets"]) != (choice != winner)
                    or not cert["passed"] or cert["axioms"] or digest(cert["source"]) != cert["source_digest"]):
                raise ValueError("a registered shared guard lost its proof or deadline receipt")
            branch["closed"] = choice == winner
            if events[0]["packet"] != row["packet"] or events[0]["menu"] != row["menu"] or events[0]["tactic"] != row["menu"][choice]:
                raise ValueError("the two proofs disagree about the registered opening observation")
            for attempt, event in enumerate(events):
                if "packet" not in event:
                    if split == "train":
                        raise ValueError("a certified training tactic lacks its observation")
                    continue
                packet = json.loads(event["packet"])
                if (packet["checks_remaining"] != max(0, BUDGET - attempt) or digest(packet["state"]) != event["before"]
                        or tuple(event["menu"]) != proposal_menu(packet["state"]) or event["tactic"] not in event["menu"]):
                    raise ValueError("a public tactic menu no longer matches its verifier receipt")
    if expected != set(groups) or any(winners != {0, 1} for winners in groups.values()) or len(witness["rows"]) != 2 * len(groups):
        raise ValueError("the shared-rewrite test lost an independent layout pair")
    return witness


def batches(training):
    grouped = defaultdict(list)
    for root, row in training["rows"].items():
        for branch in row["branches"]:
            events, previous = [], None
            for attempt, step in enumerate(branch["events"][:BUDGET], 1):
                menu, tactic = step["menu"], step["tactic"]
                eligible = eligible_rewrites(menu, previous)
                if tactic not in eligible:
                    raise ValueError("a demonstrated tactic disagrees with the live action restriction")
                events.append({"input": json.loads(step["packet"]), "candidate_menu": menu, "attempt": attempt,
                               "selected_index": menu.index(tactic), "rewrite_choices": [menu.index(action) for action in eligible],
                               "kind": "tactic", "status": "accepted_nonterminal"})
                previous = tactic
            grouped[row["case"]["cluster"]].append({"root": root, "family": row["case"]["family"], "events": events,
                                                   "tactic_checks": BUDGET, "within_cpu_budget": True,
                                                   "verified_closed": branch["closed"]})
    return [grouped[key] for key in sorted(grouped)]


def model(source):
    from ppt_gym.shared_rewrite import SharedRewritePolicy

    _, checkpoint = load_routed_checkpoint(source)
    policy = SharedRewritePolicy(**checkpoint["architecture"]).eval().requires_grad_(False)
    missing, unexpected = policy.load_state_dict(checkpoint["state_dict"], strict=False)
    if set(missing) != {"shared_transport." + name for name, _ in policy.shared_transport.named_parameters()} or unexpected:
        raise ValueError("the shared-rewrite fit requires the verified routed warm checkpoint")
    return policy, checkpoint["architecture"]


def fit_trm(args, tree):
    from ppt_gym.ram_learning import tensor_hash
    from ppt_gym.shared_rewrite import ConjunctiveGraphPolicy

    training = read_cohort(args.training, "train")
    audit = json.loads(args.audit.read_bytes())
    _, warm, identity, _ = next(audited_policies(args.sessions, args.audit, audit["cohort_sha256"],
                                                audit["reference_sha256"], selected=("trm_rl",)))
    policy = ConjunctiveGraphPolicy("trm", warm.width).eval()
    policy.load_state_dict(warm.state_dict())
    result = fit_matched(policy, "trm_rl", batches(training), {}, False, 1)
    save_policy(policy, args.checkpoint, {"architecture": "trm", "width": warm.width})
    return {"training_sha256": sha256_file(args.training), "checkpoint_sha256": sha256_file(args.checkpoint),
            "weights_sha256": tensor_hash(policy.state_dict()), "warm": identity, "training": result, "memory": tree.stats()}


def save_policy(policy, path, architecture):
    import torch

    from ppt_gym.ram_learning import tensor_hash

    state = policy.state_dict()
    with path.open("xb") as handle:
        torch.save({"architecture": architecture, "state_dict": state, "weights_sha256": tensor_hash(state)}, handle)


def live_returns(repl, policy, row, tree):
    import torch

    from ppt_gym.conjunctive_rewrite import proposal_menu
    from ppt_gym.hybrid_rl.__main__ import certify, proof_return

    case = SearchCase(**row["case"])
    receipt = digest(json.loads(row["packet"])["state"])
    outcomes = []
    with torch.no_grad():
        for opening in row["menu"]:
            events, closed, within, macs = trajectory(repl, case, policy, tree, opening=opening, proposals=proposal_menu)
            if not within or not events or events[0]["before"] != receipt or events[0]["tactic"] != opening:
                raise ValueError("a live shared-rewrite rollout lost its time budget or exact opening state")
            script = [entry["tactic"] for entry in events]
            replay, repeated, _, _ = trajectory(repl, case, policy, tree, script, proposals=proposal_menu)
            cert = certify(case, {"live_kernel": closed, "accepted_script": script})
            if replay != events or repeated != closed or (closed and (not cert or not cert["passed"] or cert["axioms"]
                                                                    or digest(cert["source"]) != cert["source_digest"])):
                raise ValueError("a live shared-rewrite rollout failed replay or independent certification")
            reward = proof_return({"verified_closed": closed, "within_cpu_budget": within, "tactic_checks": len(events)})
            outcomes.append({"events": events, "reward": reward, "certificate": cert, "closed": closed, "macs": macs})
    return outcomes


def scores(policy, cache, packet, menu):
    import torch

    key = packet, tuple(menu)
    if key not in cache:
        with torch.no_grad():
            base, _, view = policy.replay_inputs(packet, menu, 4)
        cache[key] = base, view
    base, view = cache[key]
    return policy.decision_scores(base, view, menu, 4)


def train_arm(policy, training, grouped, tree, *, repl=None):
    import torch

    from ppt_gym.ram_learning import tensor_hash

    policy.shared_transport.requires_grad_(True)
    frozen = {key: value for key, value in policy.state_dict().items() if not key.startswith("shared_transport.")}
    before = tensor_hash(frozen)
    initial = tensor_hash(policy.shared_transport.state_dict())
    optimizer = torch.optim.Adam(policy.shared_transport.parameters(), lr=.08)
    cache, traces, decisions = {}, {}, 0
    for number, group in enumerate(grouped, 1):
        terms = []
        if repl is not None:
            for root in dict.fromkeys(episode["root"] for episode in group):
                row = training["rows"][root]
                outcomes = live_returns(repl, policy, row, tree)
                logits = scores(policy, cache, row["packet"], row["menu"])
                probabilities = (logits / .15).softmax(-1)
                rewards = logits.new_tensor([trace["reward"] for trace in outcomes])
                terms.append(-(probabilities * (rewards - rewards.mean())).sum())
                traces[root] = {"probabilities": probabilities.detach().tolist(), "returns": outcomes}
        else:
            for episode in group:
                for step in episode["events"]:
                    choices = step["rewrite_choices"]
                    if len(choices) < 2:
                        continue
                    logits = scores(policy, cache, canonical(step["input"]), step["candidate_menu"])[choices] / .15
                    terms.append(-logits.log_softmax(0)[choices.index(step["selected_index"])])
        loss = torch.stack(terms).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("the joint rewrite return update became nonfinite")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.shared_transport.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        decisions += len(terms)
        if number % 8 == 0:
            print(f"Updated the shared transport after {number} layout pairs.", flush=True)
    if tensor_hash(frozen) != before or tensor_hash(policy.shared_transport.state_dict()) == initial:
        raise ValueError("the shared proof return fit failed to isolate changes to its new transport")
    return {"decisions": decisions, "native_feedback": traces,
            "parameters": {name: float(parameter.detach()) for name, parameter in policy.shared_transport.named_parameters()}}


def train(args, tree):
    from lcgrt.lean_repl import LeanRepl

    training = read_cohort(args.training, "train")
    grouped = batches(training)
    args.checkpoints.mkdir()
    results = {}
    for arm in ("live_rl", "imitation"):
        policy, architecture = model(args.warm)
        if arm == "live_rl":
            with LeanRepl(timeout=15) as repl:
                if repl.version != training["lean"]:
                    raise ValueError("the shared-rewrite fit requires its registered Lean verifier")
                record = train_arm(policy, training, grouped, tree, repl=repl)
        else:
            record = train_arm(policy, training, grouped, tree)
        path = args.checkpoints / (arm + ".pt")
        save_policy(policy, path, architecture)
        results[arm] = {**record, "checkpoint_sha256": sha256_file(path)}
        print(arm, record["parameters"], flush=True)
    return {"training_sha256": sha256_file(args.training), "warm_sha256": sha256_file(args.warm),
            "results": results, "memory": tree.stats()}


def calibrate_joint_scale(policy, training):
    import torch

    contrasts, targets, layouts = [], [], []
    with torch.no_grad():
        for row in training["rows"].values():
            menu = row["menu"]
            base, _, view = policy.replay_inputs(row["packet"], menu, 4)
            joint = policy.shared_transport(view, menu, 8).clamp_min(1e-8).log()
            contrasts.append(joint[0] - joint[1])
            targets.append((2. if row["winner"] == 0 else -2.) - (base[0] - base[1]))
            layouts.append(row["case"]["cluster"])
        x, y = torch.stack(contrasts), torch.stack(targets)
        scale = x.square().mean().clamp_min(1e-8)

        def solve(rows, penalty):
            return ((x[rows] * y[rows]).sum() / (x[rows].square().sum() + penalty * scale)).clamp_min(0.)

        losses = {}
        for penalty in (.01, .1, 1., 10.):
            errors = []
            for layout in sorted(set(layouts)):
                held = torch.tensor([other == layout for other in layouts])
                errors.append((y[held] - x[held] * solve(~held, penalty)).square().mean())
            losses[penalty] = float(torch.stack(errors).mean())
        selected = min(losses, key=losses.get)
        weight = solve(slice(None), selected)
        policy.shared_transport.readout_weight.copy_(weight)
    return {"weight": float(weight), "selected_penalty": selected, "held_layout_losses": {str(k): v for k, v in losses.items()}}


def evaluate(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.conjunctive_rewrite import proposal_menu
    from ppt_gym.ram_learning import tensor_hash
    from ppt_gym.shared_rewrite import ConjunctiveGraphPolicy

    training, witness = read_cohort(args.training, "train"), read_cohort(args.witness, "dev")
    runs = json.loads(gzip.decompress(args.fit.read_bytes()))
    trm_receipt = json.loads(gzip.decompress(args.trm_audit.read_bytes()))
    if (runs["training_sha256"] != sha256_file(args.training) or runs["warm_sha256"] != sha256_file(args.warm)
            or trm_receipt["training_sha256"] != runs["training_sha256"]
            or trm_receipt["checkpoint_sha256"] != sha256_file(args.trm)
            or set(training["rows"]) & set(witness["rows"])):
        raise ValueError("the held-out comparison changed a registered fit, model, or cohort")
    controls = {}
    for arm in ("live_rl", "imitation"):
        path = args.checkpoints / (arm + ".pt")
        if runs["results"][arm]["checkpoint_sha256"] != sha256_file(path):
            raise ValueError("the frozen shared-rewrite fit changed before scoring")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        policy, _ = model(args.warm)
        policy.load_state_dict(saved["state_dict"])
        if tensor_hash(policy.state_dict()) != saved["weights_sha256"]:
            raise ValueError("the shared-rewrite fit changed on CPU reload")
        controls[arm] = policy
    calibrated, _ = model(args.warm)
    ridge = fit_calibrated_margin(calibrated, training, additional_readouts=(calibrated.shared_transport.readout_weight,))
    scaled, _ = model(args.warm)
    scale = calibrate_joint_scale(scaled, training)
    fixed, _ = model(args.warm)
    independent, _ = model(args.warm)
    independent.factorized = True
    exact, _ = model(args.warm)
    exact.hard_shared = True
    trm_state = torch.load(args.trm, map_location="cpu", weights_only=True)
    trm = ConjunctiveGraphPolicy(**trm_state["architecture"]).eval()
    trm.load_state_dict(trm_state["state_dict"])
    controls.update(calibrated_shared=calibrated, calibrated_scale=scaled, frozen_shared=fixed,
                    independent_work=independent, graph_trm_rl=trm, max_product_joint=exact)
    results = {}
    with LeanRepl(timeout=15) as repl, torch.no_grad():
        if repl.version != witness["lean"]:
            raise ValueError("the shared-rewrite evaluation requires the registered Lean verifier")
        for arm, policy in controls.items():
            results[arm] = score_native(repl, policy, witness, tree, proposals=proposal_menu)
            print(arm, results[arm]["counts"], flush=True)
    return {"witness_sha256": sha256_file(args.witness), "fit_sha256": sha256_file(args.fit),
            "trm_audit_sha256": sha256_file(args.trm_audit), "ridge": ridge, "scale_calibration": scale,
            "results": results, "memory": tree.stats()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    stages = cli.add_subparsers(dest="stage", required=True)
    fit, score, trm = (stages.add_parser(name) for name in ("train", "evaluate", "trm"))
    for parser in (fit, score, trm):
        parser.add_argument("training", type=Path)
        parser.add_argument("output", type=Path)
    for parser in (fit, score):
        parser.add_argument("warm", type=Path)
        parser.add_argument("checkpoints", type=Path)
    for name in ("witness", "fit", "trm", "trm_audit"):
        score.add_argument(name, type=Path)
    for name in ("audit", "sessions", "checkpoint"):
        trm.add_argument(name, type=Path)
    args = cli.parse_args()
    tree = TreeBudget(2, 1)
    try:
        from ppt_gym.ram_campaign import configure

        configure(tree.threads, 103)
        result = {"train": train, "evaluate": evaluate, "trm": fit_trm}[args.stage](args, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(result).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
