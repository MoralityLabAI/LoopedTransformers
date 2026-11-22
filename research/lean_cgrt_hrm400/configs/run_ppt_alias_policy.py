"""Fit and compare policies that receive the same public local definitions."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from configs.check_ppt_bpd_shared_aliases import BUDGET, world
from configs.probe_ppt_bpd_dependent_policy import eligible_rewrites, trajectory
from configs.probe_ppt_bpd_shared_aliases import read_witness
from configs.run_ppt_shared_rewrite import save_policy
from ppt_gym.alias_rewrite import alias_menu
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


MAC_LIMIT = 2_400_000_000


def warm_policy(path, *, trm=False):
    import torch

    from ppt_gym.alias_policy import AliasAwarePolicy, AliasGraphPolicy
    from ppt_gym.ram_learning import tensor_hash

    package = torch.load(path, map_location="cpu", weights_only=True)
    if tensor_hash(package["state_dict"]) != package["weights_sha256"]:
        raise ValueError("the alias warm checkpoint lost its weights hash")
    policy_cls = AliasGraphPolicy if trm else AliasAwarePolicy
    policy = policy_cls(**package["architecture"]).eval()
    missing, unrecognized = policy.load_state_dict(package["state_dict"], strict=False)
    expected = set() if trm else {"normalization_logit", "preview_logit"}
    if set(missing) != expected or unrecognized:
        raise ValueError("the alias policy changed the registered inherited parameters")
    return policy, package["architecture"]


def read_fit(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def opening_menu(row):
    return tuple(next(iter(row["paths"].values()))["events"][0]["menu"])


def demonstrated_actions(row):
    for name, path in row["paths"].items():
        previous = None
        for event in path["events"][:BUDGET]:
            menu = tuple(event["menu"])
            eligible = eligible_rewrites(menu, previous)
            previous = event["tactic"]
            if previous not in eligible:
                raise ValueError("the recorded normalization step disagrees with the live action restriction")
            if len(eligible) > 1:
                yield event["packet"], menu, [menu.index(tactic) for tactic in eligible], eligible.index(previous), name


def run_search(repl, row, policy, tree, *, opening=None):
    from ppt_gym.hybrid_rl.__main__ import certify, proof_return

    case = SearchCase(**row["case"])
    events, closed, within, macs = trajectory(repl, case, policy, tree, opening=opening, proposals=alias_menu,
                                               mac_limit=MAC_LIMIT)
    if (not events or events[0]["before"] != digest(json.loads(row["packet"])["state"])
            or (opening is not None and events[0]["tactic"] != opening)):
        raise ValueError("the normalization rollout disagrees with its registered opening")
    script = [event["tactic"] for event in events]
    replay, repeated, _, _ = trajectory(repl, case, policy, tree, script, proposals=alias_menu)
    _, _, aliases, _, _, _ = world(case.split, tuple(row["shape"]), row["swapped"])
    cert = certify(case, {"live_kernel": closed, "accepted_script": script}, timeout=60, aliases=aliases)
    if replay != events or repeated != closed or (closed and (not cert or not cert["passed"] or cert["axioms"]
                                                            or digest(cert["source"]) != cert["source_digest"])):
        raise ValueError("the alias proof failed native replay or independent certification")
    reward = proof_return({"verified_closed": closed, "within_cpu_budget": within, "tactic_checks": len(events)})
    return {"events": events, "certificate": cert, "reward": reward, "closed": closed and within, "within": within, "macs": macs}


def cached_logits(policy, cache, packet, menu):
    import torch

    key = packet, menu
    if key not in cache:
        with torch.no_grad():
            base, _, view = policy.replay_inputs(packet, menu, 4)
        cache[key] = base, view
    base, view = cache[key]
    return policy.decision_scores(base, view, menu, 4)


def update(optimizer, parameters, losses):
    import torch

    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("the alias policy update became nonfinite")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()


def train_arm(policy, groups, tree, *, repl=None):
    import torch
    from torch.nn import functional as F

    from ppt_gym.ram_learning import tensor_hash

    names = {"normalization_logit", "preview_logit", "shared_transport.selection_logit"}
    policy.requires_grad_(False)
    parameters = [value.requires_grad_(True) for key, value in policy.named_parameters() if key in names]
    frozen = {key: value for key, value in policy.state_dict().items() if key not in names}
    keep = tensor_hash(frozen)
    before = {key: float(getattr(policy, key).detach()) for key in ("normalization_logit", "preview_logit")}
    before["shared_transport.selection_logit"] = float(policy.shared_transport.selection_logit.detach())
    optimizer = torch.optim.Adam(parameters, lr=.08)
    cache, feedback, decisions = {}, {}, 0
    for number, pair in enumerate(groups.values(), 1):
        losses = []
        for row in pair:
            if repl is None:
                for packet, menu, choices, label, _ in demonstrated_actions(row):
                    scores = cached_logits(policy, cache, packet, menu)[choices] / .15
                    losses.append(-F.log_softmax(scores, dim=0)[label])
            else:
                menu = opening_menu(row)
                traces = {action: run_search(repl, row, policy, tree, opening=action) for action in menu}
                if not all(trial["within"] for trial in traces.values()):
                    raise ValueError("an alias training opening exceeded its CPU budget")
                logits = cached_logits(policy, cache, row["packet"], menu)
                rewards = logits.new_tensor([traces[action]["reward"] for action in menu])
                probabilities = (logits / .15).softmax(-1)
                losses.append(-(probabilities * (rewards - rewards.mean())).sum())
                feedback[row["case"]["lineage"]] = {"trials": traces, "probabilities": probabilities.detach().tolist()}
        update(optimizer, parameters, losses)
        decisions += len(losses)
        if number % 3 == 0:
            print(f"Alias updates after {number} of {len(groups)} layouts.", flush=True)
    after = {name: float(value.detach()) for name, value in policy.named_parameters() if name in names}
    if len(after) != len(names) or before == after or tensor_hash(frozen) != keep:
        raise ValueError("the alias fit failed to isolate changes to the registered recurrent parameters")
    return {"before": before, "after": after, "decisions": decisions, "live_feedback": feedback}


def train(args, tree):
    from lcgrt.lean_repl import LeanRepl

    witness, groups = read_witness(args.training, "train")
    args.checkpoints.mkdir()
    results = {}
    for arm in ("live_rl", "imitation"):
        print(f"Fitting {arm} on {len(groups)} alias layouts.", flush=True)
        policy, architecture = warm_policy(args.hybrid)
        if arm == "live_rl":
            with LeanRepl() as repl:
                if repl.version != witness["lean"]:
                    raise ValueError("the alias learner needs the registered Lean verifier")
                result = train_arm(policy, groups, tree, repl=repl)
        else:
            result = train_arm(policy, groups, tree)
        checkpoint = args.checkpoints / f"{arm}.pt"
        save_policy(policy, checkpoint, architecture)
        results[arm] = {**result, "checkpoint_sha256": sha256_file(checkpoint)}
        print(arm, result["after"], flush=True)
    return {"training_sha256": sha256_file(args.training), "hybrid_sha256": sha256_file(args.hybrid),
            "tactic_checks": BUDGET, "mac_limit": MAC_LIMIT, "results": results, "memory": tree.stats()}


def train_trm(args, tree):
    import torch
    from torch.nn import functional as F

    _, groups = read_witness(args.training, "train")
    policy, architecture = warm_policy(args.trm, trm=True)
    parameters = list(policy.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001)
    decisions = 0
    for number, pair in enumerate(groups.values(), 1):
        losses = []
        for row in pair:
            menu = opening_menu(row)
            logits, _ = policy(row["packet"], menu, 1)
            viable = set(row["viable_openings"])
            native = logits.new_tensor([float(tactic in viable) for tactic in menu])
            losses.append(F.smooth_l1_loss(logits - logits.mean(), 2 * (native - native.mean())))
            for packet, choices, eligible, target, first in demonstrated_actions(row):
                if first not in viable or packet == row["packet"]:
                    continue
                scores, _ = policy(packet, choices, 1)
                losses.append(-scores[eligible].log_softmax(dim=0)[target])
        update(optimizer, parameters, losses)
        decisions += len(losses)
        if number % 6 == 0:
            print(f"Graph TRM updates after {number} of {len(groups)} layouts.", flush=True)
    save_policy(policy, args.checkpoint, architecture)
    return {"training_sha256": sha256_file(args.training), "base_trm_sha256": sha256_file(args.trm),
            "checkpoint_sha256": sha256_file(args.checkpoint), "decisions": decisions, "memory": tree.stats()}


def calibrate(policy, groups, *, gain_only):
    import torch

    heads = (policy.shared_transport.readout_weight,)
    if not gain_only:
        heads = (policy.actor.weight, policy.graph.action.weight, policy.graph.transport.action.weight, *heads)
    policy.requires_grad_(False)
    for head in heads:
        head.requires_grad_(True)
    features, targets, layout_ids = [], [], []
    for i, pair in enumerate(groups.values()):
        for row in pair:
            menu = opening_menu(row)
            logits, _ = policy(row["packet"], menu, 4)
            centered = logits - logits.mean()
            for name, margin in zip(menu, centered, strict=True):
                jacobian = torch.autograd.grad(margin, heads, retain_graph=True, materialize_grads=True)
                features.append(torch.cat([value.detach().flatten() for value in jacobian]))
                target = 1. if name in row["viable_openings"] else -1.
                targets.append(target - float(margin.detach()))
                layout_ids.append(i)
    x = torch.stack(features)
    y = x.new_tensor(targets)
    scale = x.square().sum(1).median().clamp_min(1e-8)
    gram = x @ x.T

    def solve(selected, penalty):
        sub = gram[selected][:, selected]
        return x[selected].T @ torch.linalg.solve(sub + penalty * scale * torch.eye(len(sub)), y[selected])

    losses = {}
    for penalty in (.01, .1, 1., 10.):
        errors = []
        for i in range(len(groups)):
            held = torch.tensor([value == i for value in layout_ids])
            errors.append((x[held] @ solve(~held, penalty) - y[held]).square().mean())
        losses[penalty] = float(torch.stack(errors).mean())
    best = min(losses, key=losses.get)
    adjustment = solve(slice(None), best)
    with torch.no_grad():
        for head, delta in zip(heads, torch.split(adjustment, [head.numel() for head in heads]), strict=True):
            head.add_(delta.reshape_as(head))
        if gain_only:
            heads[0].clamp_min_(0.)
    policy.requires_grad_(False)
    return {"penalty": best, "held_layout_losses": {str(key): value for key, value in losses.items()},
            "gain": float(policy.shared_transport.readout_weight)}


def native_scores(repl, policy, witness, tree):
    import torch

    rows, forms = {}, defaultdict(list)
    depth = 4 if policy.architecture == "hybrid" else 1
    with torch.no_grad():
        for root, row in witness["rows"].items():
            menu = opening_menu(row)
            opening = menu[int(policy(row["packet"], menu, depth)[0].argmax())]
            actual = run_search(repl, row=row, policy=policy, tree=tree)
            correct = opening in row["viable_openings"]
            if actual["events"][0]["tactic"] != opening:
                raise ValueError("the live alias policy differed from its opening readout")
            forms[row["case"]["cluster"]].append((correct, actual["closed"]))
            rows[root] = {**actual, "opening": opening, "opening_viable": correct}
    counts = {name: {"worlds": sum(sum(v[index] for v in pair) for pair in forms.values()),
                     "pairs": sum(all(v[index] for v in pair) for pair in forms.values())}
              for index, name in enumerate(("opening", "search"))}
    return {"counts": counts, "rows": rows}


def evaluate(args, tree):
    import torch

    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.ram_learning import tensor_hash

    training, groups = read_witness(args.training, "train")
    witness, _ = read_witness(args.witness, "dev")
    if set(training["rows"]) & set(witness["rows"]):
        raise ValueError("alias evaluation changed the registered held-out split")
    policies, details = {}, {}
    if args.group != "controls":
        fit = read_fit(args.fit)
        if (fit["training_sha256"] != sha256_file(args.training) or fit["hybrid_sha256"] != sha256_file(args.hybrid)
                or fit["tactic_checks"] != BUDGET or fit["mac_limit"] != MAC_LIMIT):
            raise ValueError("the alias learner provenance changed")
        for arm in ("live_rl", "imitation"):
            checkpoint = args.checkpoints / f"{arm}.pt"
            if fit["results"][arm]["checkpoint_sha256"] != sha256_file(checkpoint):
                raise ValueError("a frozen alias learner checkpoint changed")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            policy, _ = warm_policy(args.hybrid)
            policy.load_state_dict(state["state_dict"])
            if tensor_hash(policy.state_dict()) != state["weights_sha256"]:
                raise ValueError("the alias learner changed on CPU reload")
            policies[arm] = policy
        details["fit_sha256"] = sha256_file(args.fit)
    if args.group != "learned":
        proof = read_fit(args.trm_fit)
        if proof["training_sha256"] != sha256_file(args.training) or proof["checkpoint_sha256"] != sha256_file(args.alias_trm):
            raise ValueError("the alias-aware graph TRM provenance changed")
        policies["graph_trm_with_definitions"], _ = warm_policy(args.alias_trm, trm=True)
        policies["fixed_alias_bank"], _ = warm_policy(args.hybrid)
        policies["hard_alias_reference"], _ = warm_policy(args.hybrid)
        policies["hard_alias_reference"].hard_shared = True
        for label, scalar in (("calibrated_alias_gain", True), ("calibrated_alias_affine", False)):
            policies[label], _ = warm_policy(args.hybrid)
            details[label] = calibrate(policies[label], groups, gain_only=scalar)
        details["trm_fit_sha256"] = sha256_file(args.trm_fit)
    results = {}
    with LeanRepl() as repl:
        if repl.version != witness["lean"]:
            raise ValueError("alias evaluation needs its registered Lean verifier")
        for name, policy in policies.items():
            results[name] = native_scores(repl, policy, witness, tree)
            print(name, results[name]["counts"], flush=True)
    return {**details, "group": args.group, "training_sha256": sha256_file(args.training),
            "witness_sha256": sha256_file(args.witness), "tactic_checks": BUDGET, "mac_limit": MAC_LIMIT,
            "results": results, "memory": tree.stats()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    fit, trm, score = (commands.add_parser(name) for name in ("train", "trm", "evaluate"))
    for parser in (fit, trm, score):
        for name in ("training", "output"):
            parser.add_argument(name, type=Path)
    for parser in (fit, score):
        for name in ("hybrid", "checkpoints"):
            parser.add_argument(name, type=Path)
    for name in ("trm", "checkpoint"):
        trm.add_argument(name, type=Path)
    for name in ("witness", "fit", "alias_trm", "trm_fit"):
        score.add_argument(name, type=Path)
    score.add_argument("--group", choices=("all", "learned", "controls"), default="all")
    args = cli.parse_args()
    tree = TreeBudget(2, 1)
    try:
        from ppt_gym.ram_campaign import configure

        configure(tree.threads, 103)
        result = {"train": train, "trm": train_trm, "evaluate": evaluate}[args.stage](args, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(result).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
