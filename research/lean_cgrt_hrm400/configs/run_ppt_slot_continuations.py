"""Continue the typed two-slot policy on multi-decision native Lean searches."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import random

from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym import dependent_witness
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.two_slot_fixture import BUNDLES, PROBES, ROUTE, continuation_fixtures

from configs.run_ppt_two_slot import (
    CPU_LIMIT, EXPLORATION_TEMPERATURE, MAC_LIMIT, TRM_SCHEMA, calibrate_readout, read_fit, score_pairs, trajectory,
)
from configs.pretrain_ppt_slot_mixed import SCHEMA as MIXED_SCHEMA


SCHEMA = "ppt.two-slot-native-continuation.v1"
SEED = 831
EPOCHS = 2
TRM_LOOKAHEAD_PASSES = 2
TRM_ARMS = {"trm", "trm_paired", "trm_workspace", "trm_lookahead"}
EXPLORATORY_PAIR_MARGIN = 3
RESULTS = Path(__file__).resolve().parents[1] / "results/ppt_hybrid_rl"
WITNESS = {"dev": RESULTS / "ppt-two-slot-continuation-witness.json.gz",
           "confirm": RESULTS / "ppt-two-slot-continuation-confirm-witness.json.gz",
           "counterfactual": RESULTS / "ppt-two-slot-continuation-counterfactual-witness.json.gz",
           "shared-witness": RESULTS / "ppt-shared-witness-horn-dev.json.gz",
           "deep-witness": RESULTS / "ppt-deep-witness-horn-dev.json.gz"}
WITNESS_COHORTS = {"shared-witness": dependent_witness.cohorts, "deep-witness": dependent_witness.deep_cohorts}


def witness_cases(split, subset):
    return [(case, dependent_witness.FORK[winner]) for case, winner, _ in WITNESS_COHORTS[split](subset)]


def training_cases(split):
    return witness_cases(split, "train") if split in WITNESS_COHORTS else list(continuation_fixtures("train"))


def heldout_registry(split):
    witness = json.loads(gzip.decompress(WITNESS[split].read_bytes()))
    cohort = witness_cases(split, "dev") if split in WITNESS_COHORTS else list(continuation_fixtures(split))
    if (witness["lean"] != LEAN_VERSION
            or [(row["case"], row["winner"]) for row in witness["rows"]]
            != [(asdict(case), winner) for case, winner in cohort]):
        raise ValueError("the continuation cohort disagrees with its frozen native proof witness")
    return cohort


def decisions(policy, episode):
    """Replay public observations and carry, with gradients for the chosen path."""
    from ppt_gym.two_slot import candidate_state, slot_tensor

    events, carry = episode["events"], None
    budget = episode.get("budget", PROBES[ROUTE].checks)
    for index, event in enumerate(events):
        if event.get("certificate_only"):
            break
        goals, menu = event["goals"], event["menu"]
        scores, slots = policy(goals, menu, budget - index, carry)
        selected = menu.index(event["action"])
        if len(menu) > 1:
            yield index, selected, scores, slot_tensor(slots)[:, 0]
        if index + 1 < len(events):
            carry = policy.advance(candidate_state(slots, selected), events[index + 1]["goals"])


def training_return(trace, certificate_cost=False):
    return 0. if certificate_cost and not trace["success"] else trace["reward"]


def probe_continuation(repl, tree, case, policy, opening, rng, *, late_proof_credit=False):
    budget = dependent_witness.BUDGETS.get(getattr(case, "family", None), PROBES[ROUTE].checks)
    draws = tuple(rng.random() for _ in range(budget))
    original = trajectory(repl, tree, case, policy=policy, fork=opening, draws=draws,
                          late_proof_credit=late_proof_credit)
    first_suffix = original.get("fork_index", 1) + 1
    for index, event in enumerate(original["events"][first_suffix:], first_suffix):
        if len(event["menu"]) < 2:
            continue
        prefix = tuple(item["action"] for item in original["events"][:index])
        alternatives = []
        for action in event["menu"]:
            if action == event["action"]:
                continue
            trace = trajectory(repl, tree, case, policy=policy, forced_prefix=(*prefix, action), draws=draws,
                               late_proof_credit=late_proof_credit)
            if (trace["events"][:index] != original["events"][:index] or len(trace["events"]) <= index
                    or any(trace["events"][index][key] != event[key] for key in ("goals", "menu"))
                    or trace["events"][index]["action"] != action):
                raise ValueError("the continuation intervention changed its native decision state")
            alternatives.append(trace)
        return original, {"index": index, "alternatives": alternatives, "draws": draws}
    return original, None


def branch_trials(record, branch):
    trace = record["trials"][branch]
    probe = record.get("probes", [None] * len(record["trials"]))[branch]
    return (trace, *probe["alternatives"]) if probe is not None else (trace,)


def branch_return(record, branch, certificate_cost):
    return max(training_return(t, certificate_cost) for t in branch_trials(record, branch))


def learn_pair(policy, optimizer, records, baseline, *, certificate_cost=False):
    import torch

    opening_losses, suffix_losses = [], []
    for record in records:
        for branch, trace in enumerate(record["trials"]):
            probe = record.get("probes", [None] * len(record["trials"]))[branch]
            for index, selected, scores, _ in decisions(policy, trace):
                probabilities = (scores / EXPLORATION_TEMPERATURE).log_softmax(0)
                if index == trace.get("fork_index", 1):
                    if branch == 0:
                        rewards = scores.new_tensor([branch_return(record, i, certificate_cost)
                                                     for i in range(len(record["trials"]))])
                        opening_losses.append(-(probabilities.exp() * (rewards - rewards.mean())).sum())
                elif probe is not None and index == probe["index"]:
                    outcomes = {t["events"][index]["action"]: training_return(t, certificate_cost)
                                for t in branch_trials(record, branch)}
                    rewards = scores.new_tensor([outcomes[action] for action in trace["events"][index]["menu"]])
                    suffix_losses.append(-(probabilities.exp() * (rewards - rewards.mean())).sum())
                else:
                    suffix_losses.append(-(training_return(trace, certificate_cost) - baseline) * probabilities[selected])
    loss = torch.stack(opening_losses).mean()
    if suffix_losses:
        loss = loss + torch.stack(suffix_losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    mean_return = sum(training_return(t, certificate_cost) for rec in records for t in rec["trials"]) / 4
    return .8 * baseline + .2 * mean_return


def collect(repl, tree, policy, *, split="dev", certificate_cost=False, paired_continuations=False,
            late_proof_credit=False):
    import torch

    cases = training_cases(split)
    openings = dependent_witness.FORK if split in WITNESS_COHORTS else BUNDLES
    fork_index = dependent_witness.FORK_INDEX if split in WITNESS_COHORTS else 1
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
    rng = random.Random(SEED)
    records, baseline = [], 0.
    for epoch in range(EPOCHS):
        for left, right in zip(cases[::2], cases[1::2], strict=True):
            pair = []
            for case, _ in (left, right):
                record = {"case": asdict(case), "epoch": epoch}
                if paired_continuations:
                    trials, probes = zip(*(probe_continuation(repl, tree, case, policy, action, rng,
                                                              late_proof_credit=late_proof_credit)
                                           for action in openings), strict=True)
                    record["probes"] = probes
                else:
                    trials = [trajectory(repl, tree, case, policy=policy, fork=action, rng=rng,
                                         late_proof_credit=late_proof_credit) for action in openings]
                record["trials"] = trials
                if any(not trial["within"] or len(trial["events"]) <= fork_index
                       for i in range(len(trials)) for trial in branch_trials(record, i)):
                    raise ValueError("a routed training trial was censored before its terminal return")
                pair.append(record)
                records.append(record)
            baseline = learn_pair(policy, optimizer, pair, baseline, certificate_cost=certificate_cost)
        print(f"Finished live pass {epoch + 1} of {EPOCHS}.", flush=True)
    return records


def recorded_decisions(policy, record, certificate_cost, *, successful_only=False):
    for branch in range(len(record["trials"])):
        root_return = branch_return(record, branch, certificate_cost)
        for trace in branch_trials(record, branch):
            if successful_only and not trace["success"]:
                continue
            for index, action, scores, features in decisions(policy, trace):
                target = root_return if index == trace.get("fork_index", 1) else training_return(trace, certificate_cost)
                yield action, scores, features, target


def readout(policy, records, *, certificate_cost=False):
    import torch

    features, returns, clusters = [], [], []
    with torch.no_grad():
        for record in records:
            for action, _, vectors, target in recorded_decisions(policy, record, certificate_cost):
                features.append(vectors[action])
                returns.append(target)
                clusters.append(record["case"]["cluster"])
    return calibrate_readout(policy, torch.stack(features), returns, clusters)


def offline(policy, records, *, imitate, certificate_cost=False):
    import torch
    from torch.nn import functional as F

    optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
    for _ in range(EPOCHS):
        for record in records:
            losses = []
            for action, scores, _, target in recorded_decisions(policy, record, certificate_cost, successful_only=imitate):
                value = scores[action]
                term = -scores.log_softmax(0)[action] if imitate else F.smooth_l1_loss(value, value.new_tensor(target))
                losses.append(term)
            if losses:
                optimizer.zero_grad()
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()


def replay_paired(policy, records, *, certificate_cost):
    """Give TRM the same optimizer updates and certified outcomes as the live learner."""
    import torch

    optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
    baseline = 0.
    for pair in zip(records[::2], records[1::2], strict=True):
        baseline = learn_pair(policy, optimizer, pair, baseline, certificate_cost=certificate_cost)


def warm_start(policy, state):
    missing, unexpected = policy.load_state_dict(state, strict=False)
    expected = {name for name in policy.state_dict() if name.startswith("path_bank.")}
    if unexpected or set(missing) != expected:
        raise ValueError("the continuation warm start changed parameters outside the added bank")


def fit(repl, tree, args):
    import torch

    from ppt_gym.two_slot import TwoSlotPolicy, TwoSlotTRM

    warm = read_fit(args.source)
    parent_hash = sha256_file(args.source)
    prior_trm = torch.load(args.trm, map_location="cpu", weights_only=True)
    if (not warm.get("bindings") or prior_trm["schema"] != TRM_SCHEMA
            or prior_trm["parent_sha256"] != parent_hash):
        raise ValueError("the routed comparison requires matching typed warm starts")
    warm_trm_hash = sha256_file(args.trm)
    hybrid_state, trm_state = warm["learned"], prior_trm["weights"]
    mixed_receipt = {}
    if args.mixed_warm is not None:
        mixed = torch.load(args.mixed_warm, map_location="cpu", weights_only=True)
        if mixed["schema"] != MIXED_SCHEMA or mixed["parents"] != {"hybrid": parent_hash, "trm": warm_trm_hash}:
            raise ValueError("the mixed replay checkpoint must share the registered warm parents")
        hybrid_state, trm_state = mixed["weights"]["hybrid"], mixed["weights"]["trm"]
        mixed_receipt = {"mixed_warm_sha256": sha256_file(args.mixed_warm), "mixed_source_counts": mixed["counts"]}
    witnesses = args.split in WITNESS_COHORTS
    settings = {"bindings": True, "paths": args.paths, "witnesses": witnesses}
    initial = TwoSlotPolicy(**settings, residual_carry=args.residual_carry, witness_workspace=args.witness_workspace)
    warm_start(initial, hybrid_state)
    learned, calibrated, imitation = (deepcopy(initial) for _ in range(3))
    records = collect(repl, tree, learned, split=args.split, certificate_cost=args.certificate_cost,
                      paired_continuations=args.paired_continuations, late_proof_credit=args.late_proof_credit)
    ridge = readout(calibrated, records, certificate_cost=args.certificate_cost)
    offline(imitation, records, imitate=True)
    trm = TwoSlotTRM(**settings)
    warm_start(trm, trm_state)
    paired_trms = {}
    if witnesses:
        paired_trms["trm_paired"] = deepcopy(trm)
    if args.witness_workspace:
        workspace_trm = deepcopy(trm)
        workspace_trm.witness_workspace = True
        paired_trms["trm_workspace"] = workspace_trm
    if args.split == "deep-witness":
        lookahead_trm = deepcopy(trm)
        lookahead_trm.passes = TRM_LOOKAHEAD_PASSES
        paired_trms["trm_lookahead"] = lookahead_trm
    offline(trm, records, imitate=False, certificate_cost=args.certificate_cost)
    states = {"live_rl": learned, "calibrated": calibrated, "imitation": imitation, "trm": trm, "warm": initial}
    if args.split == "deep-witness":
        return_regression = deepcopy(initial)
        offline(return_regression, records, imitate=False, certificate_cost=args.certificate_cost)
        states["return_regression"] = return_regression
    for name, policy in paired_trms.items():
        replay_paired(policy, records, certificate_cost=args.certificate_cost)
        states[name] = policy
    with args.output.open("xb") as handle:
        torch.save({"schema": SCHEMA, "lean": repl.version, "warm_sha256": parent_hash,
                    "warm_trm_sha256": warm_trm_hash, **mixed_receipt, "ridge": ridge, "records": records,
                    "witness_sha256": sha256_file(WITNESS[args.split]), "split": args.split,
                    "paths": args.paths, "certificate_cost": args.certificate_cost,
                    "witness_workspace": args.witness_workspace,
                    "late_proof_credit": args.late_proof_credit,
                    "residual_carry": args.residual_carry, "paired_continuations": args.paired_continuations,
                    "training": digest([asdict(case) for case, _ in training_cases(args.split)]),
                    "passes": {key: model.passes for key, model in states.items()},
                    "weights": {key: model.state_dict() for key, model in states.items()}, "memory": tree.stats()}, handle)


def read_continuation_fit(checkpoint, lean):
    import torch

    fit = torch.load(checkpoint, map_location="cpu", weights_only=True)
    split = fit.get("split", "dev")
    if (fit["schema"] != SCHEMA or fit["lean"] != lean
            or fit["witness_sha256"] != sha256_file(WITNESS[split])
            or fit["training"] != digest([asdict(case) for case, _ in training_cases(split)])):
        raise ValueError("the routed fit does not match the registered training cohort")
    return fit


def restore_policy(fit, name, weights):
    from ppt_gym.two_slot import TwoSlotPolicy, TwoSlotTRM

    trm = name in TRM_ARMS
    workspace = fit.get("witness_workspace", False) and (name == "trm_workspace" or
                                                         (not trm and name != "workspace_reset"))
    source_arm = "live_rl" if name in {"workspace_reset", "slot_reset"} else name
    passes = fit.get("passes", {}).get(source_arm, 1 if trm else 2)
    if name == "trm_lookahead" and passes != TRM_LOOKAHEAD_PASSES:
        raise ValueError("the stateless lookahead TRM must use its registered depth")
    settings = {"bindings": True, "paths": fit.get("paths", False),
                "witnesses": fit.get("split", "dev") in WITNESS_COHORTS, "witness_workspace": workspace,
                "passes": passes}
    policy = (TwoSlotTRM(**settings) if trm else
              TwoSlotPolicy(**settings, residual_carry=fit.get("residual_carry", False))).eval()
    policy.load_state_dict(weights)
    return policy


def paired_margins(results):
    """Count layout-level wins and losses for the registered deep comparison."""
    candidate = results["live_rl"]["pairs"]
    margins = {}
    for name, result in results.items():
        if name == "live_rl":
            continue
        control = result["pairs"]
        if candidate.keys() != control.keys():
            raise ValueError(f"the paired comparison has different layouts in {name}")
        differences = [int(row["complete"]) - int(control[key]["complete"]) for key, row in candidate.items()]
        wins, losses = differences.count(1), differences.count(-1)
        margins[name] = {"live_rl_only": wins, "control_only": losses, "ties": differences.count(0),
                         "net_complete_pairs": wins - losses}
    return {"layouts": len(candidate), "minimum_exploratory_gain": EXPLORATORY_PAIR_MARGIN, "controls": margins}


def evaluate(repl, tree, args):
    fit = read_continuation_fit(args.checkpoint, repl.version)
    split = fit.get("split", "dev")
    witnesses = split in WITNESS_COHORTS
    results, cohort = {}, heldout_registry(split)
    arms = [*fit["weights"].items(), ("slot_reset", fit["weights"]["live_rl"])]
    workspace = fit.get("witness_workspace", False)
    if workspace:
        arms.append(("workspace_reset", fit["weights"]["live_rl"]))
    for name, weights in arms:
        policy = restore_policy(fit, name, weights)
        rows = []
        for case, winner in cohort:
            trial = trajectory(repl, tree, case, policy=policy, reset_slots=name == "slot_reset")
            events = trial["events"]
            fork_index = trial.get("fork_index", 1)
            viable = len(events) > fork_index and events[fork_index]["action"] == winner
            if trial["success"] and not viable:
                raise ValueError("a certified continuation contradicted its native fork witness")
            rows.append({"case": asdict(case), "viable_opening": viable, **trial})
        results[name] = score_pairs(rows)
        results[name]["viable_openings"] = sum(row["viable_opening"] for row in rows)
        if witnesses:
            results[name]["parameters"] = sum(p.numel() for p in policy.parameters())
            results[name]["passes"] = policy.passes
            results[name]["max_macs"] = max(row["macs"] for row in rows)
        print(name, results[name]["proofs"], results[name]["complete_pairs"], flush=True)
    report = {"schema": SCHEMA, "lean": repl.version, "checkpoint_sha256": sha256_file(args.checkpoint),
              "witness_sha256": fit["witness_sha256"],
              "split": split, "paths": fit.get("paths", False), "certificate_cost": fit.get("certificate_cost", False),
              "witness_workspace": workspace,
              "late_proof_credit": fit.get("late_proof_credit", False),
              "residual_carry": fit.get("residual_carry", False), "paired_continuations": fit.get("paired_continuations", False),
              "cohort": digest([asdict(case) for case, _ in cohort]), "results": results,
              "tactic_checks": dependent_witness.BUDGETS[cohort[0][0].family] if witnesses else PROBES[ROUTE].checks,
              "mac_limit": MAC_LIMIT, "cpu_limit": CPU_LIMIT,
              "memory": tree.stats()}
    if split == "deep-witness":
        report["paired_margins"] = paired_margins(results)
    return report


def main():
    from ppt_gym.ram_campaign import configure

    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    train = commands.add_parser("fit")
    train.add_argument("source", type=Path)
    train.add_argument("trm", type=Path)
    train.add_argument("--paths", action="store_true")
    train.add_argument("--certificate-cost", action="store_true")
    train.add_argument("--mixed-warm", type=Path)
    train.add_argument("--residual-carry", action="store_true")
    train.add_argument("--paired-continuations", action="store_true")
    train.add_argument("--witness-workspace", action="store_true")
    train.add_argument("--late-proof-credit", action="store_true")
    splits = train.add_mutually_exclusive_group()
    splits.add_argument("--confirm", dest="split", action="store_const", const="confirm", default="dev")
    splits.add_argument("--counterfactual", dest="split", action="store_const", const="counterfactual")
    splits.add_argument("--shared-witness", dest="split", action="store_const", const="shared-witness")
    splits.add_argument("--deep-witness", dest="split", action="store_const", const="deep-witness")
    score = commands.add_parser("evaluate")
    score.add_argument("checkpoint", type=Path)
    for parser in (train, score):
        parser.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.stage == "fit":
        if args.split in WITNESS_COHORTS and args.paths:
            cli.error("shared-witness splits use their own typed proof bank")
        if args.witness_workspace and args.split not in WITNESS_COHORTS:
            cli.error("--witness-workspace requires a shared-witness split")
        if args.late_proof_credit and args.split != "deep-witness":
            cli.error("--late-proof-credit requires the deep witness split")
    if args.output.exists():
        cli.error("output already exists")
    tree = TreeBudget(2, 1)
    try:
        configure(tree.threads, SEED)
        with LeanRepl() as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the continuation study requires its pinned Lean verifier")
            if args.stage == "fit":
                heldout_registry(args.split)
                fit(repl, tree, args)
                return
            result = evaluate(repl, tree, args)
        with args.output.open("x", encoding="utf-8") as target:
            target.write(canonical(result) + "\n")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
