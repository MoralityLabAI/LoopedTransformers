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
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.two_slot_fixture import BUNDLES, PROBES, ROUTE, continuation_fixtures

from configs.run_ppt_two_slot import (
    CPU_LIMIT, EXPLORATION_TEMPERATURE, MAC_LIMIT, TRM_SCHEMA, calibrate_readout, read_fit, score_pairs, trajectory,
)


SCHEMA = "ppt.two-slot-native-continuation.v1"
SEED = 831
EPOCHS = 2
WITNESS = Path(__file__).resolve().parents[1] / "results/ppt_hybrid_rl/ppt-two-slot-continuation-witness.json.gz"


def heldout_registry():
    witness = json.loads(gzip.decompress(WITNESS.read_bytes()))
    cohort = list(continuation_fixtures("dev"))
    if (witness["lean"] != LEAN_VERSION
            or [(row["case"], row["winner"]) for row in witness["rows"]]
            != [(asdict(case), winner) for case, winner in cohort]):
        raise ValueError("the continuation cohort disagrees with its frozen native proof witness")
    return cohort


def decisions(policy, episode):
    """Replay public observations and carry, with gradients for the chosen path."""
    events, carry = episode["events"], None
    for index, event in enumerate(events):
        goals, menu = event["goals"], event["menu"]
        scores, slots = policy(goals, menu, PROBES[ROUTE].checks - index, carry)
        selected = menu.index(event["action"])
        if len(menu) > 1:
            yield index, selected, scores, slots[:, 0]
        if index + 1 < len(events):
            carry = policy.advance(slots[selected], events[index + 1]["goals"])


def learn_pair(policy, optimizer, worlds, baseline):
    import torch

    opening_losses, suffix_losses = [], []
    for traces in worlds:
        for branch, trace in enumerate(traces):
            for index, selected, scores, _ in decisions(policy, trace):
                probabilities = (scores / EXPLORATION_TEMPERATURE).log_softmax(0)
                if index == 1:
                    if branch == 0:
                        rewards = scores.new_tensor([trial["reward"] for trial in traces])
                        opening_losses.append(-(probabilities.exp() * (rewards - rewards.mean())).sum())
                else:
                    suffix_losses.append(-(trace["reward"] - baseline) * probabilities[selected])
    loss = torch.stack(opening_losses).mean()
    if suffix_losses:
        loss = loss + torch.stack(suffix_losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()


def collect(repl, tree, policy):
    import torch

    cases = list(continuation_fixtures("train"))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
    rng = random.Random(SEED)
    records, baseline = [], 0.
    for epoch in range(EPOCHS):
        for left, right in zip(cases[::2], cases[1::2], strict=True):
            pair = []
            for case, _ in (left, right):
                trials = [trajectory(repl, tree, case, policy=policy, fork=action, rng=rng) for action in BUNDLES]
                if any(not trial["within"] or len(trial["events"]) < 2 for trial in trials):
                    raise ValueError("a routed training trial was censored before its terminal return")
                pair.append(trials)
                records.append({"case": asdict(case), "epoch": epoch, "trials": trials})
            learn_pair(policy, optimizer, pair, baseline)
            baseline = .8 * baseline + .2 * (sum(t["reward"] for trials in pair for t in trials) / 4)
        print(f"Finished live pass {epoch + 1} of {EPOCHS}.", flush=True)
    return records


def readout(policy, records):
    import torch

    features, returns, clusters = [], [], []
    with torch.no_grad():
        for record in records:
            for trace in record["trials"]:
                for _, action, _, vectors in decisions(policy, trace):
                    features.append(vectors[action])
                    returns.append(trace["reward"])
                    clusters.append(record["case"]["cluster"])
    return calibrate_readout(policy, torch.stack(features), returns, clusters)


def offline(policy, records, *, imitate):
    import torch
    from torch.nn import functional as F

    optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
    for _ in range(EPOCHS):
        for record in records:
            losses = []
            for trace in record["trials"]:
                if imitate and not trace["success"]:
                    continue
                for _, action, scores, _ in decisions(policy, trace):
                    value = scores[action]
                    term = -scores.log_softmax(0)[action] if imitate else F.smooth_l1_loss(
                        value, value.new_tensor(trace["reward"]))
                    losses.append(term)
            if losses:
                optimizer.zero_grad()
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()


def fit(repl, tree, args):
    import torch

    from ppt_gym.two_slot import TwoSlotPolicy, TwoSlotTRM

    warm = read_fit(args.source)
    parent_hash = sha256_file(args.source)
    prior_trm = torch.load(args.trm, map_location="cpu", weights_only=True)
    if (not warm.get("bindings") or prior_trm["schema"] != TRM_SCHEMA
            or prior_trm["parent_sha256"] != parent_hash):
        raise ValueError("the routed comparison requires matching typed warm starts")
    initial = TwoSlotPolicy(bindings=True)
    initial.load_state_dict(warm["learned"])
    learned, calibrated, imitation = (deepcopy(initial) for _ in range(3))
    records = collect(repl, tree, learned)
    ridge = readout(calibrated, records)
    offline(imitation, records, imitate=True)
    trm = TwoSlotTRM(bindings=True)
    trm.load_state_dict(prior_trm["weights"])
    offline(trm, records, imitate=False)
    states = {"live_rl": learned, "calibrated": calibrated, "imitation": imitation, "trm": trm, "warm": initial}
    with args.output.open("xb") as handle:
        torch.save({"schema": SCHEMA, "lean": repl.version, "warm_sha256": parent_hash,
                    "warm_trm_sha256": sha256_file(args.trm), "ridge": ridge, "records": records,
                    "witness_sha256": sha256_file(WITNESS),
                    "training": digest([asdict(case) for case, _ in continuation_fixtures("train")]),
                    "weights": {key: model.state_dict() for key, model in states.items()}, "memory": tree.stats()}, handle)


def evaluate(repl, tree, args):
    import torch

    from ppt_gym.two_slot import TwoSlotPolicy, TwoSlotTRM

    fit = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (fit["schema"] != SCHEMA or fit["lean"] != repl.version
            or fit["witness_sha256"] != sha256_file(WITNESS)
            or fit["training"] != digest([asdict(case) for case, _ in continuation_fixtures("train")])):
        raise ValueError("the routed fit does not match the registered training cohort")
    results, cohort = {}, heldout_registry()
    for name, weights in (*fit["weights"].items(), ("slot_reset", fit["weights"]["live_rl"])):
        policy = (TwoSlotTRM if name == "trm" else TwoSlotPolicy)(bindings=True).eval()
        policy.load_state_dict(weights)
        rows = []
        for case, winner in cohort:
            trial = trajectory(repl, tree, case, policy=policy, reset_slots=name == "slot_reset")
            events = trial["events"]
            viable = len(events) > 1 and events[1]["action"] == winner
            if trial["success"] and not viable:
                raise ValueError("a certified continuation contradicted its native fork witness")
            rows.append({"case": asdict(case), "viable_opening": viable, **trial})
        results[name] = score_pairs(rows)
        results[name]["viable_openings"] = sum(row["viable_opening"] for row in rows)
        print(name, results[name]["proofs"], results[name]["complete_pairs"], flush=True)
    return {"schema": SCHEMA, "lean": repl.version, "checkpoint_sha256": sha256_file(args.checkpoint),
            "witness_sha256": fit["witness_sha256"],
            "cohort": digest([asdict(case) for case, _ in cohort]), "results": results,
            "tactic_checks": PROBES[ROUTE].checks, "mac_limit": MAC_LIMIT, "cpu_limit": CPU_LIMIT,
            "memory": tree.stats()}


def main():
    from ppt_gym.ram_campaign import configure

    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    train = commands.add_parser("fit")
    train.add_argument("source", type=Path)
    train.add_argument("trm", type=Path)
    score = commands.add_parser("evaluate")
    score.add_argument("checkpoint", type=Path)
    for parser in (train, score):
        parser.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("output already exists")
    tree = TreeBudget(2, 1)
    try:
        configure(tree.threads, SEED)
        with LeanRepl() as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the continuation study requires its pinned Lean verifier")
            if args.stage == "fit":
                heldout_registry()
                fit(repl, tree, args)
                return
            result = evaluate(repl, tree, args)
        with args.output.open("x", encoding="utf-8") as target:
            target.write(canonical(result) + "\n")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
