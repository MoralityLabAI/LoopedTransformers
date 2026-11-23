"""Fit and compare guard acquisition and recovery on native Lean traces."""

import argparse
from collections import defaultdict
from copy import deepcopy
from math import fsum
from pathlib import Path

import torch

from configs.freeze_ppt_guard_recovery import certify_routes
from configs.freeze_ppt_native_progress import native_sources
from configs.run_ppt_guard_fork_fit import truth_head
from configs.run_ppt_live_quote_fit import train, warm_values
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, REPL_EXE
from ppt_gym.data import sha256_file
from ppt_gym.guard_recovery import ALLOWANCES, ENTER, PROBE, SCHEMA as BANK_SCHEMA
from ppt_gym.guard_recovery_learning import action_features, baked_policy, raw_guards, source_targets, trace_policy
from ppt_gym.live_quote_policy import LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.guard-recovery-fit.v1"
SEED = 1103
UPDATES = 600
BATCH = 32


def registered(args):
    bank = read_receipt(args.bank)
    pins = {name: sha256_file(getattr(args, name)) for name in ("bank", "warm", "prior", "bpd", "mathcan")}
    train = {case["pair"] for case in bank["source"]}
    held = {case["pair"] for case in bank["development"]}
    if (bank["schema"] != BANK_SCHEMA or len(train) != 48 or len(held) != 40 or train & held
            or bank["lean"] != sha256_file(LEAN_EXE) or bank["repl"] != sha256_file(REPL_EXE)
            or bank["imports"] != {getattr(args, name).name: pins[name] for name in ("bpd", "mathcan")}):
        raise ValueError("the native guard recovery cohort or original definitions changed")
    return bank, pins


def models_for():
    model = TwoSlotQuoteActor()
    return {"bellman_two_slot": model, "empirical_two_slot": deepcopy(model), "bellman_trm": LiveQuoteTRM()}


def fit_direct(source, rng):
    reader = truth_head()
    unique = {}
    for case in source:
        for token, output in zip(raw_guards(case), case["truth"], strict=True):
            key = tuple(token)
            if unique.setdefault(key, output) != output:
                raise ValueError("Lean reported inconsistent truth for a source guard")
    x, y = torch.tensor(list(unique)), torch.tensor(list(unique.values()), dtype=torch.float32)
    opt = torch.optim.AdamW(reader.parameters(), lr=.002)
    for indices in torch.randint(len(x), (UPDATES, BATCH), generator=rng):
        score = reader(x[indices]).flatten()
        loss = nn.functional.binary_cross_entropy_with_logits(score, y[indices])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return reader.eval(), len(unique)


@torch.no_grad()
def encode_case(case, reader, prior):
    descriptors = raw_guards(case)
    confidence = reader(torch.tensor(descriptors)).sigmoid().clamp(.02, .98).flatten().tolist()
    opening = ("apply h0", "apply h1")
    warm = warm_values(prior, case["fork"], opening)
    return descriptors, confidence, tuple(warm[move] for move in opening)


def fit(args, tree):
    bank, pins = registered(args)
    source = bank["source"]
    prior = MixedSourcePrior(args)
    rng = torch.Generator().manual_seed(SEED)
    reader, distinct = fit_direct(source, rng)
    rows, backed, empirical, groups = [], [], [], []
    folds = {key: idx % 6 for idx, key in enumerate(sorted({case["pair"] for case in source}))}
    for case in source:
        descriptors, confidence, warm = encode_case(case, reader, prior)
        for state, move, bellman, sampled in source_targets(case):
            rows.append(action_features(descriptors, confidence, warm, state, move))
            backed.append(bellman)
            empirical.append(sampled)
            groups.append(folds[case["pair"]])
    x, backed, empirical = torch.tensor(rows, dtype=torch.float32), torch.tensor(backed), torch.tensor(empirical)
    batches = torch.randint(len(x), (UPDATES, BATCH), generator=rng)
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if initial["bellman_two_slot"] != initial["empirical_two_slot"]:
        raise ValueError("the matched recovery actors have different initial weights")
    for arm, model in models.items():
        target = empirical if arm == "empirical_two_slot" else backed
        train(model, x, torch.stack((target, target), -1), batches, rate=.002)
        print("fitted", arm, flush=True)
    frozen = ridge(x.flatten(1), backed, torch.tensor(groups))
    weights = {arm: model.state_dict() for arm, model in models.items()}
    weights["guard_reader"] = reader.state_dict()
    with args.fit.open("xb") as handle:
        torch.save({"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial,
                    "weight_sha": {arm: tensor_hash(state) for arm, state in weights.items()},
                    "sampler_sha": tensor_hash({"batches": batches}), "calibration": frozen,
                    "guard_observations": distinct, "decisions": len(x), "source_counts": prior.counts,
                    "memory": tree.stats()}, handle)
    print("sealed recovery fit", sha256_file(args.fit), flush=True)


def safe_probe(state, actions):
    if state.branch >= 0:
        return actions[0]
    if state.remaining >= 5:
        available = [action for action in PROBE[:2] if action in actions]
        if available and ENTER[0] in actions:
            return available[0]
    return next(action for action in actions if action in ENTER)


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair"], row["allowance"]].append(row["proved"])
    return {"proofs": sum(row["proved"] for row in rows), "complete_pairs": sum(all(group) for group in groups.values()),
            "count": len(rows), "checks": sum(row["checks"] for row in rows),
            "utility": fsum(row["utility"] for row in rows) / len(rows)}


@torch.no_grad()
def evaluate(args, tree):
    bank, pins = registered(args)
    fitted = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fitted["schema"] != SCHEMA or fitted["pins"] != pins:
        raise ValueError("the locked recovery fit changed")
    models = models_for()
    reader = truth_head()
    for arm, model in {**models, "guard_reader": reader}.items():
        weights = fitted["weights"][arm]
        if tensor_hash(weights) != fitted["weight_sha"][arm]:
            raise ValueError("a sealed recovery actor changed")
        model.load_state_dict(weights)
        model.eval()
    prior = MixedSourcePrior(args)
    result, certificates = defaultdict(list), {}
    for case in bank["development"]:
        descriptors, confidence, warm = encode_case(case, reader, prior)
        supervised = baked_policy(confidence)
        best_branch = ENTER[0] if all(case["truth"][:2]) else ENTER[1]
        for arm in (*models, "calibrated", "supervised_planner", "safe_probe", "fold_oracle", "paid_bit_swap"):
            def select(state, actions):
                if arm == "fold_oracle":
                    return best_branch if state.branch < 0 else actions[0]
                if arm == "safe_probe":
                    return safe_probe(state, actions)
                if arm == "supervised_planner":
                    return supervised(state)
                x = torch.tensor([action_features(descriptors, confidence, warm, state, action) for action in actions])
                if arm == "calibrated":
                    values = apply_ridge(x.flatten(1), fitted["calibration"])
                else:
                    model = models["bellman_two_slot" if arm == "paid_bit_swap" else arm]
                    values = model(x)
                best = int(values.argmax())
                return actions[best] if float(values[best]) > 0. else None

            for allowance in ALLOWANCES:
                row = trace_policy(case, allowance, select, swap_after_probe=arm == "paid_bit_swap")
                tags = {"module": case["module"], "pair": case["pair"], "world": case["world"], "family": case["family"]}
                result[arm].append({**tags, **row})
                if row["proved"]:
                    certificates[case["module"], tuple(row["proof"])] = case
    natural = {(row["module"], row["allowance"]): row for row in result["bellman_two_slot"]}
    for row in result["paid_bit_swap"]:
        prefix = row["patched_at"]
        base = natural[row["module"], row["allowance"]]
        if prefix is None:
            if row != base:
                raise ValueError("an intervention changed an unpatched trajectory")
        elif row["trace"][:prefix] != base["trace"][:prefix]:
            raise ValueError("the observed-bit patch changed its paid prefix")
    with native_sources((args.bpd, args.mathcan)) as (directory, env, imports):
        independent = certify_routes(directory, env, imports, [(case, key[1]) for key, case in certificates.items()])
    summaries = {arm: summary(rows) for arm, rows in result.items()}
    for arm, row in summaries.items():
        print(arm, row, flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "results": result, "summary": summaries, "independent": independent, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "fit", "results"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
