"""Fit planner-anchored recovery corrections from native proof return."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from configs.run_ppt_failed_prefix import SCHEMA as ANCHOR_SCHEMA
from configs.run_ppt_failed_prefix import audit, choose, locked, pack, public_data, summary, train
from configs.run_ppt_guard_belief import root_contrast
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import ALLOWANCES, WORLDS, trace_case
from ppt_gym.failed_prefix_learning import COUNT, STARTS, STATES, policy_distribution, returns
from ppt_gym.failed_prefix_residual import BOUND, RecoveryCorrection, augment, bounded_oracle, bounded_scores, redistribute
from ppt_gym.live_quote_policy import FEATURES, LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.failed-prefix-residual.v1"
SEED = 1129
UPDATES = 360
BATCH = 16
FOLDS = 4
FLOOR = .025
INTERVENTIONS = ("chosen_residual_swap", "orthogonal_residual_sham")


def affine_joint():
    return nn.Linear(9 * FEATURES, len(WORLDS))


def inputs(args):
    bank, pins = locked(args)
    archive = torch.load(args.anchor, map_location="cpu", weights_only=True)
    weights = archive["weights"]["affine_joint"]
    if (archive["schema"] != ANCHOR_SCHEMA or archive["pins"] != pins
            or tensor_hash(weights) != archive["weight_sha"]["affine_joint"]):
        raise ValueError("the archived affine planner changed")
    model = affine_joint()
    model.load_state_dict(weights)
    return bank, {**pins, "anchor": sha256_file(args.anchor)}, model.eval()


def models_for():
    recurrent = RecoveryCorrection(TwoSlotQuoteActor())
    return {"two_slot_return": recurrent, "two_slot_q": deepcopy(recurrent),
            "trm_return": RecoveryCorrection(LiveQuoteTRM()),
            "affine_return": RecoveryCorrection(nn.Sequential(nn.Flatten(1), nn.Linear(3 * FEATURES, 1)))}


def out_of_fold(source, data, rng):
    folds = {root: index % FOLDS for index, root in enumerate(sorted({case["pair"] for case in source}))}
    labels = torch.tensor([folds[case["pair"]] for case in source])
    probabilities = torch.empty(len(source), len(WORLDS))
    for fold in range(FOLDS):
        kept = [feature[labels != fold] for feature in data]
        batches = torch.randint(len(kept[0]), (UPDATES, BATCH), generator=rng)
        reader = affine_joint()
        train(reader, "affine_joint", kept, batches)
        with torch.no_grad():
            probabilities[labels == fold] = reader(data[2][labels == fold].flatten(1)).softmax(dim=-1)
    return probabilities.tolist(), folds


def augment_source(rows, cases, probabilities):
    tokens, codes, _, menu, edges, pass_cost, fail_cost, truth, _ = rows
    augmented = tokens.clone()
    dense = torch.full(menu.shape, -torch.inf)
    for row, (case, weights) in enumerate(zip(cases, probabilities, strict=True)):
        active = codes[row] < 3 * COUNT
        indices = codes[row, active]
        features, scores = augment(tokens[row, active], indices.tolist(), case["routes"], weights)
        augmented[row, active] = features
        dense[row].flatten()[indices] = torch.tensor(scores)
    return augmented, codes, menu, edges, pass_cost, fail_cost, truth, dense


def fit_residual(model, arm, data, batches):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    compatible = torch.tensor([[not bool(failed & (1 << index)) for index in range(3)] for _, failed in STATES])
    for indices in batches:
        features, codes, menu, edges, pass_cost, fail_cost, truth, base = [row[indices] for row in data]
        scores = bounded_scores(model, features, codes, base)
        if arm == "two_slot_q":
            target = returns(edges, menu, pass_cost, fail_cost, truth)
            reachable = ~(truth[:, None] & ~compatible).any(-1)
            mask = menu & reachable.unsqueeze(-1)
            loss = F.mse_loss(scores[mask], target[mask])
        else:
            policy = policy_distribution(scores)
            loss = -returns(edges, menu, pass_cost, fail_cost, truth, policy).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def fit(args, tree):
    bank, pins, _ = inputs(args)
    source = bank["source"]
    rng = torch.Generator().manual_seed(SEED)
    raw = pack(source, MixedSourcePrior(args))
    prior, folds = out_of_fold(source, raw, rng)
    data = augment_source(raw, source, prior)
    batches = torch.randint(len(source), (UPDATES, BATCH), generator=rng)
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if initial["two_slot_return"] != initial["two_slot_q"]:
        raise ValueError("the matched recurrent controls changed initialization")
    for arm, model in models.items():
        fit_residual(model, arm, data, batches)
        print("fitted", arm, flush=True)
    weights = {name: model.state_dict() for name, model in models.items()}
    result = {"schema": SCHEMA, "pins": pins, "bound": BOUND, "folds": folds, "weights": weights, "initial": initial,
              "weight_sha": {arm: tensor_hash(value) for arm, value in weights.items()},
              "sampler_sha": tensor_hash({"batches": batches}), "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(result, handle)
    print("sealed recovery correction", sha256_file(args.fit), flush=True)


def score_case(case, allowance, positions, base, residual=None, intervention=None):
    changes, patch = [], None

    def select(remaining, failed, menu):
        nonlocal patch
        values = {arm: base[positions[(remaining * 8 + failed) * 3 + arm]] for arm in menu}
        frozen = choose(values)
        if residual is None:
            return frozen
        delta = {arm: residual[positions[(remaining * 8 + failed) * 3 + arm]] for arm in menu}
        elected = choose({arm: values[arm] + delta[arm] for arm in menu})
        if elected != frozen:
            spent = allowance - remaining
            changes.append({"spent": spent, "base": frozen, "learned": elected})
            if intervention and patch is None and len(menu) == 3 and elected is not None and frozen is not None:
                sign = None if intervention == INTERVENTIONS[0] else (1 if int(case["module"][-1], 16) % 2 else -1)
                shifted, dose = redistribute(delta, elected, frozen, sham_sign=sign)
                altered = choose({arm: values[arm] + shifted[arm] for arm in menu})
                patch = {"spent": spent, "before": elected, "after": altered, "l2_dose": dose}
                return altered
        return elected

    return {**trace_case(case, allowance, select), "changes": changes, "patch": patch}


@torch.no_grad()
def evaluate(args, tree):
    bank, pins, reader = inputs(args)
    archive = read_receipt(args.archive)
    if (archive["schema"] != ANCHOR_SCHEMA or archive["pins"] != {key: pins[key] for key in ("bank", "warm", "prior")}
            or archive["fit_sha256"] != pins["anchor"]):
        raise ValueError("the frozen reference traces belong to a different recovery fit")
    fitted = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fitted["schema"] != SCHEMA or fitted["pins"] != pins or fitted["bound"] != BOUND:
        raise ValueError("the locked recovery correction changed")
    models = models_for()
    for arm, model in models.items():
        weights = fitted["weights"][arm]
        if tensor_hash(weights) != fitted["weight_sha"][arm]:
            raise ValueError("the frozen recovery correction changed")
        model.load_state_dict(weights)
        model.eval()
    prior = MixedSourcePrior(args)
    records = defaultdict(list)
    interventions = {"two_slot_return": INTERVENTIONS}
    if args.affine_intervention:
        interventions["affine_return"] = tuple(f"affine_{arm}" for arm in INTERVENTIONS)
    intervention_arms = tuple(arm for group in interventions.values() for arm in group)
    for case in bank["development"]:
        raw, indices, _, _ = public_data(case, prior)
        raw = torch.tensor(raw)
        positions = {code: row for row, code in enumerate(indices)}
        root = raw[[positions[STARTS[-1] * 3 + arm] for arm in range(3)]]
        weights = reader(root.reshape(1, -1)).softmax(dim=-1)[0].tolist()
        tokens, base = augment(raw, indices, case["routes"], weights)
        residuals = {arm: model(tokens).flatten().tolist() for arm, model in models.items()}
        oracle = bounded_oracle(case["routes"], weights, case["truth"])
        tags = {name: case[name] for name in ("pair", "module", "family")}
        for allowance in ALLOWANCES:
            rows = {arm: score_case(case, allowance, positions, base, residual) for arm, residual in residuals.items()}
            rows["frozen_planner"] = score_case(case, allowance, positions, base)
            rows["bounded_oracle"] = {**trace_case(case, allowance, oracle), "changes": [], "patch": None}
            for parent, names in interventions.items():
                natural = rows[parent]
                for arm, mode in zip(names, INTERVENTIONS, strict=True):
                    row = score_case(case, allowance, positions, base, residuals[parent], mode)
                    patch = row["patch"]
                    if patch is None and natural["trace"] != row["trace"]:
                        raise ValueError("an unpatched recovery changed")
                    if patch:
                        left = [event for event in natural["trace"] if allowance - event["before"] < patch["spent"]]
                        right = [event for event in row["trace"] if allowance - event["before"] < patch["spent"]]
                        if left != right:
                            raise ValueError("the recovery intervention changed its paid prefix")
                    rows[arm] = row
                a, b = (rows[arm]["patch"] for arm in names)
                if (a is None) != (b is None) or a and (a["spent"], a["l2_dose"]) != (b["spent"], b["l2_dose"]):
                    raise ValueError("the recovery interventions disagree on eligibility or physical dose")
            for arm, row in rows.items():
                records[arm].append({**tags, **row})
    baseline = [{**row, "changes": []} for row in archive["results"]["affine_joint"]]
    if baseline != records["frozen_planner"]:
        raise ValueError("the frozen affine planner did not reproduce its archived trajectories")
    if any(a["utility"] > b["utility"] + 1e-9 for a, b in zip(records["two_slot_return"], records["bounded_oracle"], strict=True)):
        raise ValueError("a learned policy exceeded the correction envelope")
    certificates, replays = audit(args, bank["development"], records, arms=(*models, *intervention_arms),
                                  patched_arms=intervention_arms)
    controls = {**records, "archived_two_slot_return": archive["results"]["two_slot_return"]}
    primary = records["two_slot_return"]
    contrasts = {arm: root_contrast(primary, rows) for arm, rows in controls.items() if rows is not primary}
    comparisons = {arm: {**summary(rows), "by_budget": {budget: summary([row for row in rows if row["allowance"] == budget])
                                                       for budget in ALLOWANCES}} for arm, rows in controls.items()}
    for arm, row in comparisons.items():
        print(arm, row["proofs"], row["checks"], row["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit), "bound": BOUND,
                                 "floor": FLOOR, "archive_sha256": sha256_file(args.archive), "summary": comparisons,
                                 "root_contrasts": contrasts, "results": records, "independent": certificates,
                                 "live_replays": replays, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "anchor", "fit", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--affine-intervention", action="store_true",
                        help="also score the post-hoc equal-dose controls on the affine return policy")
    args = parser.parse_args()
    if args.phase == "eval" and args.archive is None:
        parser.error("eval requires the audited --archive")
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
