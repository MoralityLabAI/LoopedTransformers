"""Reward-fit a recurrent recovery menu with a frozen affine stop gate."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from configs.run_ppt_failed_prefix import audit, choose, pack, public_data, summary, train
from configs.run_ppt_failed_prefix_residual import SCHEMA as AFFINE_SCHEMA
from configs.run_ppt_failed_prefix_residual import affine_joint, augment_source, fit_residual, inputs
from configs.run_ppt_guard_belief import root_contrast
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import ALLOWANCES, trace_case
from ppt_gym.failed_prefix_learning import COUNT, STARTS, STATES, returns
from ppt_gym.failed_prefix_menu import BOUND, AffineMenu, MenuActor, corrections, group_candidates, policy, rank_oracle
from ppt_gym.failed_prefix_residual import RecoveryCorrection, augment, redistribute
from ppt_gym.live_quote_policy import FEATURES
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.failed-prefix-menu.v1"
SEED = 1130
UPDATES = 360
BATCH = 16
FOLDS = 4
FLOOR = .025
INTERVENTIONS = ("chosen_menu_swap", "orthogonal_menu_sham")


def affine_correction():
    return RecoveryCorrection(nn.Sequential(nn.Flatten(1), nn.Linear(3 * FEATURES, 1)))


def models_for():
    recurrent = MenuActor()
    return {"menu_return": recurrent, "menu_q": deepcopy(recurrent),
            "menu_trm_return": MenuActor(trm=True), "menu_affine_return": AffineMenu()}


def locked(args):
    bank, pins, reader = inputs(args)
    previous = torch.load(args.correction, map_location="cpu", weights_only=True)
    state = previous["weights"]["affine_return"]
    if (previous["schema"] != AFFINE_SCHEMA or previous["pins"] != pins
            or tensor_hash(state) != previous["weight_sha"]["affine_return"]):
        raise ValueError("the archived affine return controller changed")
    model = affine_correction()
    model.load_state_dict(state)
    return bank, {**pins, "correction": sha256_file(args.correction)}, reader, model.eval()


def sample(length, rng):
    return torch.randint(length, (UPDATES, BATCH), generator=rng)


def take(data, indices):
    return tuple(tensor[indices] for tensor in data)


def fit_prior(data, indices, rng):
    inside = take(data, indices)
    reader = affine_joint()
    train(reader, "affine_joint", inside, sample(len(inside[0]), rng))
    return reader


@torch.no_grad()
def priors(reader, roots):
    return reader(roots.flatten(1)).softmax(-1).tolist()


@torch.no_grad()
def menu_inputs(raw, indices, routes, weights, affine):
    tokens, base = augment(raw, indices, routes, weights)
    delta = affine(tokens).flatten().tolist()
    values = [value + adjustment for value, adjustment in zip(base, delta, strict=True)]
    menus, states, tensor = group_candidates(tokens, indices, values)
    anchor = defaultdict(dict)
    for code, value in zip(indices, values, strict=True):
        state, arm = divmod(code, 3)
        anchor[state][arm] = value
    return menus, states, tensor, anchor


def cross_fit(source, raw, rng):
    """Hold the outer theorem pair out of both the label and reward fits."""
    fold_for = {root: i % FOLDS for i, root in enumerate(sorted({case["pair"] for case in source}))}
    labels = torch.tensor([fold_for[case["pair"]] for case in source])
    rows, fold_weights = {}, {}
    for outer in range(FOLDS):
        keep = (labels != outer).nonzero().flatten()
        held = (labels == outer).nonzero().flatten()
        nested = {}
        for inner in range(FOLDS):
            if inner == outer:
                continue
            trained = fit_prior(raw, (labels != outer) & (labels != inner), rng)
            predict = (labels == inner).nonzero().flatten()
            nested.update(zip(predict.tolist(), priors(trained, raw[2][predict]), strict=True))
        training = augment_source(take(raw, keep), [source[i] for i in keep.tolist()],
                                  [nested[i] for i in keep.tolist()])
        affine = affine_correction()
        fit_residual(affine, "affine_return", training, sample(len(keep), rng))
        reader = fit_prior(raw, keep, rng)
        fold_weights[str(outer)] = {"prior": tensor_hash(reader.state_dict()),
                                    "reward": tensor_hash(affine.state_dict())}
        for index, weights in zip(held.tolist(), priors(reader, raw[2][held]), strict=True):
            active = raw[1][index] < 3 * COUNT
            rows[index] = menu_inputs(raw[0][index, active], raw[1][index, active].tolist(),
                                      source[index]["routes"], weights, affine)
        print("cross-fitted affine policy for source fold", outer, flush=True)
    return [rows[i] for i in range(len(source))], fold_for, fold_weights


def packed_menu(rows, raw):
    width = max(len(row[1]) for row in rows)
    tokens = torch.zeros(len(rows), width, 3, 3, FEATURES)
    states = torch.full((len(rows), width), COUNT)
    base = torch.full((len(rows), width, 3), -torch.inf)
    acting = torch.zeros(len(rows), width, dtype=torch.bool)
    for i, (menus, ids, values, anchor) in enumerate(rows):
        tokens[i, :len(ids)], states[i, :len(ids)], base[i, :len(ids)] = menus, torch.tensor(ids), values
        acting[i, :len(ids)] = torch.tensor([choose(anchor[state]) is not None for state in ids])
    _, _, _, legal, links, pass_cost, fail_cost, truth, _ = raw
    return tokens, states, base, acting, legal, links, pass_cost, fail_cost, truth


def fit_menu(model, arm, data, batches):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    compatible = torch.tensor([[not bool(failed & (1 << arm)) for arm in range(3)] for _, failed in STATES]
                              + [[False] * 3])
    for batch in batches:
        tokens, states, base, acting, legal, links, pass_cost, fail_cost, truth = take(data, batch)
        delta = corrections(model, tokens, base)
        if arm == "menu_q":
            with torch.no_grad():
                target = returns(links, legal, pass_cost, fail_cost, truth)
                target = F.pad(target, (0, 0, 0, 1)).gather(1, states.unsqueeze(-1).expand_as(base))
                real = ~(truth[:, None] & ~compatible[states]).any(-1)
                mask = base.isfinite() & (acting & real).unsqueeze(-1)
            loss = F.mse_loss((base + delta)[mask], target[mask])
        else:
            loss = -returns(links, legal, pass_cost, fail_cost, truth, policy(base, delta, states, acting)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def fit(args, tree):
    bank, pins, _, _ = locked(args)
    rng = torch.Generator().manual_seed(SEED)
    source = bank["source"]
    raw = pack(source, MixedSourcePrior(args))
    rows, folds, fold_weights = cross_fit(source, raw, rng)
    data = packed_menu(rows, raw)
    batches = sample(len(source), rng)
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if initial["menu_return"] != initial["menu_q"]:
        raise ValueError("the matched menu models changed initialization")
    for arm, model in models.items():
        fit_menu(model, arm, data, batches)
        print("fitted", arm, flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    result = {"schema": SCHEMA, "pins": pins, "bound": BOUND, "weights": weights, "initial": initial,
              "weight_sha": {arm: tensor_hash(state) for arm, state in weights.items()}, "folds": folds,
              "fold_weights": fold_weights, "sampler_sha": tensor_hash({"batches": batches}), "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(result, handle)
    print("sealed joint-menu fit", sha256_file(args.fit), flush=True)


def score_case(case, allowance, anchor, residual=None, intervention=None):
    changes, patch = [], None

    def select(remaining, failed, menu):
        nonlocal patch
        state = remaining * 8 + failed
        base = anchor[state]
        original = choose(base)
        if original is None or residual is None:
            return original
        delta = {arm: residual[state][arm] for arm in menu}
        selected = max(menu, key=lambda arm: base[arm] + delta[arm])
        if selected != original:
            spent = allowance - remaining
            changes.append({"spent": spent, "base": original, "learned": selected})
            if intervention and patch is None and len(menu) == 3:
                sign = None if intervention == INTERVENTIONS[0] else (1 if int(case["module"][-1], 16) % 2 else -1)
                moved, dose = redistribute(delta, selected, original, sign)
                altered = max(menu, key=lambda arm: base[arm] + moved[arm])
                patch = {"spent": spent, "before": selected, "after": altered, "l2_dose": dose}
                return altered
        return selected

    return {**trace_case(case, allowance, select), "changes": changes, "patch": patch}


@torch.no_grad()
def evaluate(args, tree):
    bank, pins, reader, affine = locked(args)
    archive = read_receipt(args.archive)
    if (archive["schema"] != AFFINE_SCHEMA or archive["fit_sha256"] != pins["correction"]
            or archive["pins"] != {key: pins[key] for key in ("bank", "warm", "prior", "anchor")}):
        raise ValueError("the archived affine return traces changed")
    fit = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fit["schema"] != SCHEMA or fit["pins"] != pins or fit["bound"] != BOUND:
        raise ValueError("the locked menu fit changed")
    models = models_for()
    for arm, model in models.items():
        weights = fit["weights"][arm]
        if tensor_hash(weights) != fit["weight_sha"][arm]:
            raise ValueError("a frozen menu weight changed")
        model.load_state_dict(weights)
        model.eval()
    prior = MixedSourcePrior(args)
    records = defaultdict(list)
    for case in bank["development"]:
        raw, indices, _, _ = public_data(case, prior)
        raw = torch.tensor(raw)
        root = raw[[indices.index(STARTS[-1] * 3 + arm) for arm in range(3)]]
        weights, = priors(reader, root.unsqueeze(0))
        tokens, states, base, anchor = menu_inputs(raw, indices, case["routes"], weights, affine)
        residuals = {arm: dict(zip(states, corrections(model, tokens[None], base[None])[0].tolist(), strict=True))
                     for arm, model in models.items()}
        oracle = rank_oracle(case["routes"], case["truth"], anchor)
        tags = {name: case[name] for name in ("pair", "module", "family")}
        for allowance in ALLOWANCES:
            rows = {arm: score_case(case, allowance, anchor, residual) for arm, residual in residuals.items()}
            rows["affine_anchor"] = score_case(case, allowance, anchor)
            rows["rank_oracle"] = {**trace_case(case, allowance, oracle), "changes": [], "patch": None}
            natural = rows["menu_return"]["trace"]
            for arm in INTERVENTIONS:
                row = score_case(case, allowance, anchor, residuals["menu_return"], arm)
                patch = row["patch"]
                spent = patch["spent"] if patch else allowance
                prefix = lambda trace: [event for event in trace if allowance - event["before"] < spent]
                if prefix(natural) != prefix(row["trace"]) or patch is None and natural != row["trace"]:
                    raise ValueError("a menu intervention changed its paid prefix")
                rows[arm] = row
            a, b = (rows[arm]["patch"] for arm in INTERVENTIONS)
            if (a is None) != (b is None) or a and (a["spent"], a["l2_dose"]) != (b["spent"], b["l2_dose"]):
                raise ValueError("menu interventions disagree on eligibility or dose")
            for arm, row in rows.items():
                records[arm].append({**tags, **row})
    expected = archive["results"]["affine_return"]
    if any(actual["trace"] != old["trace"] or actual["utility"] != old["utility"]
           or actual["module"] != old["module"] or actual["allowance"] != old["allowance"]
           for actual, old in zip(records["affine_anchor"], expected, strict=True)):
        raise ValueError("the frozen affine return traces changed")
    if any(a["utility"] > b["utility"] + 1e-9 for a, b in zip(records["menu_return"], records["rank_oracle"], strict=True)):
        raise ValueError("a learned menu exceeded the fixed-stop envelope")
    certificates, replays = audit(args, bank["development"], records, (*models, *INTERVENTIONS),
                                  patched_arms=INTERVENTIONS)
    controls = {**records, "archived_supervised_planner": archive["results"]["frozen_planner"],
                "archived_independent_trm": archive["results"]["trm_return"]}
    primary = records["menu_return"]
    contrasts = {arm: root_contrast(primary, rows) for arm, rows in controls.items() if rows is not primary}
    results = {arm: {**summary(rows), "by_budget": {budget: summary([row for row in rows if row["allowance"] == budget])
                                                   for budget in ALLOWANCES}} for arm, rows in controls.items()}
    for arm, row in results.items():
        print(arm, row["proofs"], row["checks"], row["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit), "bound": BOUND,
                                 "archive_sha256": sha256_file(args.archive), "floor": FLOOR, "summary": results,
                                 "root_contrasts": contrasts, "results": records, "independent": certificates,
                                 "live_replays": replays, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "anchor", "correction", "fit", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--archive", type=Path)
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
