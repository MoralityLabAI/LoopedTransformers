"""Compare guarded Bellman beliefs with matched supervision and calibration."""

import argparse
from collections import defaultdict
from copy import deepcopy
from math import exp, fsum, log, sqrt
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.freeze_ppt_guard_recovery import certify_routes, replay_recovery
from configs.freeze_ppt_native_progress import native_sources
from configs.run_ppt_guard_recovery import encode_case, summary
from configs.run_ppt_guard_residual import SCHEMA as RESIDUAL_SCHEMA
from configs.run_ppt_guard_residual import cross_fit, load_inputs, scaled
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.guard_belief import LOGIT_BOUND, BeliefCorrection, BeliefPlanner, belief_inputs
from ppt_gym.guard_recovery import ACTIONS, ALLOWANCES, feasible_menu
from ppt_gym.guard_recovery_learning import (GUARD_WORLDS, baked_scores, false_guard_weights,
                                             posterior_scores, source_targets, trace_policy)
from ppt_gym.live_quote_policy import LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.guard-belief.v1"
SEED = 1123
UPDATES = 600
BATCH = 32
SWAPS = {"witness_swap": (2, 3, 0, 1), "within_witness_swap": (1, 0, 3, 2)}
FLOOR = .025


def models_for():
    standard = BeliefCorrection(TwoSlotQuoteActor())
    return {"bellman_belief": standard, "empirical_belief": deepcopy(standard),
            "supervised_belief": deepcopy(standard), "bellman_trm": BeliefCorrection(LiveQuoteTRM()),
            "affine_calibration": BeliefCorrection()}


def prepare(source, cache, planner):
    features, priors, labels, decisions, backed, empirical = [], [], [], [], [], []
    for index, case in enumerate(source):
        descriptors, probability, warm = cache[case["module"]]
        features.append(belief_inputs(descriptors, probability, warm))
        priors.append([log(weight) for weight in false_guard_weights(probability)])
        labels.append(GUARD_WORLDS.index(tuple(case["truth"])))
        for state, move, long_value, short_value in source_targets(case):
            if move in feasible_menu(state):
                decisions.append((index, planner.index[state], ACTIONS.index(move)))
                backed.append(long_value)
                empirical.append(short_value)
    return tuple(torch.tensor(rows) for rows in (features, priors, labels, decisions, backed, empirical))


def train(model, arm, planner, data, batches):
    features, priors, labels, decisions, backed, empirical = data
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    rows = torch.arange(batches.shape[1])
    for indices in batches:
        cases, states, actions = decisions[indices].unbind(1)
        belief = model(features[cases], priors[cases])
        if arm in ("supervised_belief", "affine_calibration"):
            loss = F.cross_entropy(belief, labels[cases])
        else:
            score = planner.scores(belief)[rows, states, actions]
            target = empirical if arm == "empirical_belief" else backed
            loss = F.mse_loss(score, target[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


def fit(args, tree):
    bank, pins, _ = load_inputs(args)
    source = bank["source"]
    rng = torch.Generator().manual_seed(SEED)
    cache, folds, temperature, losses = cross_fit(source, MixedSourcePrior(args), rng)
    planner = BeliefPlanner()
    data = prepare(source, cache, planner)
    batches = torch.randint(len(data[3]), (UPDATES, BATCH), generator=rng)
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if len({initial[arm] for arm in ("bellman_belief", "empirical_belief", "supervised_belief")}) != 1:
        raise ValueError("the matched two-slot arms changed initialization")
    for arm, model in models.items():
        train(model, arm, planner, data, batches)
        print("fitted", arm, flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    receipt = {"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial, "folds": folds,
               "weight_sha": {arm: tensor_hash(value) for arm, value in weights.items()},
               "sampler_sha": tensor_hash({"batches": batches}), "temperature": temperature,
               "temperature_losses": losses, "decisions": len(data[3]), "states": len(planner.states),
               "logit_bound": LOGIT_BOUND, "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(receipt, handle)
    print("sealed belief fit", sha256_file(args.fit), flush=True)


def policy_scores(weights, correction):
    return posterior_scores([base * exp(delta) for base, delta in zip(weights, correction, strict=True)],
                            action_space=feasible_menu)


def pick(scores):
    action = max(scores, key=scores.__getitem__)
    return action if scores[action] > 0. else None


def score_case(case, allowance, natural, base, counterfactual=None):
    changes, patch = [], None

    def select(state, _menu):
        nonlocal patch
        selected, frozen = pick(natural(state)), pick(base(state))
        if selected != frozen:
            step = allowance - 2 - state.remaining
            changes.append({"step": step, "native": frozen, "learned": selected})
            if counterfactual is not None and patch is None:
                altered = pick(counterfactual(state))
                patch = {"step": step, "before": selected, "after": altered}
                return altered
        return selected

    trace = trace_policy(case, allowance, select, action_space=feasible_menu)
    return {**trace, "changes": changes, "patch": patch}


def audit_routes(args, development, records, certificates, arms=("bellman_belief", "supervised_belief", *SWAPS)):
    cases = {case["module"]: case for case in development}
    replays = []
    with native_sources((args.bpd, args.mathcan)) as (directory, env, imports):
        independent = certify_routes(directory, env, imports, [(case, key[1]) for key, case in certificates.items()])
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native Lean toolchain")
            for arm in arms:
                for proved in (True, False):
                    row = next((r for r in records[arm] if r["proved"] == proved and r["changes"]), None)
                    if row is not None:
                        replay_recovery(repl, cases[row["module"]], row)
                        replays.append({"arm": arm, "module": row["module"], "proved": proved,
                                        "allowance": row["allowance"]})
    return independent, replays


def root_contrast(candidate, control):
    groups = defaultdict(list)
    for a, b in zip(candidate, control, strict=True):
        key = ("module", "pair", "allowance")
        if tuple(a[k] for k in key) != tuple(b[k] for k in key):
            raise ValueError("the root comparison lost its paired scenarios")
        groups[a["pair"]].append(a["utility"] - b["utility"])
    roots = {root: fsum(rows) / len(rows) for root, rows in groups.items()}
    return {"effects": roots, "mean": fsum(roots.values()) / len(roots),
            "wins": sum(v > 1e-9 for v in roots.values()), "losses": sum(v < -1e-9 for v in roots.values())}


@torch.no_grad()
def evaluate(args, tree):
    bank, pins, reader = load_inputs(args)
    archive = read_receipt(args.action_control)
    if archive["schema"] != RESIDUAL_SCHEMA or archive["pins"] != pins or archive.get("pairwise", False):
        raise ValueError("the archived bounded action control uses a different pilot")
    fitted = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fitted["schema"] != SCHEMA or fitted["pins"] != pins or fitted["logit_bound"] != LOGIT_BOUND:
        raise ValueError("the locked belief fit changed")
    models = models_for()
    for arm, model in models.items():
        weights = fitted["weights"][arm]
        if tensor_hash(weights) != fitted["weight_sha"][arm]:
            raise ValueError("the frozen belief correction changed")
        model.load_state_dict(weights)
        model.eval()
    prior = MixedSourcePrior(args)
    records, certificates, beliefs = defaultdict(list), {}, {}
    for case in bank["development"]:
        descriptors, probability, warm = encode_case(case, reader, prior)
        weights = false_guard_weights(probability)
        tokens = torch.tensor([belief_inputs(descriptors, probability, warm)])
        corrections = {arm: model.correction(tokens)[0].tolist() for arm, model in models.items()}
        scores = {arm: policy_scores(weights, delta) for arm, delta in corrections.items()}
        base = baked_scores(probability, action_space=feasible_menu)
        scores["frozen_planner"] = base
        scores["temperature_calibration"] = baked_scores(scaled(probability, fitted["temperature"]),
                                                         action_space=feasible_menu)
        delta = corrections["bellman_belief"]
        patches = {arm: policy_scores(weights, [delta[i] for i in order]) for arm, order in SWAPS.items()}
        doses = {arm: sqrt(fsum((delta[i] - delta[j]) ** 2 for i, j in enumerate(order)))
                 for arm, order in SWAPS.items()}
        beliefs[case["module"]] = {"prior": weights, "corrections": corrections, "doses": doses,
                                    "false_guard": GUARD_WORLDS.index(tuple(case["truth"]))}
        tags = {name: case[name] for name in ("module", "pair", "family")}
        selectors = {arm: (value, None) for arm, value in scores.items()}
        selectors.update({arm: (scores["bellman_belief"], value) for arm, value in patches.items()})
        for allowance in ALLOWANCES:
            natural = None
            for arm, (scores_at, intervention) in selectors.items():
                row = score_case(case, allowance, scores_at, base, intervention)
                if arm == "bellman_belief":
                    natural = row
                if intervention is not None:
                    patch = row["patch"]
                    step = patch["step"] if patch else len(natural["trace"])
                    if (row["trace"][:step] != natural["trace"][:step]
                            or patch is None and row["trace"] != natural["trace"]):
                        raise ValueError("a belief intervention changed the paid prefix")
                records[arm].append({**tags, **row})
                if row["proved"]:
                    certificates[case["module"], tuple(row["proof"])] = case
        for query in (*scores.values(), *patches.values()):
            query.cache_clear()
    if records["frozen_planner"] != [{**row, "changes": [], "patch": None}
                                     for row in recorded_base(archive)]:
        raise ValueError("the frozen reference traces changed")
    independent, replays = audit_routes(args, bank["development"], records, certificates)
    primary = records["bellman_belief"]
    contrasts = {arm: root_contrast(primary, rows) for arm, rows in records.items() if rows is not primary}
    controls = {**records, "archived_action_residual": archive["results"]["bellman_residual"]}
    contrasts["archived_action_residual"] = root_contrast(primary, controls["archived_action_residual"])
    results = {arm: {**summary(rows), "by_budget": {budget: summary([r for r in rows if r["allowance"] == budget])
                                                   for budget in ALLOWANCES}} for arm, rows in controls.items()}
    for arm, values in results.items():
        print(arm, values["proofs"], values["checks"], values["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "action_control_sha256": sha256_file(args.action_control),
                                 "fitting": {k: fitted[k] for k in ("initial", "weight_sha", "sampler_sha",
                                                                   "temperature", "decisions", "states", "memory")},
                                 "floor": FLOOR, "summary": results, "root_contrasts": contrasts, "beliefs": beliefs,
                                 "results": records, "independent": independent, "live_replays": replays,
                                 "memory": tree.stats()})


def recorded_base(archive):
    for row in archive["results"]["frozen_planner"]:
        yield {key: value for key, value in row.items() if key not in ("residual_changes", "swap_at")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "planner", "fit", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--action-control", type=Path)
    args = parser.parse_args()
    if args.phase == "eval" and args.action_control is None:
        parser.error("eval requires the audited 1109 --action-control receipt")
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
