"""Train a tactic-family correction from exact source policy returns."""

import argparse
from collections import defaultdict
from copy import deepcopy
from math import log
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.run_ppt_guard_belief import SCHEMA as BELIEF_SCHEMA
from configs.run_ppt_guard_belief import audit_routes, pick, root_contrast
from configs.run_ppt_guard_recovery import encode_case, summary
from configs.run_ppt_guard_residual import SCHEMA as RESIDUAL_SCHEMA
from configs.run_ppt_guard_residual import cross_fit, load_inputs, scaled
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import sha256_file
from ppt_gym.guard_belief import belief_inputs
from ppt_gym.guard_recovery import ACTIONS, ALLOWANCES, CHARGE, PROBE, VERIFY, feasible_menu
from ppt_gym.guard_recovery_learning import GUARD_WORLDS, baked_scores, false_guard_weights, supports, trace_policy
from ppt_gym.guard_return import DOSE, TEMPERATURE, ReturnPlanner, TacticCorrection
from ppt_gym.live_quote_policy import LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.guard-policy-return.v1"
SEED = 1124
UPDATES = 600
BATCH = 24
FLOOR = .025
CHECK_CONTROLS = ("two_slot_q", "trm_return", "affine_return")


def models_for():
    actor = TacticCorrection(TwoSlotQuoteActor())
    return {"two_slot_return": actor, "two_slot_q": deepcopy(actor),
            "trm_return": TacticCorrection(LiveQuoteTRM()), "affine_return": TacticCorrection()}


@torch.no_grad()
def prepare(cases, cache, planner):
    tokens, priors, truths = [], [], []
    for case in cases:
        descriptors, probability, warm = cache[case["module"]]
        tokens.append(belief_inputs(descriptors, probability, warm))
        priors.append([log(weight) for weight in false_guard_weights(probability)])
        truths.append(GUARD_WORLDS.index(tuple(case["truth"])))
    tokens, priors, truths = (torch.tensor(values) for values in (tokens, priors, truths))
    base = planner.scores(priors)
    targets = planner.rollout(truths)
    consistent = torch.tensor([[False] + [supports(state, world) for state in planner.states] for world in GUARD_WORLDS])
    relevant = (consistent[truths] & planner.unresolved).unsqueeze(-1) & base.isfinite()
    return tokens, truths, base, targets, relevant


def train(model, arm, planner, data, batches):
    tokens, truths, base, targets, relevant = data
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for indices in batches:
        values = base[indices] + planner.addends(model(tokens[indices]))
        if arm == "two_slot_q":
            mask = relevant[indices]
            loss = F.mse_loss(values[mask], targets[indices][mask])
        else:
            loss = -planner.rollout(truths[indices], values).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        values = base + planner.addends(model(tokens))
        return (planner.rollout(truths, values).mean(dim=0) - 2 * CHARGE).tolist()


def fit(args, tree):
    bank, pins, _ = load_inputs(args)
    rng = torch.Generator().manual_seed(SEED)
    cache, folds, temperature, losses = cross_fit(bank["source"], MixedSourcePrior(args), rng)
    planner = ReturnPlanner()
    data = prepare(bank["source"], cache, planner)
    batches = torch.randint(len(data[0]), (UPDATES, BATCH), generator=rng)
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if initial["two_slot_return"] != initial["two_slot_q"]:
        raise ValueError("the return and Q objectives do not have matched initial weights")
    source_utility = {"frozen_planner": (planner.rollout(data[1], data[2]).mean(dim=0) - 2 * CHARGE).tolist()}
    for arm, model in models.items():
        source_utility[arm] = train(model, arm, planner, data, batches)
        print("fitted", arm, source_utility[arm], flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    receipt = {"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial,
               "weight_sha": {arm: tensor_hash(value) for arm, value in weights.items()},
               "sampler_sha": tensor_hash({"batches": batches}), "temperature": temperature,
               "temperature_losses": losses, "folds": folds, "source_soft_utility": source_utility,
               "states": len(planner.states), "dose": DOSE, "softmax": TEMPERATURE, "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(receipt, handle)
    print("sealed tactic correction", sha256_file(args.fit), flush=True)


def score_case(case, allowance, planner, base, addends=None, swap=None):
    changes, patch = [], None

    def choose(state, menu):
        nonlocal patch
        scores = base(state)
        frozen = pick(scores)
        if addends is None:
            return frozen
        state_row = addends[planner.index[state]]
        residuals = {move: state_row[ACTIONS.index(move)] for move in menu}
        action = pick({move: scores[move] + residuals[move] for move in menu})
        if action != frozen:
            step = allowance - 2 - state.remaining
            changes.append({"step": step, "native": frozen, "learned": action, "remaining": state.remaining})
            same_check = any(action in family and frozen in family for family in (PROBE, VERIFY))
            if (patch is None and action is not None and frozen is not None
                    and (swap == "action_swap" or swap == "check_swap" and same_check)):
                residuals[action], residuals[frozen] = residuals[frozen], residuals[action]
                altered = pick({move: scores[move] + residuals[move] for move in menu})
                patch = {"step": step, "before": action, "after": altered,
                         "l2_dose": 2 ** .5 * abs(residuals[action] - residuals[frozen])}
                return altered
        return action

    trace = trace_policy(case, allowance, choose, action_space=feasible_menu)
    return {**trace, "changes": changes, "patch": patch}


def checked_archive(path, schema, pins):
    archive = read_receipt(path)
    if archive["schema"] != schema or archive["pins"] != pins or archive.get("pairwise", False):
        raise ValueError("the archived comparator belongs to a different pilot")
    return archive


@torch.no_grad()
def evaluate(args, tree):
    bank, pins, reader = load_inputs(args)
    action_archive = checked_archive(args.action_control, RESIDUAL_SCHEMA, pins)
    belief_archive = checked_archive(args.belief_control, BELIEF_SCHEMA, pins)
    fitted = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (fitted["schema"] != SCHEMA or fitted["pins"] != pins or fitted["dose"] != DOSE
            or fitted["softmax"] != TEMPERATURE):
        raise ValueError("the locked return fit changed")
    models = models_for()
    for arm, model in models.items():
        weights = fitted["weights"][arm]
        if tensor_hash(weights) != fitted["weight_sha"][arm]:
            raise ValueError("a frozen tactic correction changed")
        model.load_state_dict(weights)
        model.eval()
    planner = ReturnPlanner()
    prior = MixedSourcePrior(args)
    records, certificates, heads = defaultdict(list), {}, {}
    for case in bank["development"]:
        descriptors, probability, warm = encode_case(case, reader, prior)
        tokens = torch.tensor([belief_inputs(descriptors, probability, warm)])
        outputs = {arm: model(tokens) for arm, model in models.items()}
        full = outputs["two_slot_return"]
        only_commit = full.clone()
        only_commit[..., 2:] = 0
        only_check = full.clone()
        only_check[..., :2] = 0
        heads[case["module"]] = {arm: block[0].tolist() for arm, block in outputs.items()}
        if args.check_controls:
            for arm in CHECK_CONTROLS:
                check = outputs[arm].clone()
                check[..., :2] = 0
                outputs[f"{arm}_check_only"] = check
            outputs["check_swap"] = only_check
        outputs.update({"commit_only": only_commit, "check_only": only_check, "action_swap": full})
        addends = {arm: planner.addends(block)[0].tolist() for arm, block in outputs.items()}
        base = baked_scores(probability, action_space=feasible_menu)
        calibrated = baked_scores(scaled(probability, fitted["temperature"]), action_space=feasible_menu)
        tags = {name: case[name] for name in ("module", "pair", "family")}
        for allowance in ALLOWANCES:
            natural = {}
            for arm in (*addends, "frozen_planner", "temperature_calibration"):
                scores = calibrated if arm == "temperature_calibration" else base
                row = score_case(case, allowance, planner, scores, addends.get(arm), arm)
                natural[arm] = row
            for arm, row in natural.items():
                if arm in ("action_swap", "check_swap"):
                    parent = natural["two_slot_return" if arm == "action_swap" else "check_only"]
                    patch = row["patch"]
                    step = patch["step"] if patch else len(parent["trace"])
                    if (row["trace"][:step] != parent["trace"][:step]
                            or patch is None and row["trace"] != parent["trace"]):
                        raise ValueError("a tactic intervention changed the paid prefix")
                records[arm].append({**tags, **row})
                if row["proved"]:
                    certificates[case["module"], tuple(row["proof"])] = case
        base.cache_clear()
        calibrated.cache_clear()
    if records["frozen_planner"] != belief_archive["results"]["frozen_planner"]:
        raise ValueError("the frozen reference traces changed")
    arms = ("two_slot_return", "trm_return", "commit_only", "check_only", "action_swap")
    if args.check_controls:
        arms += ("check_swap",)
    independent, replays = audit_routes(args, bank["development"], records, certificates, arms)
    primary = records["two_slot_return"]
    archived_action = action_archive["results"]["bellman_residual"]
    archived_belief = belief_archive["results"]["supervised_belief"]
    archived_routed = [action if action["allowance"] == 6 else belief
                       for action, belief in zip(archived_action, archived_belief, strict=True)]
    controls = {**records, "archived_action_residual": archived_action,
                "archived_supervised_belief": archived_belief, "archived_budget_router": archived_routed}
    contrasts = {arm: root_contrast(primary, rows) for arm, rows in controls.items() if rows is not primary}
    results = {arm: {**summary(rows), "by_budget": {budget: summary([r for r in rows if r["allowance"] == budget])
                                                   for budget in ALLOWANCES}} for arm, rows in controls.items()}
    for arm, row in results.items():
        print(arm, row["proofs"], row["checks"], row["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "archives": {k: sha256_file(getattr(args, k)) for k in ("action_control", "belief_control")},
                                 "fitting": {k: fitted[k] for k in ("initial", "weight_sha", "sampler_sha", "temperature",
                                                                   "source_soft_utility", "states", "memory")},
                                 "posthoc_check_controls": args.check_controls,
                                 "floor": FLOOR, "summary": results, "root_contrasts": contrasts, "heads": heads,
                                 "results": records, "independent": independent, "live_replays": replays,
                                 "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "planner", "fit", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--action-control", type=Path)
    parser.add_argument("--belief-control", type=Path)
    parser.add_argument("--check-controls", action="store_true",
                        help="add post-hoc check-only readouts for the matched controls")
    args = parser.parse_args()
    if args.phase == "eval" and (args.action_control is None or args.belief_control is None):
        parser.error("eval requires the audited --action-control and --belief-control receipts")
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
