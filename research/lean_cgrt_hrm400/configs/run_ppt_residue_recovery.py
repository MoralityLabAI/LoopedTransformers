"""Train a residue posterior from observed Lean proof return."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from configs.freeze_ppt_residue_recovery import COUNTS
from configs.run_ppt_guard_belief import root_contrast
from configs.run_ppt_live_quote_fit import warm_values
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_paired_recovery import manipulate, replays, summarize
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, REPL_EXE
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import ALLOWANCES, CHARGE, WORLDS, legal, planned_values
from ppt_gym.failed_prefix_learning import policy_distribution, recovery_graph, returns
from ppt_gym.live_quote_policy import FEATURES, LiveQuoteTRM, TwoSlotQuoteActor, pad_features
from ppt_gym.paired_recovery import STOP, one_trial, solve_target, stop_or_pick
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.residue_recovery import FIRST_BUDGET, MODULUS, SCHEMA as BANK_SCHEMA
from ppt_gym.residue_recovery import compatibility, route_of, world_weights
from ppt_gym.rewritten_recovery import planned_batch


SCHEMA = "ppt.residue-recovery-fit.v1"
SEED = 1133
UPDATES = 400
BATCH = 16
FLOOR = .025
INTERVENTIONS = ("chosen_residue_swap", "orthogonal_residue_sham")


def locked(args):
    bank = read_receipt(args.bank)
    if (bank["schema"] != BANK_SCHEMA or bank["lean"] != sha256_file(LEAN_EXE) or bank["repl"] != sha256_file(REPL_EXE)
            or bank["imports"] != {path.name: sha256_file(path) for path in (args.bpd, args.mathcan)}
            or any(len(bank[name]) != count or not all(row["certified_residue"] for row in bank[name])
                   for name, count in COUNTS.items())
            or {pair["pair"] for pair in bank["source"]} & {pair["pair"] for pair in bank["development"]}):
        raise ValueError("the native residue bank or runtime changed")
    return bank, {name: sha256_file(getattr(args, name)) for name in ("bank", "warm", "prior")}


def models_for():
    actor = TwoSlotQuoteActor()
    affine = nn.Sequential(nn.Flatten(1), nn.Linear(3 * FEATURES, 1))
    return {"two_slot_return": actor, "two_slot_supervised": deepcopy(actor),
            "trm_return": LiveQuoteTRM(), "affine_return": affine, "affine_supervised": deepcopy(affine)}


def curves(case, prior):
    saved = warm_values(prior, case["saved"], ("apply hr0",))
    restarts = warm_values(prior, case["fork"], ("apply hr1", "apply hr2"))
    return [saved["apply hr0"], restarts["apply hr1"], restarts["apply hr2"]]


def source_trial(first):
    values = {STOP: 0.}
    for arm in legal(first["routes"], FIRST_BUDGET, 0):
        route = first["routes"][arm]
        p = len(route["residues"]) / MODULUS
        values[arm] = p * (1 - CHARGE * route["pass_cost"]) - (1 - p) * CHARGE * route["fail_cost"]
    choice = max(values, key=values.__getitem__)
    row = one_trial(first, FIRST_BUDGET, choice)
    return choice, (row["proved"] if choice != STOP else None), row


def joint_uniform_first(first, second, allowance):
    """Choose the next source attempt using uniform residue probabilities and target costs."""
    def suffix(compatible):
        count = sum(compatible)
        weights = world_weights(second, [float(ok) / count for ok in compatible], WORLDS)
        forecast = planned_values(second["routes"], weights)
        return max((0., *forecast(allowance, 0).values()))

    values = {STOP: suffix([True] * MODULUS)}
    for arm in legal(first["routes"], FIRST_BUDGET, 0):
        route = first["routes"][arm]
        positive = compatibility(first, arm, True)
        negative = compatibility(first, arm, False)
        probability = sum(positive) / MODULUS
        success = 1. - CHARGE * route["pass_cost"] + suffix(positive)
        failure = -CHARGE * route["fail_cost"] + suffix(negative)
        values[arm] = probability * success + (1. - probability) * failure
    choice = max(values, key=values.__getitem__)
    trace = one_trial(first, FIRST_BUDGET, choice)
    supported = compatibility(first, choice, trace["proved"] if choice != STOP else None)
    probabilities = [float(ok) / sum(supported) for ok in supported]
    return trace, probabilities


def public_tokens(pair, choice, observed, warm_a, warm_b):
    a, b = pair["first"], pair["second"]
    left, right = route_of(a), route_of(b)
    support = compatibility(a, choice, observed)
    rows = []
    for residue, (source, target) in enumerate(zip(left, right, strict=True)):
        shared = [residue / (MODULUS - 1), pair["base"] / 8, pair["depth"] / 64, pair["family"] == "BPD.geom"]

        def describe(case, arm, warm):
            route = case["routes"][arm]
            membership = [float(index in route["residues"]) for index in range(MODULUS)]
            return pad_features([*shared, route["pass_cost"] / 7, route["fail_cost"] / 7,
                                 arm == 0, arm == 1, *membership, *warm[arm]])

        context = [*shared, choice == source, observed is not None, bool(observed), sum(support) / MODULUS,
                   len(a["routes"][source]["residues"]) / MODULUS, len(b["routes"][target]["residues"]) / MODULUS]
        rows.append([describe(a, source, warm_a), describe(b, target, warm_b), pad_features(context)])
    return rows, support


def projection(target):
    return [world_weights(target, tuple(float(residue == i) for residue in range(MODULUS)), WORLDS)
            for i in range(MODULUS)]


def prepare(pairs, prior):
    entries = []
    for pair in pairs:
        choice, observed, _ = source_trial(pair["first"])
        target = pair["second"]
        rows, mask = public_tokens(pair, choice, observed, curves(pair["first"], prior), curves(target, prior))
        _, menu, edges = recovery_graph(target["routes"])
        success = [float(route["pass_cost"]) for route in target["routes"]]
        failed = [float(route["fail_cost"]) for route in target["routes"]]
        entries.append((rows, mask, projection(target), menu, edges, success, failed, target["truth"], pair["residue"]))
    return tuple(torch.tensor(values) for values in zip(*entries, strict=True))


def train(model, arm, data, batches):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for batch in batches:
        tokens, support, mapping, menu, links, pass_cost, fail_cost, truth, label = (value[batch] for value in data)
        logits = model(tokens.flatten(0, 1)).reshape_as(support).masked_fill(~support, -torch.inf)
        if arm.endswith("supervised"):
            loss = F.cross_entropy(logits, label)
        else:
            weights = (logits.softmax(-1).unsqueeze(-1) * mapping).sum(1)
            scores = planned_batch(links, menu, pass_cost, fail_cost, weights)
            loss = -returns(links, menu, pass_cost, fail_cost, truth, policy_distribution(scores)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def fit(args, tree):
    bank, pins = locked(args)
    data = prepare(bank["source"], MixedSourcePrior(args))
    rng = torch.Generator().manual_seed(SEED)
    batches = torch.randint(len(bank["source"]), (UPDATES, BATCH), generator=rng)
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    for left, right in (("two_slot_return", "two_slot_supervised"), ("affine_return", "affine_supervised")):
        if initial[left] != initial[right]:
            raise ValueError("the matched residue fits changed their initialization")
    for arm, model in models.items():
        train(model, arm, data, batches)
        print("fitted", arm, flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    checkpoint = {"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial,
                  "weight_sha": {arm: tensor_hash(value) for arm, value in weights.items()},
                  "sampler_sha": tensor_hash({"batches": batches}), "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(checkpoint, handle)
    print("sealed residue policy", sha256_file(args.fit), flush=True)


def distribution(logits, support):
    return logits.masked_fill(~support, -torch.inf).softmax(-1).tolist()


def intervene(pair, allowance, support, logits, baseline):
    case = pair["second"]
    mapped = route_of(case)
    values = distribution(logits, support)
    corrected = planned_values(case["routes"], world_weights(case, values, WORLDS))
    anchor = planned_values(case["routes"], world_weights(case, baseline, WORLDS))
    chosen, old = stop_or_pick(corrected(allowance, 0)), stop_or_pick(anchor(allowance, 0))
    eligible = {i: logits[i].item() for i, allowed in enumerate(support.tolist()) if allowed}
    if (old is None or chosen is None or old == chosen or len(eligible) < 3
            or any(not any(mapped[i] == arm for i in eligible) for arm in (chosen, old))):
        return {arm: (values, None) for arm in INTERVENTIONS}
    a, b = (max((i for i in eligible if mapped[i] == arm), key=eligible.__getitem__) for arm in (chosen, old))
    output = {}
    for arm in INTERVENTIONS:
        sign = None if arm == INTERVENTIONS[0] else (1 if int(pair["pair"][-1], 16) % 2 else -1)
        edited, dose = manipulate(eligible, a, b, sham_sign=sign)
        perturbed = logits.clone()
        for index, value in edited.items():
            perturbed[index] = value
        output[arm] = distribution(perturbed, support), {"l2_dose": dose, "residues": [a, b]}
    return output


@torch.no_grad()
def evaluate(args, tree):
    bank, pins = locked(args)
    fit = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fit["schema"] != SCHEMA or fit["pins"] != pins:
        raise ValueError("the locked residue fit changed")
    models = models_for()
    for arm, model in models.items():
        weights = fit["weights"][arm]
        if tensor_hash(weights) != fit["weight_sha"][arm]:
            raise ValueError("a residue policy changed after fitting")
        model.load_state_dict(weights)
        model.eval()
    prior = MixedSourcePrior(args)
    records, cases = defaultdict(list), []
    for pair in bank["development"]:
        a, b = pair["first"], pair["second"]
        cases.extend((a, b))
        choice, observed, first = source_trial(a)
        features, allowed = public_tokens(pair, choice, observed, curves(a, prior), curves(b, prior))
        tokens, support = torch.tensor(features), torch.tensor(allowed)
        scores = {arm: model(tokens).flatten() for arm, model in models.items()}
        baseline = distribution(torch.zeros(MODULUS), support)
        forecasts = {arm: distribution(value, support) for arm, value in scores.items()}
        forecasts.update({"uniform_reset": [1. / MODULUS] * MODULUS, "uniform_conditioned": baseline,
                          "actual_residue": [float(pair["residue"] == i) for i in range(MODULUS)]})
        tags = {"pair": pair["pair"], "module": a["module"], "family": pair["family"],
                "first_module": a["module"], "second_module": b["module"]}
        for allowance in ALLOWANCES:
            trial = {arm: (weights, None) for arm, weights in forecasts.items()}
            trial.update(intervene(pair, allowance, support, scores["two_slot_return"], baseline))
            patches = [trial[arm][1] for arm in INTERVENTIONS]
            if (patches[0] is None) != (patches[1] is None) or patches[0] and patches[0]["l2_dose"] != patches[1]["l2_dose"]:
                raise ValueError("the residue interventions disagree on eligibility or physical dose")
            for arm, (weights, patch) in trial.items():
                second = solve_target(b, allowance, world_weights(b, weights, WORLDS))
                inferred = max(range(MODULUS), key=weights.__getitem__)
                records[arm].append({**tags, "allowance": allowance, "first": first, "second": second,
                                     "proofs": int(first["proved"]) + int(second["proved"]),
                                     "checks": first["checks"] + second["checks"], "utility": second["utility"],
                                     "patch": patch, "residue_correct": inferred == pair["residue"]})
            if args.acquisition_diagnostic:
                planned_first, planned_weights = joint_uniform_first(a, b, allowance)
                planned_second = solve_target(b, allowance, world_weights(b, planned_weights, WORLDS))
                records["joint_uniform_diagnostic"].append(
                    {**tags, "allowance": allowance, "first": planned_first, "second": planned_second,
                     "proofs": int(planned_first["proved"]) + int(planned_second["proved"]),
                     "checks": planned_first["checks"] + planned_second["checks"], "utility": planned_second["utility"],
                     "patch": None, "residue_correct": None,
                     "changed_first_choice": planned_first["trace"] != first["trace"]})
    audits = ("two_slot_return", "two_slot_supervised", "trm_return", *INTERVENTIONS)
    if args.acquisition_diagnostic:
        audits += ("joint_uniform_diagnostic",)
    verified, live = replays(args, cases, records, audits, patched_arms=INTERVENTIONS)
    main = records["two_slot_return"]
    contrasts = {arm: root_contrast(main, rows) for arm, rows in records.items() if rows is not main}
    summary = {arm: {**summarize(rows), "by_budget": {budget: summarize([x for x in rows if x["allowance"] == budget])
                                                    for budget in ALLOWANCES}} for arm, rows in records.items()}
    for arm, row in summary.items():
        print(arm, row["second_proofs"], row["checks"], row["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "floor": FLOOR, "summary": summary, "root_contrasts": contrasts,
                                 "results": records, "independent": verified, "live_replays": live,
                                 "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "fit", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--acquisition-diagnostic", action="store_true",
                        help="also score the post-hoc uniform planner that anticipates the second proof")
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
