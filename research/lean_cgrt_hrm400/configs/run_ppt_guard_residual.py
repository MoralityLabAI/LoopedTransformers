"""Compare source-trained bounded corrections with an intact guard planner."""

import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.freeze_ppt_guard_recovery import certify_routes, replay_recovery
from configs.freeze_ppt_native_progress import native_sources
from configs.run_ppt_guard_fork_fit import truth_head
from configs.run_ppt_guard_recovery import SCHEMA as BASE_SCHEMA
from configs.run_ppt_guard_recovery import encode_case, fit_direct, registered, summary
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.guard_planner_residual import RADIUS, BoundedCorrection, hindsight_policy, packed_action
from ppt_gym.guard_recovery import ALLOWANCES, feasible_menu
from ppt_gym.guard_recovery_learning import baked_scores, source_targets, trace_policy
from ppt_gym.live_quote_policy import LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.guard-planner-residual.v1"
SEED = 1109
FOLDS = 4
UPDATES = 600
BATCH = 32
TEMPERATURES = (.35, .5, .75, 1., 1.5, 2., 3.)


def load_inputs(args):
    bank, pins = registered(args)
    planner = torch.load(args.planner, map_location="cpu", weights_only=True)
    if planner["schema"] != BASE_SCHEMA or planner["pins"] != pins:
        raise ValueError("the frozen supervised guard planner changed")
    state = planner["weights"]["guard_reader"]
    if tensor_hash(state) != planner["weight_sha"]["guard_reader"]:
        raise ValueError("the frozen direct guard reader changed")
    reader = truth_head()
    reader.load_state_dict(state)
    return bank, {**pins, "planner": sha256_file(args.planner)}, reader.eval()


def models_for():
    model = BoundedCorrection(TwoSlotQuoteActor())
    return {"bellman_residual": model, "empirical_residual": deepcopy(model),
            "trm_residual": BoundedCorrection(LiveQuoteTRM())}


@torch.no_grad()
def scaled(probability, temperature):
    return (torch.tensor(probability).logit() / temperature).sigmoid().clamp(.02, .98).tolist()


def cross_fit(source, prior, rng):
    folds = {pair: index % FOLDS for index, pair in enumerate(sorted({case["pair"] for case in source}))}
    cache = {}
    predictions, truth = [], []
    for fold in range(FOLDS):
        reader, _ = fit_direct([case for case in source if folds[case["pair"]] != fold], rng)
        observed = {}
        for case in source:
            if folds[case["pair"]] != fold:
                continue
            descriptors, probability, warm = encode_case(case, reader, prior)
            cache[case["module"]] = (descriptors, probability, warm)
            for row, p, bit in zip(descriptors, probability, case["truth"], strict=True):
                # A closed guard contributes once to this fold's reliability fit.
                observed.setdefault(tuple(row), (p, bit))
        for p, bit in observed.values():
            predictions.append(p)
            truth.append(bit)
    target = torch.tensor(truth, dtype=torch.float32)
    loss = {str(t): float(F.binary_cross_entropy(torch.tensor(scaled(predictions, t)), target)) for t in TEMPERATURES}
    chosen = min(TEMPERATURES, key=lambda t: loss[str(t)])
    return cache, folds, chosen, loss


def prepare(cases, cache, folds):
    x, offsets, backed, empirical, groups = [], [], [], [], []
    contrasts = []
    for case in cases:
        descriptors, probability, warm = cache[case["module"]]
        values = baked_scores(probability, action_space=feasible_menu)
        comparable = defaultdict(list)
        for state, move, long_value, short_value in source_targets(case):
            scores = values(state)
            if move not in scores:
                continue
            base = scores[move]
            maximum = max(scores.values())
            if base + RADIUS >= max(0., maximum - RADIUS):
                comparable[state].append(len(x))
            x.append(packed_action(descriptors, probability, warm, state, move, base, maximum))
            offsets.append(base)
            backed.append(long_value)
            empirical.append(short_value)
            groups.append(folds[case["pair"]])
        contrasts.extend(pair for members in comparable.values() for pair in combinations(members, 2))
        values.cache_clear()
    return tuple(torch.tensor(values) for values in (x, offsets, backed, empirical, groups, contrasts))


def train(model, rows, offsets, targets, batches, *, pairwise=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for indices in batches:
        if pairwise:
            flat = indices.flatten()
            scores = model(rows[flat], offsets[flat]).reshape(-1, 2)
            loss = F.mse_loss(scores[:, 0] - scores[:, 1], targets[indices[:, 0]] - targets[indices[:, 1]])
        else:
            scores = model(rows[indices], offsets[indices])
            loss = F.mse_loss(scores, targets[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    return model.eval()


def fit(args, tree):
    bank, pins, _ = load_inputs(args)
    source = bank["source"]
    prior = MixedSourcePrior(args)
    rng = torch.Generator().manual_seed(SEED)
    cache, folds, temperature, calibration_scores = cross_fit(source, prior, rng)
    rows, base, backed, empirical, groups, contrasts = prepare(source, cache, folds)
    if args.pairwise:
        batches = contrasts[torch.randint(len(contrasts), (UPDATES, BATCH), generator=rng)]
    else:
        batches = torch.randint(len(rows), (UPDATES, BATCH), generator=rng)
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if initial["bellman_residual"] != initial["empirical_residual"]:
        raise ValueError("the residual objectives do not have matched initial weights")
    for arm, model in models.items():
        targets = empirical if arm == "empirical_residual" else backed
        train(model, rows, base, targets, batches, pairwise=args.pairwise)
        print("fitted", arm, flush=True)
    if args.pairwise:
        a, b = contrasts.unbind(1)
        gap = (rows[a] - rows[b]).flatten(1)
        advantage = ((backed[a] - base[a]) - (backed[b] - base[b])).clamp(-2 * RADIUS, 2 * RADIUS)
        calibration = ridge(torch.cat((gap, -gap)), torch.cat((advantage, -advantage)), groups[a].repeat(2))
    else:
        calibration = ridge(rows.flatten(1), (backed - base).clamp(-RADIUS, RADIUS), groups)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    with args.fit.open("xb") as handle:
        torch.save({"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial,
                    "weight_sha": {arm: tensor_hash(value) for arm, value in weights.items()},
                    "sampler_sha": tensor_hash({"batches": batches}), "calibration": calibration,
                    "temperature": temperature, "temperature_losses": calibration_scores,
                    "decisions": len(rows), "contrasts": len(contrasts), "pairwise": args.pairwise,
                    "radius": RADIUS, "memory": tree.stats()}, handle)
    print("sealed residual fit", sha256_file(args.fit), flush=True)


def action_choice(values, menu):
    best = int(values.argmax())
    return menu[best] if float(values[best]) > 0. else None


@torch.no_grad()
def evaluate(args, tree):
    bank, pins, reader = load_inputs(args)
    fitted = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fitted["schema"] != SCHEMA or fitted["pins"] != pins or fitted["radius"] != RADIUS:
        raise ValueError("the residual fit changed its source or registered dose")
    models = models_for()
    for arm, model in models.items():
        state = fitted["weights"][arm]
        if tensor_hash(state) != fitted["weight_sha"][arm]:
            raise ValueError("the frozen residual actor changed")
        model.load_state_dict(state)
        model.eval()
    prior = MixedSourcePrior(args)
    records, certificates = defaultdict(list), {}
    arms = (*models, "frozen_planner", "temperature_calibration", "ridge_residual", "swap_residual", "bounded_hindsight")
    for case in bank["development"]:
        descriptors, probability, warm = encode_case(case, reader, prior)
        values = baked_scores(probability, action_space=feasible_menu)
        temperature = baked_scores(scaled(probability, fitted["temperature"]), action_space=feasible_menu)
        upper_bound = hindsight_policy(values, case["truth"])
        tags = {"module": case["module"], "pair": case["pair"], "family": case["family"]}
        for arm in arms:
            for allowance in ALLOWANCES:
                patch = None
                changes = []

                def select(state, menu):
                    nonlocal patch
                    if arm == "bounded_hindsight":
                        return upper_bound(state)
                    scores = (temperature if arm == "temperature_calibration" else values)(state)
                    base = torch.tensor([scores[move] for move in menu], dtype=torch.float64)
                    base_pick = action_choice(base, menu)
                    if arm in ("frozen_planner", "temperature_calibration"):
                        return base_pick
                    x = torch.tensor([packed_action(descriptors, probability, warm, state, move, scores[move],
                                                    max(scores.values())) for move in menu])
                    if arm == "ridge_residual":
                        residual = apply_ridge(x.flatten(1), fitted["calibration"]).clamp(-RADIUS, RADIUS)
                    else:
                        model = models["bellman_residual" if arm == "swap_residual" else arm]
                        residual = model.residual(x)
                    pick = action_choice(base + residual, menu)
                    if pick != base_pick:
                        step = allowance - 2 - state.remaining
                        changes.append({"step": step, "base": base_pick, "corrected": pick})
                        if arm == "swap_residual" and patch is None and pick is not None and base_pick is not None:
                            a, b = menu.index(base_pick), menu.index(pick)
                            altered = residual.clone()
                            altered[[a, b]] = residual[[b, a]]
                            patch = step
                            return action_choice(base + altered, menu)
                    return pick

                row = trace_policy(case, allowance, select, action_space=feasible_menu)
                records[arm].append({**tags, **row, "residual_changes": changes, "swap_at": patch})
                if row["proved"]:
                    certificates[case["module"], tuple(row["proof"])] = case
        values.cache_clear()
        temperature.cache_clear()
    natural = {(row["module"], row["allowance"]): row for row in records["bellman_residual"]}
    for donor in records["swap_residual"]:
        row = natural[donor["module"], donor["allowance"]]
        step = donor["swap_at"]
        if step is None and row["trace"] != donor["trace"]:
            raise ValueError("the residual sham changed an unpatched trajectory")
        if step is not None and row["trace"][:step] != donor["trace"][:step]:
            raise ValueError("a residual intervention changed its physical prefix")
    with native_sources((args.bpd, args.mathcan)) as (directory, env, imports):
        independent = certify_routes(directory, env, imports, [(case, key[1]) for key, case in certificates.items()])
        sample = {}
        for case in bank["development"]:
            sample[case["module"]] = case
        replays = []
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native Lean recovery toolchain")
            for arm in ("bellman_residual", "empirical_residual", "swap_residual"):
                selected = next((row for row in records[arm] if row["proved"] and row["residual_changes"]), None)
                if selected is not None:
                    replay_recovery(repl, sample[selected["module"]], selected)
                    replays.append({"arm": arm, "module": selected["module"], "allowance": selected["allowance"]})
    summaries = {arm: summary(rows) for arm, rows in records.items()}
    for arm, row in summaries.items():
        print(arm, row, flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "results": records, "summary": summaries, "independent": independent,
                                 "live_replays": replays, "pairwise": fitted.get("pairwise", False),
                                 "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    parser.add_argument("--pairwise", action="store_true", help="fit action contrasts inside the public correction bound")
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "planner", "fit", "results"):
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
